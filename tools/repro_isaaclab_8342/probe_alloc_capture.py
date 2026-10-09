"""Probe graph allocations for Isaac Lab issue 8342 on Windows."""

import argparse
import ctypes
import faulthandler
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np

import warp as wp


@wp.kernel
def add_one(values: wp.array[float]):
    i = wp.tid()
    values[i] += 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("thread_local", "relaxed", "global"), default="thread_local")
    parser.add_argument("--stream", choices=("default", "new", "nonblocking"), default="default")
    parser.add_argument("--legacy-work", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--allocations", type=int, default=32)
    parser.add_argument("--elements", type=int, default=1024 * 1024)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--kernel-first", action="store_true")
    parser.add_argument("--pressure-mib", type=int, default=0)
    parser.add_argument("--torch", action="store_true")
    args = parser.parse_args()
    faulthandler.enable()
    faulthandler.dump_traceback_later(45, repeat=True)
    wp.config.kernel_cache_dir = str(Path(__file__).parent / "kernel_cache" / wp.__version__)
    wp.init()
    device = wp.get_device("cuda:0")
    if args.torch:
        import torch  # noqa: PLC0415

        torch_buffer = torch.zeros(8192, dtype=torch.int32, device="cuda:0")
        print("Torch:", torch.__version__, "buffer elements:", torch_buffer.numel(), flush=True)
    driver = ctypes.WinDLL("nvcuda.dll")
    driver.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    driver.cuCtxSetCurrent.restype = ctypes.c_int
    status = driver.cuCtxSetCurrent(device.context)
    if status:
        raise RuntimeError(f"cuCtxSetCurrent returned {status}")
    if args.stream == "nonblocking":
        driver.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        driver.cuStreamCreate.restype = ctypes.c_int
        handle = ctypes.c_void_p()
        status = driver.cuStreamCreate(ctypes.byref(handle), 1)
        if status:
            raise RuntimeError(f"cuStreamCreate returned {status}")
        stream = wp.Stream(device, cuda_stream=handle.value)
    else:
        stream = wp.get_stream(device) if args.stream == "default" else wp.Stream(device)
    legacy_stream = wp.Stream(device, cuda_stream=0)
    values = wp.zeros(1024, dtype=float, device=device)
    background = wp.zeros(8 * 1024 * 1024, dtype=float, device=device)
    wp.load_module(device=device)
    pressure = None
    if args.pressure_mib:
        pressure = wp.zeros((args.pressure_mib, 1024, 1024), dtype=wp.uint8, device=device)
        wp.synchronize_device(device)
    print(
        json.dumps(
            {
                "stage": "initialized",
                "warp": wp.__version__,
                "source": wp.__file__,
                "platform": platform.platform(),
                "device": str(device),
                "gpu": device.name,
                "mempool": device.is_mempool_enabled,
                "pressure_bytes": pressure.size if pressure is not None else 0,
                "stream": stream.cuda_stream,
                "args": vars(args),
            }
        ),
        flush=True,
    )
    driver.cuStreamGetFlags.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    driver.cuStreamGetFlags.restype = ctypes.c_int
    flags = ctypes.c_uint()
    result = driver.cuStreamGetFlags(stream.cuda_stream, ctypes.byref(flags))
    if result:
        raise RuntimeError(f"cuStreamGetFlags returned {result}")
    print(f"CUDA stream flags: {flags.value} (1 means nonblocking)", flush=True)
    driver.cuMemAllocAsync.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_void_p]
    driver.cuMemAllocAsync.restype = ctypes.c_int
    driver.cuMemFreeAsync.argtypes = [ctypes.c_uint64, ctypes.c_void_p]
    driver.cuMemFreeAsync.restype = ctypes.c_int
    if args.warmup:
        warmup = wp.empty(args.elements, dtype=float, device=device)
        del warmup
        wp.launch(add_one, 1024, inputs=[values], device=device)
        wp.synchronize_device(device)
        values.zero_()
    started = time.monotonic()
    gc.disable()
    try:
        for iteration in range(args.rounds):
            if args.legacy_work:
                for _ in range(8):
                    wp.launch(add_one, background.size, inputs=[background], stream=legacy_stream)
            retained = []
            with wp.ScopedStream(stream):
                with wp.ScopedCapture(
                    stream=stream, capture_mode=getattr(wp.CaptureMode, args.mode.upper())
                ) as capture:
                    if args.kernel_first:
                        wp.launch(add_one, 1024, inputs=[values], device=device)
                    for allocation in range(args.allocations):
                        if args.raw:
                            ptr = ctypes.c_uint64()
                            status = driver.cuMemAllocAsync(
                                ctypes.byref(ptr), 4 * (args.elements + allocation * 64), stream.cuda_stream
                            )
                            if status:
                                raise RuntimeError(f"cuMemAllocAsync returned {status}")
                            status = driver.cuMemFreeAsync(ptr.value, stream.cuda_stream)
                            if status:
                                raise RuntimeError(f"cuMemFreeAsync returned {status}")
                        else:
                            temporary = wp.empty(args.elements + allocation * 64, dtype=float, device=device)
                            temporary.fill_(float(allocation))
                            if allocation % 2:
                                retained.append(temporary)
                            del temporary
                    if not args.kernel_first:
                        wp.launch(add_one, 1024, inputs=[values], device=device)
                wp.capture_launch(capture.graph)
                wp.capture_launch(capture.graph)
                wp.synchronize_device(device)
            retained.clear()
            del capture
            if iteration % 10 == 0:
                print(json.dumps({"stage": "progress", "round": iteration + 1}), flush=True)
    finally:
        gc.enable()
    np.testing.assert_array_equal(values.numpy(), np.full(1024, 2 * args.rounds, dtype=np.float32))
    print(
        json.dumps(
            {
                "stage": "passed",
                "captures": args.rounds,
                "allocations": args.rounds * args.allocations,
                "seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
