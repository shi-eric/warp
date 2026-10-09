"""Check captured allocations while another CUDA stream has pending host work."""

import argparse
import ctypes
import faulthandler
import threading
import time
from pathlib import Path

import warp as wp


@wp.kernel
def prefix_kernel(output: wp.array[wp.int32]):
    index = wp.tid()
    output[index] = index


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("thread_local", "relaxed"), default="thread_local")
parser.add_argument("--nonblocking", action="store_true")
parser.add_argument("--raw", action="store_true")
parser.add_argument("--kernel-first", action="store_true")
args = parser.parse_args()
faulthandler.enable()
faulthandler.dump_traceback_later(30, repeat=True)
wp.config.kernel_cache_dir = str(Path(__file__).parent / "kernel_cache" / wp.__version__)
wp.init()
with wp.ScopedDevice("cuda:0") as device:
    driver = ctypes.WinDLL("nvcuda.dll")
    callback_type = ctypes.WINFUNCTYPE(None, ctypes.c_void_p)
    driver.cuLaunchHostFunc.argtypes = [ctypes.c_void_p, callback_type, ctypes.c_void_p]
    driver.cuLaunchHostFunc.restype = ctypes.c_int
    driver.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    driver.cuStreamCreate.restype = ctypes.c_int
    driver.cuMemAllocAsync.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_void_p]
    driver.cuMemAllocAsync.restype = ctypes.c_int
    driver.cuMemFreeAsync.argtypes = [ctypes.c_uint64, ctypes.c_void_p]
    driver.cuMemFreeAsync.restype = ctypes.c_int
    stream = wp.get_stream(device)
    if args.nonblocking:
        handle = ctypes.c_void_p()
        status = driver.cuStreamCreate(ctypes.byref(handle), 1)
        if status:
            raise RuntimeError(f"cuStreamCreate returned {status}")
        stream = wp.Stream(device, cuda_stream=handle.value)
    entered = threading.Event()
    release = threading.Event()
    prefix_output = wp.empty(8192 * 40, dtype=wp.int32, device=device)
    if args.kernel_first:
        wp.launch(prefix_kernel, dim=prefix_output.size, inputs=[prefix_output], device=device)
        wp.synchronize_device(device)

    @callback_type
    def callback(_):
        entered.set()
        release.wait(15)

    status = driver.cuLaunchHostFunc(None, callback, None)
    if status:
        raise RuntimeError(f"cuLaunchHostFunc returned {status}")
    if not entered.wait(5):
        raise RuntimeError("CUDA host callback did not start")
    timer = threading.Timer(10, release.set)
    timer.start()
    try:
        print("Capturing with pending legacy-stream callback", flush=True)
        begin = time.monotonic()
        with wp.ScopedStream(stream):
            with wp.ScopedCapture(stream=stream, capture_mode=getattr(wp.CaptureMode, args.mode.upper())) as capture:
                print(f"capture_begin returned in {time.monotonic() - begin:.3f}s", flush=True)
                if args.kernel_first:
                    wp.launch(prefix_kernel, dim=prefix_output.size, inputs=[prefix_output], device=device)
                    print("Kernel recorded before allocation", flush=True)
                begin = time.monotonic()
                if args.raw:
                    ptr = ctypes.c_uint64()
                    status = driver.cuMemAllocAsync(ctypes.byref(ptr), 32768, stream.cuda_stream)
                    if status:
                        raise RuntimeError(f"cuMemAllocAsync returned {status}")
                    status = driver.cuMemFreeAsync(ptr.value, stream.cuda_stream)
                    if status:
                        raise RuntimeError(f"cuMemFreeAsync returned {status}")
                else:
                    temporary = wp.empty(8192, dtype=int, device=device)
                    temporary.zero_()
                    del temporary
                print(
                    f"Allocation returned in {time.monotonic() - begin:.3f}s; callback released={release.is_set()}",
                    flush=True,
                )
            print("Capture ended", flush=True)
        release.set()
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
    finally:
        release.set()
        timer.cancel()
faulthandler.cancel_dump_traceback_later()
print("Passed", flush=True)
