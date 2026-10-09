"""Capture the reported MuJoCo-Warp constraint allocation without Isaac Lab."""

import argparse
import ctypes
import faulthandler
import gc
import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np

import warp as wp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=8192)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--mode", choices=("thread_local", "relaxed"), default="thread_local")
    parser.add_argument("--nonblocking", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    faulthandler.enable()
    faulthandler.dump_traceback_later(90, repeat=True)
    wp.config.kernel_cache_dir = str(Path(__file__).parent / "kernel_cache" / wp.__version__)
    wp.init()
    with wp.ScopedDevice("cuda:0") as device:
        model_host = mujoco.MjModel.from_xml_string("""
            <mujoco><option timestep="0.002" solver="Newton" iterations="10"/>
              <worldbody><geom type="plane" size="10 10 0.1"/>
                <body pos="0 0 0.1"><freejoint/>
                  <geom type="sphere" size="0.1" mass="1"/>
                </body>
              </worldbody>
            </mujoco>""")
        model = mjw.put_model(model_host)
        data = mjw.make_data(model_host, nworld=args.worlds, nconmax=8, njmax=32)
        stream = wp.get_stream(device)
        if args.nonblocking:
            driver = ctypes.WinDLL("nvcuda.dll")
            driver.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
            driver.cuStreamCreate.restype = ctypes.c_int
            handle = ctypes.c_void_p()
            status = driver.cuStreamCreate(ctypes.byref(handle), 1)
            if status:
                raise RuntimeError(f"cuStreamCreate returned {status}")
            stream = wp.Stream(device, cuda_stream=handle.value)
        if args.warmup:
            print("Eager warmup", flush=True)
            mjw.step(model, data)
            wp.synchronize_device(device)
        start_time = data.time.numpy().copy()
        print(
            json.dumps(
                {"stage": "capture_start", "warp": wp.__version__, "mujoco": mujoco.__version__, "args": vars(args)}
            ),
            flush=True,
        )
        gc.disable()
        try:
            for iteration in range(args.rounds):
                with wp.ScopedStream(stream):
                    with wp.ScopedCapture(
                        stream=stream, capture_mode=getattr(wp.CaptureMode, args.mode.upper())
                    ) as capture:
                        mjw.step(model, data)
                    wp.capture_launch(capture.graph)
                    wp.synchronize_device(device)
                del capture
                if iteration % 10 == 0:
                    print(json.dumps({"stage": "progress", "round": iteration + 1}), flush=True)
        finally:
            gc.enable()
        np.testing.assert_allclose(data.time.numpy(), start_time + 0.002 * args.rounds, atol=1e-6)
        if not np.isfinite(data.qpos.numpy()).all():
            raise RuntimeError("MuJoCo positions became nonfinite")
        print(json.dumps({"stage": "passed", "captures": args.rounds, "worlds": args.worlds}), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
