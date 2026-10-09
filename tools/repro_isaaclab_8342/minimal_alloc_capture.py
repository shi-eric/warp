"""Probe the 32 KiB captured allocation reported in Isaac Lab issue 8342.

This passed locally; it is a candidate to run on an affected machine, not a
confirmed reproduction of the camera hang.
"""

import faulthandler
from pathlib import Path

import numpy as np

import warp as wp

faulthandler.enable()
faulthandler.dump_traceback_later(30, repeat=True)
wp.config.kernel_cache_dir = str(Path(__file__).parent / "kernel_cache")
wp.init()

with wp.ScopedDevice("cuda:0"):
    for iteration in range(100):
        print(f"Capture {iteration + 1}: before allocation", flush=True)
        with wp.ScopedCapture() as capture:
            # make_constraint() allocates nworld int32 entries on every step.
            temporary = wp.empty(8192, dtype=wp.int32)
            temporary.fill_(42)
        print(f"Capture {iteration + 1}: after allocation", flush=True)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(temporary.numpy(), np.full(8192, 42, dtype=np.int32))
        del temporary, capture

faulthandler.cancel_dump_traceback_later()
print("Passed: 100 captures", flush=True)
