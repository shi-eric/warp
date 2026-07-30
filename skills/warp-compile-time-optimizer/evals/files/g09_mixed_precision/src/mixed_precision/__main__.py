# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start the service: calibrate once, then serve one request batch."""

from __future__ import annotations

import numpy as np

from .service import calibrate, serve


def main() -> None:
    """Run startup calibration followed by one served batch."""
    rng = np.random.default_rng(0)
    fitted = calibrate(
        rng.random((64, 64), dtype=np.float32),
        rng.random((64, 64), dtype=np.float32),
    )
    served = serve(
        rng.random((512, 512)).astype(np.float16),
        rng.random((512, 512)).astype(np.float16),
    )
    print(f"calibrated {fitted.shape} sum {fitted.sum():.3f}")
    print(f"served {served.shape} sum {np.asarray(served, dtype=np.float32).sum():.1f}")


if __name__ == "__main__":
    main()
