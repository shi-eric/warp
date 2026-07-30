# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a deterministic PDE stepping example."""

from __future__ import annotations

import numpy as np

import warp as wp

from .solver import PDEStepper


def main() -> None:
    """Print one representative periodic finite-difference solution."""
    rows, columns = np.indices((9, 11), dtype=np.float32)
    field = np.sin(np.float32(0.4) * rows) + np.cos(np.float32(0.3) * columns)
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    print(PDEStepper("approximate", "periodic", device=device).step(field, iterations=4))


if __name__ == "__main__":
    main()
