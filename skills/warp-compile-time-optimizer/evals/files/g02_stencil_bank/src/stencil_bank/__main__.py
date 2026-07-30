# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a deterministic procedural stencil example."""

from __future__ import annotations

import numpy as np

from .analysis import StencilBank


def main() -> None:
    """Print a small CUDA stencil analysis."""
    values = np.linspace(np.float32(-1.0), np.float32(1.0), 9, dtype=np.float32)
    print(StencilBank((1, 2), device="cuda:0").analyze(values))


if __name__ == "__main__":
    main()
