# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate pointwise pipeline with one launch and no intermediates."""

from collections.abc import Callable

import numpy as np

import warp as wp
from warp.examples.optimizations.harness import Variant


@wp.kernel
def fused_pipeline(
    x: wp.array[float],
    bias: wp.array[float],
    result: wp.array[float],
):
    index = wp.tid()
    affine = x[index] * 1.25 + bias[index]
    bounded = wp.tanh(affine)
    result[index] = bounded * bounded + 0.1 * bounded


def build_variant(
    *,
    x_values: np.ndarray,
    bias_values: np.ndarray,
    iterations: int,
    device: str,
    synchronize: Callable[[], None],
) -> Variant:
    """Build the one-launch candidate with stable array allocations."""

    size = len(x_values)
    x = wp.array(x_values, dtype=wp.float32, device=device)
    bias = wp.array(bias_values, dtype=wp.float32, device=device)
    result = wp.empty(size, dtype=wp.float32, device=device)
    reset = wp.zeros(size, dtype=wp.float32, device=device)

    def prepare_trial() -> None:
        wp.copy(result, reset)

    def run() -> None:
        for _ in range(iterations):
            wp.launch(fused_pipeline, dim=size, inputs=[x, bias, result], device=device)

    def outputs() -> dict[str, np.ndarray]:
        return {"result": result.numpy()}

    return Variant(
        label="one-launch candidate",
        prepare_trial=prepare_trial,
        run=run,
        synchronize=synchronize,
        outputs=outputs,
    )
