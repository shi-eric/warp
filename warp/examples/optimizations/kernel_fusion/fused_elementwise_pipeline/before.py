# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseline pointwise pipeline with separate launches and intermediates."""

from collections.abc import Callable

import numpy as np

import warp as wp
from warp.examples.optimizations.harness import Variant


@wp.kernel
def affine_stage(
    x: wp.array[float],
    bias: wp.array[float],
    intermediate: wp.array[float],
):
    index = wp.tid()
    intermediate[index] = x[index] * 1.25 + bias[index]


@wp.kernel
def bounded_stage(
    intermediate: wp.array[float],
    bounded: wp.array[float],
):
    index = wp.tid()
    bounded[index] = wp.tanh(intermediate[index])


@wp.kernel
def polynomial_stage(
    bounded: wp.array[float],
    result: wp.array[float],
):
    index = wp.tid()
    value = bounded[index]
    result[index] = value * value + 0.1 * value


def build_variant(
    *,
    x_values: np.ndarray,
    bias_values: np.ndarray,
    iterations: int,
    device: str,
    synchronize: Callable[[], None],
) -> Variant:
    """Build the three-launch baseline with stable array allocations."""

    size = len(x_values)
    x = wp.array(x_values, dtype=wp.float32, device=device)
    bias = wp.array(bias_values, dtype=wp.float32, device=device)
    intermediate = wp.empty(size, dtype=wp.float32, device=device)
    bounded = wp.empty(size, dtype=wp.float32, device=device)
    result = wp.empty(size, dtype=wp.float32, device=device)
    reset = wp.zeros(size, dtype=wp.float32, device=device)

    def prepare_trial() -> None:
        wp.copy(result, reset)

    def run() -> None:
        for _ in range(iterations):
            wp.launch(affine_stage, dim=size, inputs=[x, bias, intermediate], device=device)
            wp.launch(bounded_stage, dim=size, inputs=[intermediate, bounded], device=device)
            wp.launch(polynomial_stage, dim=size, inputs=[bounded, result], device=device)

    def outputs() -> dict[str, np.ndarray]:
        return {"result": result.numpy()}

    return Variant(
        label="three-launch baseline",
        prepare_trial=prepare_trial,
        run=run,
        synchronize=synchronize,
        outputs=outputs,
    )
