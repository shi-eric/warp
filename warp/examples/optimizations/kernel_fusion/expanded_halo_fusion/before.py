# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseline separable stencil with one full-size intermediate."""

from collections.abc import Callable

import numpy as np

import warp as wp
from warp.examples.optimizations.harness import Variant


@wp.kernel
def horizontal_pass(
    source: wp.array2d[float],
    weights: wp.array[float],
    radius: int,
    width: int,
    intermediate: wp.array2d[float],
):
    row, column = wp.tid()
    value = float(0.0)
    for offset in range(-radius, radius + 1):
        source_column = wp.clamp(column + offset, 0, width - 1)
        value = value + source[row, source_column] * weights[offset + radius]
    intermediate[row, column] = value


@wp.kernel
def vertical_pass(
    intermediate: wp.array2d[float],
    weights: wp.array[float],
    radius: int,
    height: int,
    result: wp.array2d[float],
):
    row, column = wp.tid()
    value = float(0.0)
    for offset in range(-radius, radius + 1):
        source_row = wp.clamp(row + offset, 0, height - 1)
        value = value + intermediate[source_row, column] * weights[offset + radius]
    result[row, column] = value


def build_variant(
    *,
    input_values: np.ndarray,
    weights_values: np.ndarray,
    iterations: int,
    radius: int,
    device: str,
    synchronize: Callable[[], None],
) -> Variant:
    """Build the two-pass separable baseline with stable allocations."""

    height, width = input_values.shape
    weights = wp.array(weights_values, dtype=wp.float32, device=device)
    reset = wp.array2d(input_values, dtype=wp.float32, device=device)
    state_a = wp.array2d(input_values, dtype=wp.float32, device=device)
    state_b = wp.empty((height, width), dtype=wp.float32, device=device)
    intermediate = wp.empty((height, width), dtype=wp.float32, device=device)
    final_state = state_a if iterations % 2 == 0 else state_b

    def prepare_trial() -> None:
        wp.copy(state_a, reset)

    def run() -> None:
        source = state_a
        result = state_b
        for _ in range(iterations):
            wp.launch(
                horizontal_pass,
                dim=(height, width),
                inputs=[source, weights, radius, width],
                outputs=[intermediate],
                device=device,
            )
            wp.launch(
                vertical_pass,
                dim=(height, width),
                inputs=[intermediate, weights, radius, height],
                outputs=[result],
                device=device,
            )
            source, result = result, source

    def outputs() -> dict[str, np.ndarray]:
        return {"result": final_state.numpy()}

    return Variant(
        label="separable two-pass baseline",
        prepare_trial=prepare_trial,
        run=run,
        synchronize=synchronize,
        outputs=outputs,
    )
