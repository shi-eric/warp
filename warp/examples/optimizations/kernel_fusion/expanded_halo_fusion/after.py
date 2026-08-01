# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate product stencil with an expanded direct-load halo."""

from collections.abc import Callable

import numpy as np

import warp as wp
from warp.examples.optimizations.harness import Variant


@wp.kernel
def product_stencil(
    source: wp.array2d[float],
    weights: wp.array[float],
    radius: int,
    height: int,
    width: int,
    result: wp.array2d[float],
):
    row, column = wp.tid()
    value = float(0.0)
    for row_offset in range(-radius, radius + 1):
        source_row = wp.clamp(row + row_offset, 0, height - 1)
        row_weight = weights[row_offset + radius]
        for column_offset in range(-radius, radius + 1):
            source_column = wp.clamp(column + column_offset, 0, width - 1)
            column_weight = weights[column_offset + radius]
            value = value + source[source_row, source_column] * row_weight * column_weight
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
    """Build the one-pass candidate with stable allocations."""

    height, width = input_values.shape
    weights = wp.array(weights_values, dtype=wp.float32, device=device)
    reset = wp.array2d(input_values, dtype=wp.float32, device=device)
    state_a = wp.array2d(input_values, dtype=wp.float32, device=device)
    state_b = wp.empty((height, width), dtype=wp.float32, device=device)
    final_state = state_a if iterations % 2 == 0 else state_b

    def prepare_trial() -> None:
        wp.copy(state_a, reset)

    def run() -> None:
        source = state_a
        result = state_b
        for _ in range(iterations):
            wp.launch(
                product_stencil,
                dim=(height, width),
                inputs=[source, weights, radius, height, width],
                outputs=[result],
                device=device,
            )
            source, result = result, source

    def outputs() -> dict[str, np.ndarray]:
        return {"result": final_state.numpy()}

    return Variant(
        label="expanded-halo fused candidate",
        prepare_trial=prepare_trial,
        run=run,
        synchronize=synchronize,
        outputs=outputs,
    )
