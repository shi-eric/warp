# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseline that allocates temporary storage inside every iteration."""

from collections.abc import Callable

import numpy as np

import warp as wp
from warp.examples.optimizations.harness import Variant


@wp.kernel
def center_and_scale(
    values: wp.array[float],
    scratch: wp.array[float],
):
    index = wp.tid()
    scratch[index] = (values[index] - 0.125) * 0.75


@wp.kernel
def accumulate_energy(
    scratch: wp.array[float],
    output: wp.array[float],
):
    index = wp.tid()
    value = scratch[index]
    output[index] = output[index] + value * value


def build_variant(
    *,
    input_values: np.ndarray,
    iterations: int,
    device: str,
    synchronize: Callable[[], None],
) -> Variant:
    """Build the baseline with per-iteration scratch allocation."""

    size = len(input_values)
    values = wp.array(input_values, dtype=wp.float32, device=device)
    output = wp.empty_like(values)
    reset = wp.zeros_like(values)

    def prepare_trial() -> None:
        wp.copy(output, reset)

    def run() -> None:
        for _ in range(iterations):
            scratch = wp.empty_like(values)
            wp.launch(
                center_and_scale,
                dim=size,
                inputs=[values],
                outputs=[scratch],
                device=device,
            )
            wp.launch(
                accumulate_energy,
                dim=size,
                inputs=[scratch],
                outputs=[output],
                device=device,
            )

    def outputs() -> dict[str, np.ndarray]:
        return {"energy": output.numpy()}

    return Variant(
        label="per-iteration allocation baseline",
        prepare_trial=prepare_trial,
        run=run,
        synchronize=synchronize,
        outputs=outputs,
    )
