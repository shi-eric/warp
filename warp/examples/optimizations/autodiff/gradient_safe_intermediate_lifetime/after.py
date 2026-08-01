# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate with two forward buffers and two manual-adjoint buffers."""

from collections.abc import Callable

import numpy as np

import warp as wp
from warp.examples.optimizations.autodiff.gradient_safe_intermediate_lifetime.before import (
    affine_step,
)
from warp.examples.optimizations.harness import Variant


@wp.kernel
def affine_adjoint(
    adjacent: wp.array[float],
    previous: wp.array[float],
    scale: float,
):
    i = wp.tid()
    previous[i] = scale * adjacent[i]


def build_variant(
    *,
    input_values: np.ndarray,
    steps: int,
    scale: float,
    shift: float,
    device: str,
    synchronize: Callable[[], None],
) -> Variant:
    """Build the ping-pong affine rollout and constant manual adjoint."""

    size = len(input_values)
    forward = [
        wp.array(input_values, dtype=wp.float32, device=device),
        wp.empty(size, dtype=wp.float32, device=device),
    ]
    adjoint = [
        wp.zeros(size, dtype=wp.float32, device=device),
        wp.zeros(size, dtype=wp.float32, device=device),
    ]
    final_index = steps % 2

    def prepare_trial() -> None:
        forward[0].assign(input_values)
        adjoint[0].zero_()
        adjoint[1].zero_()
        adjoint[final_index].fill_(1.0)

    def run() -> None:
        current = 0
        for _ in range(steps):
            following = 1 - current
            wp.launch(
                affine_step,
                dim=size,
                inputs=[forward[current], forward[following], scale, shift],
                device=device,
            )
            current = following

        current = final_index
        for _ in range(steps):
            previous = 1 - current
            wp.launch(
                affine_adjoint,
                dim=size,
                inputs=[adjoint[current], adjoint[previous], scale],
                device=device,
            )
            current = previous

    def outputs() -> dict[str, np.ndarray]:
        return {
            "final_state": forward[final_index].numpy(),
            "input_gradient": adjoint[0].numpy(),
        }

    return Variant(
        label="two-buffer constant-adjoint candidate",
        prepare_trial=prepare_trial,
        run=run,
        synchronize=synchronize,
        outputs=outputs,
    )
