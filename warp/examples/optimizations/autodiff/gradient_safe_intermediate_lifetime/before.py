# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseline that retains one differentiable state for every affine step."""

from collections.abc import Callable

import numpy as np

import warp as wp
from warp.examples.optimizations.harness import Variant


@wp.kernel
def affine_step(
    x: wp.array[float],
    y: wp.array[float],
    scale: float,
    shift: float,
):
    i = wp.tid()
    y[i] = scale * x[i] + shift


def build_variant(
    *,
    input_values: np.ndarray,
    steps: int,
    scale: float,
    shift: float,
    device: str,
    synchronize: Callable[[], None],
) -> Variant:
    """Build the unique-state Warp Tape baseline."""

    size = len(input_values)
    states = [wp.array(input_values, dtype=wp.float32, device=device, requires_grad=True)]
    states.extend(wp.empty(size, dtype=wp.float32, device=device, requires_grad=True) for _ in range(steps))
    final_state = states[-1]
    input_gradient = states[0].grad

    def prepare_trial() -> None:
        states[0].assign(input_values)
        for state in states:
            state.grad.zero_()
        states[-1].grad.fill_(1.0)

    def run() -> None:
        tape = wp.Tape()
        with tape:
            for step in range(steps):
                wp.launch(
                    affine_step,
                    dim=size,
                    inputs=[states[step], states[step + 1], scale, shift],
                    device=device,
                )
        tape.backward()

    def outputs() -> dict[str, np.ndarray]:
        return {
            "final_state": final_state.numpy(),
            "input_gradient": input_gradient.numpy(),
        }

    return Variant(
        label="unique-intermediate Warp Tape baseline",
        prepare_trial=prepare_trial,
        run=run,
        synchronize=synchronize,
        outputs=outputs,
    )
