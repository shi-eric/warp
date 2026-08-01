# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseline that recomputes saved segments under fresh Warp Tapes."""

from collections.abc import Callable

import numpy as np

import warp as wp
from warp.examples.optimizations.harness import Variant


@wp.kernel
def smooth_step(x: wp.array[float], y: wp.array[float]):
    i = wp.tid()
    y[i] = 0.8 * wp.tanh(x[i]) + 0.02


def build_variant(
    *,
    input_values: np.ndarray,
    steps: int,
    segment_length: int,
    device: str,
    synchronize: Callable[[], None],
) -> Variant:
    """Build a segmented forward-recomputation and backward variant."""

    size = len(input_values)
    segment_count = steps // segment_length
    forward = [
        wp.array(input_values, dtype=wp.float32, device=device),
        wp.empty(size, dtype=wp.float32, device=device),
    ]
    boundaries = [wp.empty(size, dtype=wp.float32, device=device) for _ in range(segment_count + 1)]
    segment_states = [
        wp.empty(size, dtype=wp.float32, device=device, requires_grad=True) for _ in range(segment_length + 1)
    ]
    boundary_adjoint = wp.empty(size, dtype=wp.float32, device=device)

    def prepare_trial() -> None:
        forward[0].assign(input_values)

    def run() -> None:
        boundary_adjoint.fill_(1.0)

        wp.copy(boundaries[0], forward[0])
        current = 0
        for segment_index in range(segment_count):
            for _ in range(segment_length):
                following = 1 - current
                wp.launch(
                    smooth_step,
                    dim=size,
                    inputs=[forward[current]],
                    outputs=[forward[following]],
                    device=device,
                )
                current = following
            wp.copy(boundaries[segment_index + 1], forward[current])

        for segment_index in range(segment_count - 1, -1, -1):
            segment_states[0].assign(boundaries[segment_index])
            for state in segment_states:
                state.grad.zero_()

            tape = wp.Tape()
            with tape:
                for step in range(segment_length):
                    wp.launch(
                        smooth_step,
                        dim=size,
                        inputs=[segment_states[step]],
                        outputs=[segment_states[step + 1]],
                        device=device,
                    )
            tape.backward(grads={segment_states[-1]: boundary_adjoint})
            boundary_adjoint.assign(segment_states[0].grad)

    def outputs() -> dict[str, np.ndarray]:
        return {
            "final_state": boundaries[-1].numpy(),
            "input_gradient": boundary_adjoint.numpy(),
        }

    return Variant(
        label="segmented recomputation baseline",
        prepare_trial=prepare_trial,
        run=run,
        synchronize=synchronize,
        outputs=outputs,
    )
