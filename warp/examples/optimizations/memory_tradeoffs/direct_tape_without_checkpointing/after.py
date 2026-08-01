# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate that records the bounded rollout on one Warp Tape."""

from collections.abc import Callable

import numpy as np

import warp as wp
from warp.examples.optimizations.harness import Variant
from warp.examples.optimizations.memory_tradeoffs.direct_tape_without_checkpointing.before import (
    smooth_step,
)


def build_variant(
    *,
    input_values: np.ndarray,
    steps: int,
    device: str,
    synchronize: Callable[[], None],
) -> Variant:
    """Build one direct Warp Tape over the complete rollout."""

    size = len(input_values)
    states = [
        wp.array(
            input_values,
            dtype=wp.float32,
            device=device,
            requires_grad=True,
        )
    ]
    states.extend(
        wp.empty(
            size,
            dtype=wp.float32,
            device=device,
            requires_grad=True,
        )
        for _ in range(steps)
    )
    final_gradient = wp.ones(size, dtype=wp.float32, device=device)

    def prepare_trial() -> None:
        states[0].assign(input_values)

    def run() -> None:
        for state in states:
            state.grad.zero_()

        tape = wp.Tape()
        with tape:
            for step in range(steps):
                wp.launch(
                    smooth_step,
                    dim=size,
                    inputs=[states[step]],
                    outputs=[states[step + 1]],
                    device=device,
                )
        tape.backward(grads={states[-1]: final_gradient})

    def outputs() -> dict[str, np.ndarray]:
        return {
            "final_state": states[-1].numpy(),
            "input_gradient": states[0].grad.numpy(),
        }

    return Variant(
        label="single direct Warp Tape candidate",
        prepare_trial=prepare_trial,
        run=run,
        synchronize=synchronize,
        outputs=outputs,
    )
