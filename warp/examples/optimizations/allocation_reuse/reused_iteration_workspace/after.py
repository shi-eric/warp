# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate that reuses fixed temporary storage across iterations."""

from collections.abc import Callable

import numpy as np

import warp as wp
from warp.examples.optimizations.allocation_reuse.reused_iteration_workspace.before import (
    accumulate_energy,
    center_and_scale,
)
from warp.examples.optimizations.harness import Variant


def build_variant(
    *,
    input_values: np.ndarray,
    iterations: int,
    device: str,
    synchronize: Callable[[], None],
) -> Variant:
    """Build the candidate with one fixed scratch allocation."""

    size = len(input_values)
    values = wp.array(input_values, dtype=wp.float32, device=device)
    output = wp.empty_like(values)
    reset = wp.zeros_like(values)
    scratch = wp.empty_like(values)

    def prepare_trial() -> None:
        wp.copy(output, reset)

    def run() -> None:
        for _ in range(iterations):
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
        label="reused workspace candidate",
        prepare_trial=prepare_trial,
        run=run,
        synchronize=synchronize,
        outputs=outputs,
    )
