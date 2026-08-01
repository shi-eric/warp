# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate that differentiates a complete rollout with one Warp Tape."""

from collections.abc import Callable

import numpy as np
import torch

import warp as wp
from warp.examples.optimizations.autodiff.native_autodiff_rollout.before import rollout_step
from warp.examples.optimizations.harness import Variant


def build_variant(
    *,
    input_values: np.ndarray,
    steps: int,
    device: str,
    synchronize: Callable[[], None],
) -> Variant:
    """Build a rollout recorded and differentiated by one Warp Tape."""

    size = len(input_values)
    torch_device = wp.device_to_torch(wp.get_device(device))
    input_reset = torch.as_tensor(input_values, device=torch_device)
    input_tensor = input_reset.clone().requires_grad_(True)
    input_tensor.grad = torch.zeros_like(input_tensor)
    output_arrays = [wp.empty(size, dtype=wp.float32, device=device, requires_grad=True) for _ in range(steps)]
    final_gradient = wp.ones(size, dtype=wp.float32, device=device)
    tape = None
    final_state = None
    input_gradient = None

    def prepare_trial() -> None:
        nonlocal final_state, input_gradient
        if tape is not None:
            warp_stream = wp.stream_from_torch(input_tensor.device)
            with wp.ScopedStream(warp_stream):
                tape.zero()
        with torch.no_grad():
            input_tensor.copy_(input_reset)
            input_tensor.grad.zero_()
        final_state = None
        input_gradient = None

    def run() -> None:
        nonlocal final_state, input_gradient, tape
        input_array = wp.from_torch(input_tensor, requires_grad=True)
        states = [input_array, *output_arrays]
        warp_stream = wp.stream_from_torch(input_tensor.device)
        tape = wp.Tape()
        with wp.ScopedStream(warp_stream):
            with tape:
                for step in range(steps):
                    wp.launch(
                        rollout_step,
                        dim=size,
                        inputs=[states[step]],
                        outputs=[states[step + 1]],
                        device=device,
                    )
            tape.backward(grads={states[-1]: final_gradient})
        final_state = wp.to_torch(states[-1], requires_grad=False)
        input_gradient = wp.to_torch(input_array.grad, requires_grad=False)

    def outputs() -> dict[str, np.ndarray]:
        return {
            "final_state": final_state.detach().cpu().numpy(),
            "input_gradient": input_gradient.detach().cpu().numpy(),
        }

    return Variant(
        label="single Warp Tape candidate",
        prepare_trial=prepare_trial,
        run=run,
        synchronize=synchronize,
        outputs=outputs,
    )
