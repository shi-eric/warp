# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseline that crosses PyTorch autograd once per rollout step."""

from collections.abc import Callable

import numpy as np
import torch

import warp as wp
from warp.examples.optimizations.harness import Variant


@wp.kernel
def rollout_step(x: wp.array[float], y: wp.array[float]):
    i = wp.tid()
    y[i] = wp.sin(0.7 * x[i]) + 0.05 * x[i]


class _RolloutStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, device):
        input_array = wp.from_torch(input_tensor, requires_grad=False)
        warp_stream = wp.stream_from_torch(input_tensor.device)
        with wp.ScopedStream(warp_stream):
            output_array = wp.empty_like(input_array)
            wp.launch(
                rollout_step,
                dim=input_array.size,
                inputs=[input_array],
                outputs=[output_array],
                device=device,
            )
        ctx.device = device
        ctx.input_array = input_array
        ctx.output_array = output_array
        return wp.to_torch(output_array, requires_grad=False)

    @staticmethod
    def backward(ctx, output_gradient):
        output_gradient = output_gradient.contiguous()
        output_gradient_array = wp.from_torch(output_gradient, requires_grad=False)
        warp_stream = wp.stream_from_torch(output_gradient.device)
        with wp.ScopedStream(warp_stream):
            input_gradient_array = wp.zeros_like(ctx.input_array)
            wp.launch(
                rollout_step,
                dim=ctx.input_array.size,
                inputs=[ctx.input_array],
                outputs=[ctx.output_array],
                adj_inputs=[input_gradient_array],
                adj_outputs=[output_gradient_array],
                adjoint=True,
                device=ctx.device,
            )
        return wp.to_torch(input_gradient_array, requires_grad=False), None


def build_variant(
    *,
    input_values: np.ndarray,
    steps: int,
    device: str,
    synchronize: Callable[[], None],
) -> Variant:
    """Build a rollout with one PyTorch autograd callback per step."""

    torch_device = wp.device_to_torch(wp.get_device(device))
    input_reset = torch.as_tensor(input_values, device=torch_device)
    input_tensor = input_reset.clone().requires_grad_(True)
    input_tensor.grad = torch.zeros_like(input_tensor)
    final_state = None

    def prepare_trial() -> None:
        nonlocal final_state
        with torch.no_grad():
            input_tensor.copy_(input_reset)
            input_tensor.grad.zero_()
        final_state = None

    def run() -> None:
        nonlocal final_state
        state = input_tensor
        for _ in range(steps):
            state = _RolloutStep.apply(state, device)
        state.sum().backward()
        final_state = state

    def outputs() -> dict[str, np.ndarray]:
        return {
            "final_state": final_state.detach().cpu().numpy(),
            "input_gradient": input_tensor.grad.detach().cpu().numpy(),
        }

    return Variant(
        label="per-step PyTorch callback baseline",
        prepare_trial=prepare_trial,
        run=run,
        synchronize=synchronize,
        outputs=outputs,
    )
