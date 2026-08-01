# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseline that stages every PyTorch and Warp exchange through the host."""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

import warp as wp
from warp.examples.optimizations.harness import Variant


@wp.kernel
def transform_values(input_values: wp.array[float], output_values: wp.array[float]):
    i = wp.tid()
    value = input_values[i]
    output_values[i] = 0.5 * value + wp.sin(value)


class TorchWarpStreamState:
    """Own one non-default PyTorch stream and its Warp wrapper."""

    def __init__(self, *, input_values: np.ndarray, device: str):
        self.device = wp.get_device(device)
        self.torch_device = wp.device_to_torch(self.device)
        self.torch_stream = torch.cuda.Stream(device=self.torch_device)
        self.warp_stream = wp.stream_from_torch(self.torch_stream)

        with self.active_stream():
            self.reset_values = torch.as_tensor(
                input_values,
                dtype=torch.float32,
                device=self.torch_device,
            )
            self.torch_values = self.reset_values.clone()
        self.synchronize()
        self.assert_stream_contract()

    @contextmanager
    def active_stream(self):
        """Make both frameworks use the owned CUDA stream."""

        with torch.cuda.stream(self.torch_stream), wp.ScopedStream(self.warp_stream):
            yield

    def assert_stream_contract(self) -> None:
        default_stream = torch.cuda.default_stream(self.torch_device)
        if self.torch_stream.cuda_stream == default_stream.cuda_stream:
            raise AssertionError("trial stream must be a non-default PyTorch stream")
        if self.warp_stream.cuda_stream != self.torch_stream.cuda_stream:
            raise AssertionError("PyTorch and Warp must use the same CUDA stream")
        if self.warp_stream.device != self.device:
            raise AssertionError("converted Warp stream must target the trial CUDA device")
        if self.warp_stream.is_blocking:
            raise AssertionError("a converted non-default PyTorch stream must remain non-blocking")

    def assert_active_stream(self) -> None:
        torch_stream = torch.cuda.current_stream(self.torch_device)
        if torch_stream.cuda_stream != self.torch_stream.cuda_stream:
            raise AssertionError("PyTorch producer or consumer is not on the trial stream")
        if wp.get_stream(self.device).cuda_stream != self.warp_stream.cuda_stream:
            raise AssertionError("Warp operation is not on the trial stream")

    def synchronize(self) -> None:
        self.torch_stream.synchronize()


class HostStagedTrial(TorchWarpStreamState):
    """Mutable state retained for one host-staged trial."""

    def __init__(self, *, input_values: np.ndarray, iterations: int, device: str):
        super().__init__(input_values=input_values, device=device)
        self.iterations = iterations
        self.host_input: np.ndarray | None = None
        self.warp_values: wp.array | None = None
        self.host_result: np.ndarray | None = None

    def prepare_trial(self) -> None:
        with self.active_stream():
            self.assert_active_stream()
            self.torch_values.copy_(self.reset_values)
        self.host_input = None
        self.warp_values = None
        self.host_result = None

    def run(self) -> None:
        with self.active_stream():
            self.assert_active_stream()
            for _ in range(self.iterations):
                self.host_input = self.torch_values.detach().cpu().numpy()
                self.warp_values = wp.array(
                    self.host_input,
                    dtype=float,
                    device=self.device,
                )
                wp.launch(
                    transform_values,
                    dim=self.warp_values.size,
                    inputs=[self.warp_values],
                    outputs=[self.warp_values],
                )
                self.host_result = self.warp_values.numpy()
                self.torch_values = torch.from_numpy(self.host_result).to(self.torch_device)

    def outputs(self) -> dict[str, np.ndarray]:
        with torch.cuda.stream(self.torch_stream):
            values = self.torch_values.detach().cpu().numpy()
        return {"values": values}


@dataclass
class HostStagedVariant(Variant):
    """Harness variant that exposes its retained trial state to correctness checks."""

    trial_state: HostStagedTrial


def build_variant(
    *,
    input_values: np.ndarray,
    iterations: int,
    device: str,
) -> HostStagedVariant:
    """Build a variant that stages through fresh host and device storage."""

    state = HostStagedTrial(
        input_values=input_values,
        iterations=iterations,
        device=device,
    )
    return HostStagedVariant(
        label="host-staged PyTorch and Warp baseline",
        prepare_trial=state.prepare_trial,
        run=state.run,
        synchronize=state.synchronize,
        outputs=state.outputs,
        trial_state=state,
    )
