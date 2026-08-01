# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate that retains one CUDA buffer across PyTorch and Warp."""

from dataclasses import dataclass

import numpy as np
import torch

import warp as wp
from warp.examples.optimizations.harness import Variant
from warp.examples.optimizations.interoperability.device_resident_torch_exchange.before import (
    TorchWarpStreamState,
    transform_values,
)


class DeviceResidentTrial(TorchWarpStreamState):
    """Owners and views retained for one device-resident trial."""

    def __init__(self, *, input_values: np.ndarray, iterations: int, device: str):
        super().__init__(input_values=input_values, device=device)
        self.iterations = iterations
        with self.active_stream():
            self.warp_values = wp.from_torch(
                self.torch_values,
                dtype=wp.float32,
                requires_grad=False,
            )
        self.endpoint_torch_view: torch.Tensor | None = None
        self.ordering_consumer: torch.Tensor | None = None
        self.assert_storage_alias(endpoint_required=False)

    def prepare_trial(self) -> None:
        self.endpoint_torch_view = None
        self.ordering_consumer = None
        with self.active_stream():
            self.assert_active_stream()
            self.torch_values.copy_(self.reset_values)

    def run(self) -> None:
        with self.active_stream():
            self.assert_active_stream()
            for _ in range(self.iterations):
                wp.launch(
                    transform_values,
                    dim=self.warp_values.size,
                    inputs=[self.warp_values],
                    outputs=[self.warp_values],
                )
            self.endpoint_torch_view = wp.to_torch(
                self.warp_values,
                requires_grad=False,
            )

    def assert_storage_alias(self, *, endpoint_required: bool = True) -> None:
        if not self.torch_values.is_cuda:
            raise AssertionError("PyTorch owner must reside on CUDA")
        if self.torch_values.dtype != torch.float32:
            raise AssertionError("PyTorch owner must have float32 dtype")
        if not self.torch_values.is_contiguous():
            raise AssertionError("PyTorch owner must be contiguous")
        if self.warp_values.dtype != wp.float32 or not self.warp_values.is_contiguous:
            raise AssertionError("Warp view must be a contiguous float32 array")
        if self.warp_values.device != wp.device_from_torch(self.torch_values.device):
            raise AssertionError("PyTorch owner and Warp view must use the same CUDA device")
        if self.warp_values.ptr != self.torch_values.data_ptr():
            raise AssertionError("PyTorch owner and Warp view must share one data pointer")

        if self.endpoint_torch_view is None:
            if endpoint_required:
                raise AssertionError("endpoint PyTorch view has not been created")
            return
        if self.endpoint_torch_view.device != self.torch_values.device:
            raise AssertionError("endpoint PyTorch view must use the owner's CUDA device")
        if self.endpoint_torch_view.data_ptr() != self.torch_values.data_ptr():
            raise AssertionError("endpoint PyTorch view must share the owner's data pointer")

    def run_ordering_probe(self, value: float) -> None:
        """Schedule a real PyTorch producer, Warp transform, and PyTorch consumer."""

        self.endpoint_torch_view = None
        self.ordering_consumer = None
        with self.active_stream():
            self.assert_active_stream()
            self.torch_values.fill_(value)
            wp.launch(
                transform_values,
                dim=self.warp_values.size,
                inputs=[self.warp_values],
                outputs=[self.warp_values],
            )
            self.endpoint_torch_view = wp.to_torch(
                self.warp_values,
                requires_grad=False,
            )
            self.ordering_consumer = self.endpoint_torch_view.clone()

    def ordering_output(self) -> np.ndarray:
        if self.ordering_consumer is None:
            raise AssertionError("ordering probe has not run")
        with torch.cuda.stream(self.torch_stream):
            output = self.ordering_consumer.detach().cpu().numpy()
        return output

    def outputs(self) -> dict[str, np.ndarray]:
        self.assert_storage_alias()
        with torch.cuda.stream(self.torch_stream):
            values = self.endpoint_torch_view.detach().cpu().numpy()
        return {"values": values}


@dataclass
class DeviceResidentVariant(Variant):
    """Harness variant that exposes its retained owner and view state."""

    trial_state: DeviceResidentTrial


def build_variant(
    *,
    input_values: np.ndarray,
    iterations: int,
    device: str,
) -> DeviceResidentVariant:
    """Build a variant with one persistent zero-copy Warp view."""

    state = DeviceResidentTrial(
        input_values=input_values,
        iterations=iterations,
        device=device,
    )
    return DeviceResidentVariant(
        label="device-resident PyTorch and Warp candidate",
        prepare_trial=state.prepare_trial,
        run=state.run,
        synchronize=state.synchronize,
        outputs=state.outputs,
        trial_state=state,
    )
