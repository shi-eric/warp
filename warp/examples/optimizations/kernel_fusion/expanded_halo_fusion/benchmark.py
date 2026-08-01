# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct the separable and expanded-halo stencil comparison."""

from collections.abc import Mapping

import numpy as np

import warp as wp
from warp.examples.optimizations.harness import (
    JSONScalar,
    OptimizationCase,
    Tolerance,
    UnsupportedWorkload,
)
from warp.examples.optimizations.kernel_fusion.expanded_halo_fusion import after, before

_EXAMPLE_ID = "expanded-halo-fusion"
_SUPPORTED_RADII = frozenset((2, 4))
_DEFAULT_WORKLOAD = {
    "height": 2048,
    "iterations": 20,
    "radius": 2,
    "seed": 20260730,
    "width": 2048,
}
_FOLLOWUP_WORKLOAD = {**_DEFAULT_WORKLOAD, "radius": 4}


def _positive_integer(workload: Mapping[str, JSONScalar], name: str) -> int:
    value = workload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UnsupportedWorkload(f"{name} must be a positive integer")
    return value


def _normalized_triangular_weights(radius: int) -> np.ndarray:
    offsets = np.arange(-radius, radius + 1, dtype=np.int32)
    weights = (radius + 1 - np.abs(offsets)).astype(np.float32)
    weights /= np.sum(weights, dtype=np.float32)
    return weights


def _build_case(device: str, workload: Mapping[str, JSONScalar]) -> OptimizationCase:
    expected_keys = {"height", "iterations", "radius", "seed", "width"}
    if set(workload) != expected_keys:
        raise UnsupportedWorkload("workload must contain exactly height, iterations, radius, seed, and width")
    height = _positive_integer(workload, "height")
    width = _positive_integer(workload, "width")
    iterations = _positive_integer(workload, "iterations")
    radius = _positive_integer(workload, "radius")
    if radius not in _SUPPORTED_RADII:
        raise UnsupportedWorkload("radius must be 2 or the predeclared follow-up radius 4")
    seed = workload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise UnsupportedWorkload("seed must be a non-negative integer")

    resolved_device = wp.get_device(device)
    if not resolved_device.is_cuda:
        raise UnsupportedWorkload("expanded-halo-fusion requires a CUDA device")

    rng = np.random.default_rng(seed)
    input_values = rng.uniform(-0.75, 0.75, size=(height, width)).astype(np.float32)
    weights_values = _normalized_triangular_weights(radius)

    def synchronize() -> None:
        wp.synchronize_device(resolved_device)

    variant_arguments = {
        "input_values": input_values,
        "weights_values": weights_values,
        "iterations": iterations,
        "radius": radius,
        "device": str(resolved_device),
        "synchronize": synchronize,
    }
    return OptimizationCase(
        example_id=_EXAMPLE_ID,
        workload=dict(workload),
        baseline=before.build_variant(**variant_arguments),
        candidate=after.build_variant(**variant_arguments),
        tolerances={"result": Tolerance(atol=3.0e-5, rtol=3.0e-4)},
    )


def build_correctness_case(device: str, workload: Mapping[str, JSONScalar]) -> OptimizationCase:
    """Build a reduced case used only by the correctness entry point."""

    return _build_case(device, workload)


def build_case(device: str, workload: Mapping[str, JSONScalar]) -> OptimizationCase:
    """Build only the default or exact predeclared evidence workload."""

    if workload != _DEFAULT_WORKLOAD and workload != _FOLLOWUP_WORKLOAD:
        raise UnsupportedWorkload("evidence workload must match the default or exact predeclared radius-4 follow-up")
    return _build_case(device, workload)
