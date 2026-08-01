# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct the callback and native rollout-autodiff comparison."""

from collections.abc import Mapping

import numpy as np

import warp as wp
from warp.examples.optimizations.autodiff.native_autodiff_rollout import after, before
from warp.examples.optimizations.harness import (
    JSONScalar,
    OptimizationCase,
    Tolerance,
    UnsupportedWorkload,
)

_EXAMPLE_ID = "native-autodiff-rollout"


def _positive_integer(workload: Mapping[str, JSONScalar], name: str) -> int:
    value = workload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UnsupportedWorkload(f"{name} must be a positive integer")
    return value


def build_case(device: str, workload: Mapping[str, JSONScalar]) -> OptimizationCase:
    """Build variants that differ only in autodiff orchestration."""

    if set(workload) != {"size", "steps", "seed"}:
        raise UnsupportedWorkload("workload must contain exactly size, steps, and seed")
    size = _positive_integer(workload, "size")
    steps = _positive_integer(workload, "steps")
    seed = workload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise UnsupportedWorkload("seed must be a non-negative integer")

    resolved_device = wp.get_device(device)
    if not resolved_device.is_cuda:
        raise UnsupportedWorkload("native-autodiff-rollout requires a CUDA device")

    rng = np.random.default_rng(seed)
    input_values = rng.uniform(-0.75, 0.75, size=size).astype(np.float32)

    def synchronize() -> None:
        wp.synchronize_device(resolved_device)

    variant_arguments = {
        "input_values": input_values,
        "steps": steps,
        "device": str(resolved_device),
        "synchronize": synchronize,
    }
    tolerance = Tolerance(atol=2.0e-5, rtol=2.0e-4)
    return OptimizationCase(
        example_id=_EXAMPLE_ID,
        workload=dict(workload),
        baseline=before.build_variant(**variant_arguments),
        candidate=after.build_variant(**variant_arguments),
        tolerances={
            "final_state": tolerance,
            "input_gradient": tolerance,
        },
    )
