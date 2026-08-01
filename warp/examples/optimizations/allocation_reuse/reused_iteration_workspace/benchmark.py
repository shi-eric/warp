# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct the repeated-allocation and workspace-reuse comparison."""

from collections.abc import Mapping

import numpy as np

import warp as wp
from warp.examples.optimizations.allocation_reuse.reused_iteration_workspace import (
    after,
    before,
)
from warp.examples.optimizations.harness import (
    JSONScalar,
    OptimizationCase,
    Tolerance,
    UnsupportedWorkload,
)

_EXAMPLE_ID = "reused-iteration-workspace"


def _positive_integer(workload: Mapping[str, JSONScalar], name: str) -> int:
    value = workload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UnsupportedWorkload(f"{name} must be a positive integer")
    return value


def build_case(device: str, workload: Mapping[str, JSONScalar]) -> OptimizationCase:
    """Build variants that differ only in scratch-allocation lifetime."""

    if set(workload) != {"size", "iterations", "seed"}:
        raise UnsupportedWorkload("workload must contain exactly size, iterations, and seed")
    size = _positive_integer(workload, "size")
    iterations = _positive_integer(workload, "iterations")
    seed = workload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise UnsupportedWorkload("seed must be a non-negative integer")

    resolved_device = wp.get_device(device)
    rng = np.random.default_rng(seed)
    input_values = rng.uniform(-0.75, 1.25, size=size).astype(np.float32)

    def synchronize() -> None:
        wp.synchronize_device(resolved_device)

    variant_arguments = {
        "input_values": input_values,
        "iterations": iterations,
        "device": str(resolved_device),
        "synchronize": synchronize,
    }
    return OptimizationCase(
        example_id=_EXAMPLE_ID,
        workload=dict(workload),
        baseline=before.build_variant(**variant_arguments),
        candidate=after.build_variant(**variant_arguments),
        tolerances={"energy": Tolerance(atol=1.0e-6, rtol=1.0e-6)},
    )
