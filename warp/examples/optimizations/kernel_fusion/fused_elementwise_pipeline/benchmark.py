# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct the fused elementwise pipeline comparison."""

from collections.abc import Mapping

import numpy as np

import warp as wp
from warp.examples.optimizations.harness import (
    JSONScalar,
    OptimizationCase,
    Tolerance,
    UnsupportedWorkload,
)
from warp.examples.optimizations.kernel_fusion.fused_elementwise_pipeline import after, before

_EXAMPLE_ID = "fused-elementwise-pipeline"


def _positive_integer(workload: Mapping[str, JSONScalar], name: str) -> int:
    value = workload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UnsupportedWorkload(f"{name} must be a positive integer")
    return value


def build_case(device: str, workload: Mapping[str, JSONScalar]) -> OptimizationCase:
    """Build matching separate and fused variants for one requested workload."""

    if set(workload) != {"size", "iterations", "seed"}:
        raise UnsupportedWorkload("workload must contain exactly size, iterations, and seed")
    size = _positive_integer(workload, "size")
    iterations = _positive_integer(workload, "iterations")
    seed = workload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise UnsupportedWorkload("seed must be a non-negative integer")

    resolved_device = wp.get_device(device)
    rng = np.random.default_rng(seed)
    x_values = rng.uniform(-1.0, 1.0, size=size).astype(np.float32)
    bias_values = rng.uniform(-0.5, 0.5, size=size).astype(np.float32)

    def synchronize() -> None:
        wp.synchronize_device(resolved_device)

    variant_arguments = {
        "x_values": x_values,
        "bias_values": bias_values,
        "iterations": iterations,
        "device": str(resolved_device),
        "synchronize": synchronize,
    }
    return OptimizationCase(
        example_id=_EXAMPLE_ID,
        workload=dict(workload),
        baseline=before.build_variant(**variant_arguments),
        candidate=after.build_variant(**variant_arguments),
        tolerances={"result": Tolerance(atol=2.0e-6, rtol=2.0e-5)},
    )
