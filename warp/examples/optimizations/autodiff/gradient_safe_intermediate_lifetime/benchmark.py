# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct the gradient-safe intermediate-lifetime comparison."""

from collections.abc import Mapping

import numpy as np

import warp as wp
from warp.examples.optimizations.autodiff.gradient_safe_intermediate_lifetime import (
    after,
    before,
)
from warp.examples.optimizations.harness import (
    JSONScalar,
    OptimizationCase,
    Tolerance,
    UnsupportedWorkload,
)

_EXAMPLE_ID = "gradient-safe-intermediate-lifetime"
_SCALE = 0.99
_SHIFT = 0.0025


def _positive_integer(workload: Mapping[str, JSONScalar], name: str) -> int:
    value = workload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UnsupportedWorkload(f"{name} must be a positive integer")
    return value


def build_case(device: str, workload: Mapping[str, JSONScalar]) -> OptimizationCase:
    """Build variants for a strictly value-independent affine derivative."""

    expected_keys = {"derivative_depends_on_state", "seed", "size", "steps"}
    if set(workload) != expected_keys:
        raise UnsupportedWorkload("workload must contain exactly derivative_depends_on_state, seed, size, and steps")
    derivative_depends_on_state = workload["derivative_depends_on_state"]
    if not isinstance(derivative_depends_on_state, bool):
        raise UnsupportedWorkload("derivative_depends_on_state must be a boolean")
    if derivative_depends_on_state:
        raise UnsupportedWorkload(
            "derivative_depends_on_state=True is unsupported because overwritten primal values "
            "would be required by the manual adjoint"
        )

    size = _positive_integer(workload, "size")
    steps = _positive_integer(workload, "steps")
    seed = workload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise UnsupportedWorkload("seed must be a non-negative integer")

    resolved_device = wp.get_device(device)
    rng = np.random.default_rng(seed)
    input_values = rng.uniform(-0.75, 0.75, size=size).astype(np.float32)

    def synchronize() -> None:
        wp.synchronize_device(resolved_device)

    variant_arguments = {
        "input_values": input_values,
        "steps": steps,
        "scale": _SCALE,
        "shift": _SHIFT,
        "device": str(resolved_device),
        "synchronize": synchronize,
    }
    tolerance = Tolerance(atol=2.0e-6, rtol=2.0e-5)
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
