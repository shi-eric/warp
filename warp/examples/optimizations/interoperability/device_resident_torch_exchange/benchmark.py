# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct the host-staged and device-resident exchange comparison."""

from collections.abc import Mapping

import numpy as np

import warp as wp
from warp.examples.optimizations.harness import (
    JSONScalar,
    OptimizationCase,
    Tolerance,
    UnsupportedWorkload,
)

_EXAMPLE_ID = "device-resident-torch-exchange"


def _positive_integer(workload: Mapping[str, JSONScalar], name: str) -> int:
    value = workload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UnsupportedWorkload(f"{name} must be a positive integer")
    return value


def build_case(
    device: str,
    workload: Mapping[str, JSONScalar],
) -> OptimizationCase:
    """Build variants that differ only in framework exchange strategy."""

    if set(workload) != {"iterations", "seed", "size"}:
        raise UnsupportedWorkload("workload must contain exactly iterations, seed, and size")
    iterations = _positive_integer(workload, "iterations")
    size = _positive_integer(workload, "size")
    seed = workload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise UnsupportedWorkload("seed must be a non-negative integer")

    resolved_device = wp.get_device(device)
    if not resolved_device.is_cuda:
        raise UnsupportedWorkload("device-resident-torch-exchange requires a CUDA device")

    from warp.examples.optimizations.interoperability.device_resident_torch_exchange import (  # noqa: PLC0415
        after,
        before,
    )

    rng = np.random.default_rng(seed)
    input_values = rng.uniform(-0.75, 0.75, size=size).astype(np.float32)
    variant_arguments = {
        "input_values": input_values,
        "iterations": iterations,
        "device": str(resolved_device),
    }
    tolerance = Tolerance(atol=2.0e-6, rtol=2.0e-5)
    return OptimizationCase(
        example_id=_EXAMPLE_ID,
        workload=dict(workload),
        baseline=before.build_variant(**variant_arguments),
        candidate=after.build_variant(**variant_arguments),
        tolerances={"values": tolerance},
    )
