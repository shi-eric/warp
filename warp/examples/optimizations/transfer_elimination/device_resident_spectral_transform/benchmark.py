# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct the device-resident spectral transform comparison."""

from collections.abc import Mapping

import numpy as np

import warp as wp
from warp.examples.optimizations.harness import (
    JSONScalar,
    OptimizationCase,
    Tolerance,
    UnsupportedWorkload,
)
from warp.examples.optimizations.transfer_elimination.device_resident_spectral_transform import (
    after,
    before,
)

_EXAMPLE_ID = "device-resident-spectral-transform"


def _positive_integer(workload: Mapping[str, JSONScalar], name: str) -> int:
    value = workload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UnsupportedWorkload(f"{name} must be a positive integer")
    return value


def _build_inputs(size: int, batch: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    sequences = np.random.SeedSequence(seed).spawn(batch + 1)
    initial_pairs = np.empty((batch, size, 2), dtype=np.float32)
    for row, sequence in enumerate(sequences[:-1]):
        rng = np.random.default_rng(sequence)
        initial_pairs[row] = rng.standard_normal((size, 2), dtype=np.float32) * 0.25

    gain_rng = np.random.default_rng(sequences[-1])
    gain_pairs = np.empty((1, size, 2), dtype=np.float32)
    gain_pairs[0, :, 0] = gain_rng.uniform(0.8, 0.95, size=size)
    gain_pairs[0, :, 1] = gain_rng.uniform(-0.025, 0.025, size=size)
    return initial_pairs, gain_pairs


def build_case(device: str, workload: Mapping[str, JSONScalar]) -> OptimizationCase:
    """Build matching host-staged and device-resident transform variants."""

    if set(workload) != {"size", "batch", "iterations", "seed"}:
        raise UnsupportedWorkload("workload must contain exactly size, batch, iterations, and seed")
    size = _positive_integer(workload, "size")
    batch = _positive_integer(workload, "batch")
    iterations = _positive_integer(workload, "iterations")
    seed = workload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise UnsupportedWorkload("seed must be a non-negative integer")
    if size != after.TRANSFORM_SIZE:
        raise UnsupportedWorkload(f"size must be {after.TRANSFORM_SIZE} for this compiled tile")

    resolved_device = wp.get_device(device)
    if not resolved_device.is_cuda:
        raise UnsupportedWorkload("the device-residency comparison requires a CUDA device")

    initial_pairs, gain_pairs = _build_inputs(size, batch, seed)

    def synchronize() -> None:
        wp.synchronize_device(resolved_device)

    variant_arguments = {
        "initial_pairs": initial_pairs,
        "gain_pairs": gain_pairs,
        "iterations": iterations,
        "device": str(resolved_device),
        "synchronize": synchronize,
    }
    tolerances = {
        "imaginary": Tolerance(atol=3.0e-4, rtol=3.0e-4),
        "real": Tolerance(atol=3.0e-4, rtol=3.0e-4),
    }
    return OptimizationCase(
        example_id=_EXAMPLE_ID,
        workload=dict(workload),
        baseline=before.build_variant(**variant_arguments),
        candidate=after.build_variant(**variant_arguments),
        tolerances=tolerances,
    )
