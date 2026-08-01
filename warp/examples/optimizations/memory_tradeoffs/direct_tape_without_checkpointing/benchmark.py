# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct the segmented-recomputation and direct-Tape comparison."""

from collections.abc import Mapping

import numpy as np

import warp as wp
from warp.examples.optimizations.harness import (
    JSONScalar,
    OptimizationCase,
    Tolerance,
    UnsupportedWorkload,
)
from warp.examples.optimizations.memory_tradeoffs.direct_tape_without_checkpointing import (
    after,
    before,
)

_EXAMPLE_ID = "direct-tape-without-checkpointing"
_BYTES_PER_VALUE = 4
_CAPACITY_DIVISOR = 4
_CORRECTNESS_SCRATCH_STATES = 20
_FIXED_HEADROOM_BYTES = 35 * 1024 * 1024


def _positive_integer(workload: Mapping[str, JSONScalar], name: str) -> int:
    value = workload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UnsupportedWorkload(f"{name} must be a positive integer")
    return value


def _direct_tape_bytes(size: int, steps: int) -> int:
    state_bytes = size * _BYTES_PER_VALUE
    return (2 * (steps + 1) + 1) * state_bytes


def _combined_peak_bytes(size: int, steps: int, segment_length: int) -> int:
    state_bytes = size * _BYTES_PER_VALUE
    segment_count = steps // segment_length
    direct_tape_states = 2 * (steps + 1) + 1
    segmented_states = 2 + (segment_count + 1) + 2 * (segment_length + 1) + 1
    host_reset_and_snapshots = 1 + 4
    scalable_states = direct_tape_states + segmented_states + host_reset_and_snapshots + _CORRECTNESS_SCRATCH_STATES
    return scalable_states * state_bytes + _FIXED_HEADROOM_BYTES


def build_case(
    device: str,
    workload: Mapping[str, JSONScalar],
) -> OptimizationCase:
    """Build variants after enforcing the bounded direct-Tape capacity rule."""

    expected_keys = {"seed", "segment_length", "size", "steps"}
    if set(workload) != expected_keys:
        raise UnsupportedWorkload("workload must contain exactly seed, segment_length, size, and steps")
    size = _positive_integer(workload, "size")
    steps = _positive_integer(workload, "steps")
    segment_length = _positive_integer(workload, "segment_length")
    if steps % segment_length != 0:
        raise UnsupportedWorkload("segment_length must divide steps exactly")
    seed = workload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise UnsupportedWorkload("seed must be a non-negative integer")

    resolved_device = wp.get_device(device)
    if not resolved_device.is_cuda:
        raise UnsupportedWorkload("direct-tape-without-checkpointing requires a CUDA device")
    free_memory_bytes = resolved_device.free_memory
    if isinstance(free_memory_bytes, bool) or not isinstance(free_memory_bytes, int) or free_memory_bytes <= 0:
        raise UnsupportedWorkload("resolved CUDA device free memory is unavailable")

    direct_tape_bytes = _direct_tape_bytes(size, steps)
    capacity_budget_bytes = free_memory_bytes // _CAPACITY_DIVISOR
    combined_peak_bytes = _combined_peak_bytes(size, steps, segment_length)
    print(
        "capacity_check "
        f"direct_tape_bytes={direct_tape_bytes} "
        f"free_memory_bytes={free_memory_bytes} "
        f"budget_bytes={capacity_budget_bytes} "
        f"combined_peak_bytes={combined_peak_bytes}"
    )
    if direct_tape_bytes > capacity_budget_bytes:
        raise UnsupportedWorkload(
            f"direct Tape estimate of {direct_tape_bytes} bytes exceeds 25% "
            f"of resolved device free memory ({capacity_budget_bytes}-byte "
            f"budget from {free_memory_bytes} bytes)"
        )

    rng = np.random.default_rng(seed)
    input_values = rng.uniform(-0.75, 0.75, size=size).astype(np.float32)

    def synchronize() -> None:
        wp.synchronize_device(resolved_device)

    shared_arguments = {
        "input_values": input_values,
        "steps": steps,
        "device": str(resolved_device),
        "synchronize": synchronize,
    }
    tolerance = Tolerance(atol=3.0e-5, rtol=3.0e-4)
    return OptimizationCase(
        example_id=_EXAMPLE_ID,
        workload=dict(workload),
        baseline=before.build_variant(
            **shared_arguments,
            segment_length=segment_length,
        ),
        candidate=after.build_variant(**shared_arguments),
        tolerances={
            "final_state": tolerance,
            "input_gradient": tolerance,
        },
    )
