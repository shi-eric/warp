# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alternating paired runtime measurement for optimization variants."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

from warp.examples.optimizations.harness.model import OptimizationCase, Variant


@dataclass(frozen=True)
class PairedSamples:
    """Steady-state timings collected from paired baseline/candidate trials."""

    baseline_ns: tuple[int, ...]
    candidate_ns: tuple[int, ...]
    order: tuple[str, ...]


def _warm_up(variant: Variant, warmups: int) -> None:
    for _ in range(warmups):
        variant.prepare_trial()
        variant.run()
        variant.synchronize()


def _measure(variant: Variant, timer_ns: Callable[[], int]) -> int:
    variant.prepare_trial()
    variant.synchronize()
    start_ns = timer_ns()
    variant.run()
    variant.synchronize()
    end_ns = timer_ns()
    elapsed_ns = end_ns - start_ns
    if isinstance(elapsed_ns, bool) or not isinstance(elapsed_ns, Integral) or elapsed_ns <= 0:
        raise ValueError("measured runtime must be a positive integer number of nanoseconds")
    return int(elapsed_ns)


def run_paired(
    case: OptimizationCase,
    warmups: int,
    pairs: int,
    timer_ns: Callable[[], int] = time.perf_counter_ns,
) -> PairedSamples:
    """Measure alternating baseline/candidate trials after untimed warm-ups."""

    if warmups < 3:
        raise ValueError("warmups must be at least 3")
    if pairs < 10:
        raise ValueError("pairs must be at least 10")

    _warm_up(case.baseline, warmups)
    _warm_up(case.candidate, warmups)

    baseline_ns = []
    candidate_ns = []
    order = []
    for pair_index in range(pairs):
        if pair_index % 2 == 0:
            baseline_ns.append(_measure(case.baseline, timer_ns))
            candidate_ns.append(_measure(case.candidate, timer_ns))
            order.append("baseline-first")
        else:
            candidate_ns.append(_measure(case.candidate, timer_ns))
            baseline_ns.append(_measure(case.baseline, timer_ns))
            order.append("candidate-first")

    return PairedSamples(
        baseline_ns=tuple(baseline_ns),
        candidate_ns=tuple(candidate_ns),
        order=tuple(order),
    )
