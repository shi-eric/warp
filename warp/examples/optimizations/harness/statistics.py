# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic descriptive and paired-bootstrap runtime statistics."""

from dataclasses import asdict, dataclass

import numpy as np

from warp.examples.optimizations.harness.benchmark import PairedSamples


@dataclass(frozen=True)
class PairedSummary:
    """Robust summary of paired baseline and candidate runtime samples."""

    baseline_median_ns: float
    candidate_median_ns: float
    baseline_mad_ns: float
    candidate_mad_ns: float
    median_ratio: float
    ratio_ci_low: float
    ratio_ci_high: float
    pairs: int

    def as_dict(self) -> dict[str, int | float]:
        """Return the summary using only JSON-serializable Python numbers."""

        return asdict(self)


def _validated_arrays(samples: PairedSamples) -> tuple[np.ndarray, np.ndarray]:
    pair_count = len(samples.baseline_ns)
    if pair_count < 10:
        raise ValueError("paired statistics require at least 10 pairs")
    if len(samples.candidate_ns) != pair_count or len(samples.order) != pair_count:
        raise ValueError("baseline, candidate, and order samples must have matching lengths")
    if any(order not in ("baseline-first", "candidate-first") for order in samples.order):
        raise ValueError("pair order entries must identify the first variant")

    baseline = np.asarray(samples.baseline_ns, dtype=np.float64)
    candidate = np.asarray(samples.candidate_ns, dtype=np.float64)
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(candidate)):
        raise ValueError("runtime samples must be finite")
    if np.any(baseline <= 0) or np.any(candidate <= 0):
        raise ValueError("runtime samples must be positive")
    return baseline, candidate


def summarize_paired(
    samples: PairedSamples,
    bootstrap_seed: int,
    resamples: int,
    confidence: float = 0.95,
) -> PairedSummary:
    """Summarize paired timings with a deterministic bootstrap interval."""

    baseline, candidate = _validated_arrays(samples)
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 10_000:
        raise ValueError("resamples must be an integer of at least 10000")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be finite and between 0 and 1")

    baseline_median = float(np.median(baseline))
    candidate_median = float(np.median(candidate))
    ratios = candidate / baseline
    median_ratio = float(np.median(ratios))

    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(ratios), size=(resamples, len(ratios)))
    bootstrap_medians = np.median(ratios[indices], axis=1)
    tail_percent = (1.0 - confidence) * 50.0
    ratio_ci_low, ratio_ci_high = np.percentile(
        bootstrap_medians,
        (tail_percent, 100.0 - tail_percent),
    )

    return PairedSummary(
        baseline_median_ns=baseline_median,
        candidate_median_ns=candidate_median,
        baseline_mad_ns=float(np.median(np.abs(baseline - baseline_median))),
        candidate_mad_ns=float(np.median(np.abs(candidate - candidate_median))),
        median_ratio=median_ratio,
        ratio_ci_low=float(ratio_ci_low),
        ratio_ci_high=float(ratio_ci_high),
        pairs=len(ratios),
    )
