# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical correctness checks for runtime-optimization variants."""

from dataclasses import dataclass

import numpy as np

from warp.examples.optimizations.harness.model import OptimizationCase


@dataclass(frozen=True)
class OutputError:
    """Numerical error measurements for one observable output."""

    name: str
    max_abs: float | None
    max_rel: float | None
    finite: bool
    atol: float
    rtol: float
    max_normalized: float | None
    passed: bool


@dataclass(frozen=True)
class CorrectnessResult:
    """Aggregate numerical correctness result for an optimization case."""

    passed: bool
    outputs: dict[str, OutputError]


def _dtype_family(dtype: np.dtype) -> str:
    if np.issubdtype(dtype, np.bool_):
        return "boolean"
    if np.issubdtype(dtype, np.integer):
        return "integer"
    if np.issubdtype(dtype, np.floating):
        return "floating"
    if np.issubdtype(dtype, np.complexfloating):
        return "complex"
    return "unsupported"


def _run_variant(variant) -> None:
    variant.prepare_trial()
    variant.run()
    variant.synchronize()


def _snapshot_outputs(variant) -> dict[str, np.ndarray]:
    return {name: np.asarray(value).copy() for name, value in variant.outputs().items()}


def _validate_output_contract(case: OptimizationCase, baseline_outputs, candidate_outputs) -> None:
    baseline_keys = set(baseline_outputs)
    candidate_keys = set(candidate_outputs)
    if baseline_keys != candidate_keys:
        raise ValueError("baseline and candidate output keys must match")

    tolerance_keys = set(case.tolerances)
    if tolerance_keys != baseline_keys:
        missing_keys = baseline_keys - tolerance_keys
        unexpected_keys = tolerance_keys - baseline_keys
        details = []
        if missing_keys:
            details.append(f"missing tolerances for: {', '.join(sorted(missing_keys))}")
        if unexpected_keys:
            details.append(f"unexpected tolerances for: {', '.join(sorted(unexpected_keys))}")
        raise ValueError("; ".join(details))


def check_correctness(case: OptimizationCase) -> CorrectnessResult:
    """Run both variants once and compare all declared observable outputs."""

    _run_variant(case.baseline)
    baseline_outputs = _snapshot_outputs(case.baseline)
    _run_variant(case.candidate)
    candidate_outputs = _snapshot_outputs(case.candidate)
    _validate_output_contract(case, baseline_outputs, candidate_outputs)
    output_errors = {}

    for name in baseline_outputs:
        baseline = baseline_outputs[name]
        candidate = candidate_outputs[name]
        if baseline.shape != candidate.shape:
            raise ValueError(f"output {name!r} shape mismatch: baseline {baseline.shape}, candidate {candidate.shape}")

        baseline_family = _dtype_family(baseline.dtype)
        candidate_family = _dtype_family(candidate.dtype)
        if baseline_family == "unsupported" or baseline_family != candidate_family:
            raise ValueError(
                f"output {name!r} dtype family mismatch: baseline {baseline.dtype}, candidate {candidate.dtype}"
            )

        tolerance = case.tolerances[name]
        if (
            not np.isfinite(tolerance.atol)
            or not np.isfinite(tolerance.rtol)
            or tolerance.atol < 0.0
            or tolerance.rtol < 0.0
        ):
            raise ValueError(f"output {name!r} tolerances must be finite and non-negative")
        finite = bool(np.all(np.isfinite(baseline)) and np.all(np.isfinite(candidate)))
        if finite:
            if baseline_family in {"boolean", "integer"}:
                baseline_exact = baseline.astype(object)
                candidate_exact = candidate.astype(object)
                absolute_error = np.asarray(np.abs(candidate_exact - baseline_exact), dtype=np.float64)
                baseline_values = baseline.astype(np.float64)
            else:
                comparison_dtype = np.complex128 if baseline_family == "complex" else np.float64
                baseline_values = baseline.astype(comparison_dtype, copy=False)
                candidate_values = candidate.astype(comparison_dtype, copy=False)
                absolute_error = np.abs(candidate_values - baseline_values)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                relative_denominator = np.maximum(np.abs(baseline_values), tolerance.atol)
                relative_error = np.divide(
                    absolute_error,
                    relative_denominator,
                    out=np.full(absolute_error.shape, np.nan, dtype=np.float64),
                    where=relative_denominator != 0,
                )
                relative_error[(relative_denominator == 0) & (absolute_error == 0)] = 0.0
                normalized_denominator = tolerance.atol + tolerance.rtol * np.abs(baseline_values)
                normalized_error = np.divide(
                    absolute_error,
                    normalized_denominator,
                    out=np.full(absolute_error.shape, np.nan, dtype=np.float64),
                    where=normalized_denominator != 0,
                )
                normalized_error[(normalized_denominator == 0) & (absolute_error == 0)] = 0.0

            if np.all(np.isfinite(absolute_error)):
                max_abs = float(np.max(absolute_error, initial=0.0))
            else:
                max_abs = None
            if np.all(np.isfinite(relative_error)):
                max_rel = float(np.max(relative_error, initial=0.0))
            else:
                max_rel = None
            if np.all(np.isfinite(normalized_error)):
                max_normalized = float(np.max(normalized_error, initial=0.0))
            else:
                max_normalized = None
            passed = max_normalized is not None and max_normalized <= 1.0
        else:
            max_abs = None
            max_rel = None
            max_normalized = None
            passed = False

        output_errors[name] = OutputError(
            name=name,
            max_abs=max_abs,
            max_rel=max_rel,
            finite=finite,
            atol=float(tolerance.atol),
            rtol=float(tolerance.rtol),
            max_normalized=max_normalized,
            passed=passed,
        )

    return CorrectnessResult(
        passed=all(output.passed for output in output_errors.values()),
        outputs=output_errors,
    )
