# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data model shared by runtime-optimization example cards and the harness."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

JSONScalar = str | int | float | bool | None


class UnsupportedWorkload(RuntimeError):
    """Raised when an example cannot run for a requested configuration."""


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


@dataclass
class Variant:
    label: str
    prepare_trial: Callable[[], None]
    run: Callable[[], None]
    synchronize: Callable[[], None]
    outputs: Callable[[], Mapping[str, np.ndarray]]


@dataclass
class OptimizationCase:
    example_id: str
    workload: Mapping[str, JSONScalar]
    baseline: Variant
    candidate: Variant
    tolerances: Mapping[str, Tolerance]

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ValueError("example_id must not be blank")
        if self.baseline.label == self.candidate.label:
            raise ValueError("baseline and candidate labels must be distinct")

    def validate_output_contract(self) -> None:
        baseline_keys = set(self.baseline.outputs())
        candidate_keys = set(self.candidate.outputs())
        if baseline_keys != candidate_keys:
            raise ValueError("baseline and candidate output keys must match")

        tolerance_keys = set(self.tolerances)
        if tolerance_keys != baseline_keys:
            missing_keys = baseline_keys - tolerance_keys
            unexpected_keys = tolerance_keys - baseline_keys
            details = []
            if missing_keys:
                details.append(f"missing tolerances for: {', '.join(sorted(missing_keys))}")
            if unexpected_keys:
                details.append(f"unexpected tolerances for: {', '.join(sorted(unexpected_keys))}")
            raise ValueError("; ".join(details))
