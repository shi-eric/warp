# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared protocol for runtime-optimization example cards."""

from warp.examples.optimizations.harness.benchmark import PairedSamples, run_paired
from warp.examples.optimizations.harness.clean_room import Finding, scan_prohibited
from warp.examples.optimizations.harness.correctness import CorrectnessResult, OutputError, check_correctness
from warp.examples.optimizations.harness.environment import capture_environment
from warp.examples.optimizations.harness.evidence import (
    append_evidence,
    build_evidence_record,
    build_measured_contract,
    classify_summary,
    evidence_staleness_reasons,
    is_evidence_stale,
    validate_evidence_document,
    validate_evidence_record,
)
from warp.examples.optimizations.harness.manifest import load_manifest, validate_manifest
from warp.examples.optimizations.harness.model import (
    JSONScalar,
    OptimizationCase,
    Tolerance,
    UnsupportedWorkload,
    Variant,
)
from warp.examples.optimizations.harness.registry import ExampleRecord, discover_examples
from warp.examples.optimizations.harness.statistics import PairedSummary, summarize_paired

__all__ = [
    "CorrectnessResult",
    "ExampleRecord",
    "Finding",
    "JSONScalar",
    "OptimizationCase",
    "OutputError",
    "PairedSamples",
    "PairedSummary",
    "Tolerance",
    "UnsupportedWorkload",
    "Variant",
    "append_evidence",
    "build_evidence_record",
    "build_measured_contract",
    "capture_environment",
    "check_correctness",
    "classify_summary",
    "discover_examples",
    "evidence_staleness_reasons",
    "is_evidence_stale",
    "load_manifest",
    "run_paired",
    "scan_prohibited",
    "summarize_paired",
    "validate_evidence_document",
    "validate_evidence_record",
    "validate_manifest",
]
