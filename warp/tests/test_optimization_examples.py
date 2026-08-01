# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the shared protocol used by runtime-optimization example cards."""

import io
import json
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import warp.examples.optimizations.harness.environment as environment_module
import warp.examples.optimizations.run as runner_module
from warp.examples.optimizations.harness.benchmark import PairedSamples, run_paired
from warp.examples.optimizations.harness.clean_room import scan_prohibited
from warp.examples.optimizations.harness.correctness import CorrectnessResult, OutputError, check_correctness
from warp.examples.optimizations.harness.environment import capture_environment
from warp.examples.optimizations.harness.evidence import (
    append_evidence,
    build_evidence_record,
    classify_summary,
    is_evidence_stale,
    validate_evidence_document,
    validate_evidence_record,
)
from warp.examples.optimizations.harness.manifest import load_manifest, validate_manifest
from warp.examples.optimizations.harness.model import OptimizationCase, Tolerance, Variant
from warp.examples.optimizations.harness.registry import discover_examples
from warp.examples.optimizations.harness.statistics import PairedSummary, summarize_paired


def make_valid_manifest():
    return {
        "schema_version": 1,
        "id": "synthetic-card",
        "title": "Synthetic card",
        "category": "kernel-fusion",
        "status": "unverified",
        "summary": "Exercise manifest validation.",
        "recognition": {"signals": ["multiple_pointwise_launches"]},
        "applicability": {
            "preconditions": ["matching_iteration_domain"],
            "contraindications": ["required_intermediate_observation"],
        },
        "semantics": {
            "observable_outputs": ["result"],
            "tolerance": {"relative": 1.0e-5, "absolute": 1.0e-6},
        },
        "impact": {
            "cuda": "unverified",
            "cpu": "unverified",
            "mechanism": ["reduces_global_memory_traffic"],
        },
        "compatibility": {
            "warp": ">=1.17",
            "devices": ["cuda"],
            "evidence_max_age_days": 365,
            "limitations": ["Synthetic test fixture."],
        },
        "artifacts": {
            "python_module": "synthetic.card.benchmark",
            "baseline": "before.py",
            "candidate": "after.py",
            "correctness": "test_correctness.py",
            "benchmark": "benchmark.py",
            "explanation": "explanation.md",
            "evidence": "evidence.json",
        },
        "benchmark": {
            "workload": {"size": 1024, "iterations": 4, "seed": 17},
            "estimated_peak_bytes": 16384,
            "warmups": 3,
            "pairs": 10,
            "bootstrap_seed": 17,
            "resamples": 10000,
        },
        "clean_room": {
            "synthetic": True,
            "derived_from_private_source": False,
            "declaration": "Independently authored synthetic fixture.",
        },
    }


def write_manifest(path, manifest):
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def make_fake_variant(label, outputs):
    return Variant(
        label=label,
        prepare_trial=lambda: None,
        run=lambda: None,
        synchronize=lambda: None,
        outputs=lambda: outputs,
    )


def make_fake_case(baseline, candidate, tolerances=None):
    if tolerances is None:
        tolerances = {name: Tolerance(atol=1.0e-6, rtol=1.0e-5) for name in baseline}
    return OptimizationCase(
        example_id="synthetic-case",
        workload={"size": 1},
        baseline=make_fake_variant("baseline", baseline),
        candidate=make_fake_variant("candidate", candidate),
        tolerances=tolerances,
    )


def make_evidence_record(
    *,
    timestamp="2026-07-29T00:00:00+00:00",
    device_alias="cpu",
    is_cuda=False,
    runtime_sources_dirty=False,
    candidate_ns=None,
):
    if candidate_ns is None:
        candidate_ns = (70, 71, 69, 72, 68, 71, 67, 70, 69, 68)
    samples = PairedSamples(
        baseline_ns=(100, 102, 98, 101, 99, 103, 97, 100, 102, 98),
        candidate_ns=candidate_ns,
        order=("baseline-first", "candidate-first") * 5,
    )
    summary = summarize_paired(samples, bootstrap_seed=17, resamples=10_000)
    correctness = CorrectnessResult(
        passed=True,
        outputs={
            "result": OutputError(
                name="result",
                max_abs=0.0,
                max_rel=0.0,
                finite=True,
                passed=True,
            )
        },
    )
    environment = {
        "timestamp_utc": timestamp,
        "python": "3.12.0",
        "warp": "1.17.0",
        "os": "Linux",
        "machine": "x86_64",
        "git": {
            "revision": "0123456789abcdef",
            "repository_dirty": False,
            "runtime_sources_dirty": runtime_sources_dirty,
        },
        "device": {
            "alias": device_alias,
            "name": "Synthetic device",
            "is_cuda": is_cuda,
            "architecture": 0,
            "total_memory_bytes": 1024,
        },
        "cuda": {"toolkit": [13, 0], "driver": [13, 0]},
        "workload": {"size": 1024, "iterations": 4, "seed": 17},
    }
    return build_evidence_record(
        example_id="synthetic-card",
        environment=environment,
        correctness=correctness,
        samples=samples,
        summary=summary,
        warmups=3,
        bootstrap_seed=17,
        resamples=10_000,
        limitations=["Synthetic test fixture."],
    )


def append_evidence_in_process(path, record, start_event, result_queue):
    try:
        if not start_event.wait(timeout=30):
            raise TimeoutError("append start event timed out")
        append_evidence(Path(path), record)
    except Exception as error:
        result_queue.put(repr(error))
    else:
        result_queue.put(None)


def append_evidence_for_manifest_in_process(path, record, manifest, start_event, result_queue):
    try:
        if not start_event.wait(timeout=30):
            raise TimeoutError("append start event timed out")
        append_evidence(Path(path), record, manifest=manifest)
    except Exception as error:
        result_queue.put(type(error).__name__)
    else:
        result_queue.put(None)


@contextmanager
def temporary_runner_card(*, case_atol=1.0e-6, case_rtol=1.0e-5, status="unverified"):
    optimization_root = Path(__file__).resolve().parents[1] / "examples" / "optimizations"
    with tempfile.TemporaryDirectory(prefix="synthetic_runner_card_", dir=optimization_root) as directory:
        card_root = Path(directory)
        package_name = card_root.name
        manifest = make_valid_manifest()
        manifest["id"] = "synthetic-runner-card"
        manifest["title"] = "Synthetic runner card"
        manifest["status"] = status
        manifest["summary"] = "Exercise the runtime optimization runner."
        manifest["compatibility"]["devices"] = ["cpu"]
        manifest["benchmark"] = {
            "workload": {"size": 4, "scale": 1.0},
            "estimated_peak_bytes": 32,
            "warmups": 3,
            "pairs": 10,
            "bootstrap_seed": 17,
            "resamples": 10000,
        }
        manifest["artifacts"]["python_module"] = f"warp.examples.optimizations.{package_name}.benchmark"

        (card_root / "__init__.py").write_text("", encoding="utf-8")
        (card_root / "before.py").write_text("# Synthetic baseline fixture.\n", encoding="utf-8")
        (card_root / "after.py").write_text("# Synthetic candidate fixture.\n", encoding="utf-8")
        (card_root / "test_correctness.py").write_text("# Exercised through the runner.\n", encoding="utf-8")
        (card_root / "explanation.md").write_text("# Synthetic runner fixture\n", encoding="utf-8")
        (card_root / "evidence.json").write_text('{"schema_version": 1, "records": []}\n', encoding="utf-8")
        (card_root / "benchmark.py").write_text(
            f"""
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from warp.examples.optimizations.harness.model import (
    JSONScalar,
    OptimizationCase,
    Tolerance,
    UnsupportedWorkload,
    Variant,
)


def _mark_execution():
    marker = os.environ.get("WARP_RUNNER_SENTINEL")
    if marker:
        Path(marker).write_text("executed\\n", encoding="utf-8")


def build_case(device: str, workload: Mapping[str, JSONScalar]) -> OptimizationCase:
    size = workload["size"]
    scale = workload["scale"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise UnsupportedWorkload("size must be a positive integer")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise UnsupportedWorkload("scale must be numeric")

    baseline_output = np.empty(size, dtype=np.float32)
    candidate_output = np.empty(size, dtype=np.float32)

    def prepare():
        _mark_execution()

    def run_baseline():
        baseline_output[:] = np.arange(size, dtype=np.float32) * float(scale)

    def run_candidate():
        candidate_output[:] = np.arange(size, dtype=np.float32) * float(scale)

    return OptimizationCase(
        example_id="synthetic-runner-card",
        workload=dict(workload),
        baseline=Variant(
            label="baseline",
            prepare_trial=prepare,
            run=run_baseline,
            synchronize=lambda: None,
            outputs=lambda: {{"result": baseline_output}},
        ),
        candidate=Variant(
            label="candidate",
            prepare_trial=prepare,
            run=run_candidate,
            synchronize=lambda: None,
            outputs=lambda: {{"result": candidate_output}},
        ),
        tolerances={{
            "result": Tolerance(atol={case_atol!r}, rtol={case_rtol!r}),
        }},
    )
""".lstrip(),
            encoding="utf-8",
        )
        (card_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        yield card_root, manifest


class TestOptimizationModels(unittest.TestCase):
    def test_case_requires_distinct_variant_labels(self):
        variant = Variant(
            label="same",
            prepare_trial=lambda: None,
            run=lambda: None,
            synchronize=lambda: None,
            outputs=lambda: {"result": np.ones(4, dtype=np.float32)},
        )
        with self.assertRaisesRegex(ValueError, "distinct"):
            OptimizationCase(
                example_id="synthetic-case",
                workload={"size": 4},
                baseline=variant,
                candidate=variant,
                tolerances={"result": Tolerance(atol=1.0e-6, rtol=1.0e-5)},
            )

    def test_case_requires_tolerance_for_each_output(self):
        baseline = make_fake_variant("baseline", {"result": np.ones(4)})
        candidate = make_fake_variant("candidate", {"result": np.ones(4)})
        case = OptimizationCase(
            example_id="synthetic-case",
            workload={"size": 4},
            baseline=baseline,
            candidate=candidate,
            tolerances={},
        )
        with self.assertRaisesRegex(ValueError, "result"):
            case.validate_output_contract()

    def test_case_requires_a_nonblank_id(self):
        with self.assertRaisesRegex(ValueError, "example_id"):
            OptimizationCase(
                example_id="   ",
                workload={"size": 4},
                baseline=make_fake_variant("baseline", {"result": np.ones(4)}),
                candidate=make_fake_variant("candidate", {"result": np.ones(4)}),
                tolerances={"result": Tolerance(atol=1.0e-6, rtol=1.0e-5)},
            )

    def test_case_requires_matching_output_keys(self):
        case = OptimizationCase(
            example_id="synthetic-case",
            workload={"size": 4},
            baseline=make_fake_variant("baseline", {"baseline_result": np.ones(4)}),
            candidate=make_fake_variant("candidate", {"candidate_result": np.ones(4)}),
            tolerances={
                "baseline_result": Tolerance(atol=1.0e-6, rtol=1.0e-5),
                "candidate_result": Tolerance(atol=1.0e-6, rtol=1.0e-5),
            },
        )

        with self.assertRaisesRegex(ValueError, "output keys"):
            case.validate_output_contract()


class TestOptimizationHarness(unittest.TestCase):
    def test_correctness_snapshots_outputs_before_running_next_variant(self):
        shared_output = np.zeros(1, dtype=np.float32)
        case = OptimizationCase(
            example_id="shared-output-case",
            workload={"size": 1},
            baseline=Variant(
                label="baseline",
                prepare_trial=lambda: None,
                run=lambda: shared_output.fill(1.0),
                synchronize=lambda: None,
                outputs=lambda: {"result": shared_output},
            ),
            candidate=Variant(
                label="candidate",
                prepare_trial=lambda: None,
                run=lambda: shared_output.fill(2.0),
                synchronize=lambda: None,
                outputs=lambda: {"result": shared_output},
            ),
            tolerances={"result": Tolerance(atol=0.0, rtol=0.0)},
        )

        result = check_correctness(case)

        self.assertFalse(result.passed)
        self.assertEqual(result.outputs["result"].max_abs, 1.0)

    def test_correctness_rejects_mismatched_output_keys(self):
        case = make_fake_case(
            baseline={"baseline_result": np.array([1.0])},
            candidate={"candidate_result": np.array([1.0])},
            tolerances={"baseline_result": Tolerance(atol=1.0e-6, rtol=1.0e-5)},
        )

        with self.assertRaisesRegex(ValueError, "output keys"):
            check_correctness(case)

    def test_correctness_rejects_nonfinite_candidate(self):
        case = make_fake_case(
            baseline={"result": np.array([1.0], dtype=np.float32)},
            candidate={"result": np.array([np.nan], dtype=np.float32)},
        )
        result = check_correctness(case)
        self.assertFalse(result.passed)
        self.assertFalse(result.outputs["result"].finite)

    def test_correctness_rejects_nonfinite_baseline(self):
        case = make_fake_case(
            baseline={"result": np.array([np.inf], dtype=np.float32)},
            candidate={"result": np.array([1.0], dtype=np.float32)},
        )
        result = check_correctness(case)
        self.assertFalse(result.passed)
        self.assertFalse(result.outputs["result"].finite)

    def test_correctness_handles_scalar_and_array_outputs(self):
        case = make_fake_case(
            baseline={"scalar": 2.0, "array": np.array([0.0, 2.0])},
            candidate={"scalar": 2.05, "array": np.array([0.05, 2.2])},
            tolerances={
                "scalar": Tolerance(atol=0.1, rtol=0.0),
                "array": Tolerance(atol=0.1, rtol=0.0),
            },
        )

        result = check_correctness(case)

        self.assertTrue(result.outputs["scalar"].passed)
        self.assertAlmostEqual(result.outputs["scalar"].max_abs, 0.05)
        self.assertAlmostEqual(result.outputs["array"].max_abs, 0.2)
        self.assertAlmostEqual(result.outputs["array"].max_rel, 0.5)

    def test_correctness_rejects_shape_and_dtype_family_mismatches(self):
        cases = (
            (
                make_fake_case(
                    baseline={"result": np.array([1.0])},
                    candidate={"result": np.array([[1.0]])},
                ),
                "shape",
            ),
            (
                make_fake_case(
                    baseline={"result": np.array([1.0], dtype=np.float32)},
                    candidate={"result": np.array([1], dtype=np.int32)},
                ),
                "dtype family",
            ),
        )

        for case, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                check_correctness(case)

    def test_correctness_accepts_different_widths_in_same_dtype_family(self):
        case = make_fake_case(
            baseline={"result": np.array([1.0], dtype=np.float32)},
            candidate={"result": np.array([1.0], dtype=np.float64)},
        )

        self.assertTrue(check_correctness(case).passed)

    def test_correctness_uses_per_output_tolerance(self):
        case = make_fake_case(
            baseline={"loose": np.array([1.0]), "strict": np.array([1.0])},
            candidate={"loose": np.array([1.01]), "strict": np.array([1.01])},
            tolerances={
                "loose": Tolerance(atol=0.02, rtol=0.0),
                "strict": Tolerance(atol=0.001, rtol=0.0),
            },
        )
        result = check_correctness(case)
        self.assertTrue(result.outputs["loose"].passed)
        self.assertFalse(result.outputs["strict"].passed)

    def test_run_paired_warms_up_and_alternates_measurement_order(self):
        events = []

        def make_recording_variant(label):
            return Variant(
                label=label,
                prepare_trial=lambda: events.append(f"{label}:prepare"),
                run=lambda: events.append(f"{label}:run"),
                synchronize=lambda: events.append(f"{label}:synchronize"),
                outputs=lambda: {"result": np.array([1.0])},
            )

        case = OptimizationCase(
            example_id="timing-case",
            workload={"size": 1},
            baseline=make_recording_variant("baseline"),
            candidate=make_recording_variant("candidate"),
            tolerances={"result": Tolerance(atol=0.0, rtol=0.0)},
        )
        clock_value = 0

        def timer_ns():
            nonlocal clock_value
            events.append("timer")
            clock_value += 5
            return clock_value

        samples = run_paired(case, warmups=3, pairs=10, timer_ns=timer_ns)

        expected_warmups = []
        for label in ("baseline", "candidate"):
            for _ in range(3):
                expected_warmups.extend((f"{label}:prepare", f"{label}:run", f"{label}:synchronize"))
        self.assertEqual(events[:18], expected_warmups)
        self.assertEqual(
            events[18:30],
            [
                "baseline:prepare",
                "baseline:synchronize",
                "timer",
                "baseline:run",
                "baseline:synchronize",
                "timer",
                "candidate:prepare",
                "candidate:synchronize",
                "timer",
                "candidate:run",
                "candidate:synchronize",
                "timer",
            ],
        )
        self.assertEqual(samples.baseline_ns, (5,) * 10)
        self.assertEqual(samples.candidate_ns, (5,) * 10)
        self.assertEqual(samples.order, ("baseline-first", "candidate-first") * 5)

    def test_run_paired_rejects_insufficient_warmups_or_pairs(self):
        case = make_fake_case(
            baseline={"result": np.array([1.0])},
            candidate={"result": np.array([1.0])},
        )

        with self.assertRaisesRegex(ValueError, "warmups"):
            run_paired(case, warmups=2, pairs=10)
        with self.assertRaisesRegex(ValueError, "pairs"):
            run_paired(case, warmups=3, pairs=9)

    def test_run_paired_rejects_nonpositive_elapsed_time(self):
        case = make_fake_case(
            baseline={"result": np.array([1.0])},
            candidate={"result": np.array([1.0])},
        )

        with self.assertRaisesRegex(ValueError, "positive"):
            run_paired(case, warmups=3, pairs=10, timer_ns=lambda: 7)

    def test_statistics_rejects_invalid_samples(self):
        valid_order = ("baseline-first", "candidate-first") * 5
        invalid_samples = (
            PairedSamples((1,) * 9, (1,) * 9, valid_order[:9]),
            PairedSamples((0,) + (1,) * 9, (1,) * 10, valid_order),
            PairedSamples((-1,) + (1,) * 9, (1,) * 10, valid_order),
            PairedSamples((1,) * 10, (float("inf"),) + (1,) * 9, valid_order),
        )

        for samples in invalid_samples:
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                summarize_paired(samples, bootstrap_seed=17, resamples=10_000)

    def test_statistics_returns_hand_derived_descriptive_values(self):
        samples = PairedSamples(
            baseline_ns=(100,) * 10,
            candidate_ns=(50, 60, 70, 80, 90, 100, 110, 120, 130, 140),
            order=("baseline-first", "candidate-first") * 5,
        )

        summary = summarize_paired(samples, bootstrap_seed=17, resamples=10_000)

        self.assertEqual(summary.baseline_median_ns, 100.0)
        self.assertEqual(summary.candidate_median_ns, 95.0)
        self.assertEqual(summary.baseline_mad_ns, 0.0)
        self.assertEqual(summary.candidate_mad_ns, 25.0)
        self.assertEqual(summary.median_ratio, 0.95)
        self.assertEqual(summary.pairs, 10)
        self.assertEqual(
            set(summary.as_dict()),
            {
                "baseline_median_ns",
                "candidate_median_ns",
                "baseline_mad_ns",
                "candidate_mad_ns",
                "median_ratio",
                "ratio_ci_low",
                "ratio_ci_high",
                "pairs",
            },
        )
        self.assertTrue(all(type(value) in (int, float) for value in summary.as_dict().values()))

    def test_fixed_bootstrap_seed_returns_identical_bounds(self):
        samples = PairedSamples(
            baseline_ns=(100, 102, 98, 101, 99, 103, 97, 100, 102, 98),
            candidate_ns=(70, 71, 69, 72, 68, 71, 67, 70, 69, 68),
            order=("baseline-first", "candidate-first") * 5,
        )

        first = summarize_paired(samples, bootstrap_seed=17, resamples=10_000)
        second = summarize_paired(samples, bootstrap_seed=17, resamples=10_000)

        self.assertEqual(first.ratio_ci_low, second.ratio_ci_low)
        self.assertEqual(first.ratio_ci_high, second.ratio_ci_high)

    def test_faster_sequence_has_interval_below_parity(self):
        samples = PairedSamples(
            baseline_ns=(100, 102, 98, 101, 99, 103, 97, 100, 102, 98),
            candidate_ns=(70, 71, 69, 72, 68, 71, 67, 70, 69, 68),
            order=("baseline-first", "candidate-first") * 5,
        )
        summary = summarize_paired(samples, bootstrap_seed=17, resamples=10_000)
        self.assertLess(summary.ratio_ci_high, 1.0)

    def test_overlapping_sequence_crosses_parity(self):
        samples = PairedSamples(
            baseline_ns=(100,) * 10,
            candidate_ns=(90, 92, 94, 96, 98, 102, 104, 106, 108, 110),
            order=("baseline-first", "candidate-first") * 5,
        )

        summary = summarize_paired(samples, bootstrap_seed=17, resamples=10_000)

        self.assertLess(summary.ratio_ci_low, 1.0)
        self.assertGreater(summary.ratio_ci_high, 1.0)


class TestOptimizationManifests(unittest.TestCase):
    def test_manifest_rejects_unknown_status(self):
        manifest = make_valid_manifest()
        manifest["status"] = "fast"
        with self.assertRaisesRegex(ValueError, "status"):
            validate_manifest(manifest)

    def test_manifest_requires_clean_room_declaration(self):
        manifest = make_valid_manifest()
        del manifest["clean_room"]
        with self.assertRaisesRegex(ValueError, "clean_room"):
            validate_manifest(manifest)

    def test_manifest_rejects_schema_parity_violations(self):
        violations = (
            ("schema version", lambda manifest: manifest.__setitem__("schema_version", True), "schema_version"),
            (
                "Windows absolute artifact path",
                lambda manifest: manifest["artifacts"].__setitem__("baseline", r"C:\\outside.py"),
                "artifacts.baseline",
            ),
            (
                "parent traversal artifact path",
                lambda manifest: manifest["artifacts"].__setitem__("baseline", "../before.py"),
                "artifacts.baseline",
            ),
            ("warmup minimum", lambda manifest: manifest["benchmark"].__setitem__("warmups", 2), "warmups"),
            ("pair minimum", lambda manifest: manifest["benchmark"].__setitem__("pairs", 9), "pairs"),
            (
                "negative bootstrap seed",
                lambda manifest: manifest["benchmark"].__setitem__("bootstrap_seed", -1),
                "bootstrap_seed",
            ),
            ("resample minimum", lambda manifest: manifest["benchmark"].__setitem__("resamples", 9999), "resamples"),
        )

        for description, mutate, field in violations:
            with self.subTest(description=description):
                manifest = deepcopy(make_valid_manifest())
                mutate(manifest)
                with self.assertRaisesRegex(ValueError, field):
                    validate_manifest(manifest)

    def test_manifest_rejects_nonfinite_tolerances_and_workload_values(self):
        violations = (
            (
                "absolute tolerance",
                lambda manifest: manifest["semantics"]["tolerance"].__setitem__("absolute", float("nan")),
                "absolute",
            ),
            (
                "relative tolerance",
                lambda manifest: manifest["semantics"]["tolerance"].__setitem__("relative", float("inf")),
                "relative",
            ),
            (
                "workload",
                lambda manifest: manifest["benchmark"]["workload"].__setitem__("scale", float("-inf")),
                "workload",
            ),
        )
        for description, mutate, field in violations:
            with self.subTest(description=description):
                manifest = make_valid_manifest()
                mutate(manifest)
                with self.assertRaisesRegex(ValueError, field):
                    validate_manifest(manifest)

    def test_manifest_loader_rejects_duplicate_keys_and_nonstandard_numbers(self):
        documents = (
            '{"schema_version": 1, "schema_version": 1}',
            '{"schema_version": NaN}',
            '{"schema_version": Infinity}',
            '{"schema_version": -Infinity}',
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            for document in documents:
                with self.subTest(document=document):
                    manifest_path.write_text(document, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "duplicate JSON key|non-standard JSON"):
                        load_manifest(manifest_path)

    def test_manifest_schema_requires_nonnegative_bootstrap_seed(self):
        schema_path = (
            Path(__file__).resolve().parents[1] / "examples" / "optimizations" / "schema" / "example.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertEqual(schema["properties"]["benchmark"]["properties"]["bootstrap_seed"].get("minimum"), 0)

    def test_schema_relative_path_grammar_rejects_windows_absolute_paths(self):
        schema_path = (
            Path(__file__).resolve().parents[1] / "examples" / "optimizations" / "schema" / "example.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        pattern = schema["$defs"]["relativePath"]["pattern"]

        self.assertIsNone(re.fullmatch(pattern, r"C:\outside.py"))
        self.assertIsNone(re.fullmatch(pattern, "../before.py"))
        self.assertIsNotNone(re.fullmatch(pattern, "before.py"))

    def test_registry_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_manifest(root / "first" / "manifest.json", make_valid_manifest())
            write_manifest(root / "second" / "manifest.json", make_valid_manifest())
            with self.assertRaisesRegex(ValueError, "duplicate"):
                discover_examples(root)

    def test_clean_room_scan_uses_external_patterns(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "example.py"
            source.write_text("synthetic_forbidden_token = 1\n", encoding="utf-8")
            findings = scan_prohibited(Path(directory), ["synthetic_forbidden_token"])
            self.assertEqual(findings[0].path, source)


class TestOptimizationEvidence(unittest.TestCase):
    def test_cpu_environment_contains_only_json_serializable_public_facts(self):
        environment = capture_environment("cpu", {"size": 1024, "optional_packages": []})

        serialized = json.dumps(environment)

        self.assertIn('"alias": "cpu"', serialized)
        self.assertFalse(environment["device"]["is_cuda"])
        self.assertEqual(environment["workload"], {"size": 1024, "optional_packages": []})
        self.assertEqual(
            set(environment),
            {"timestamp_utc", "python", "warp", "os", "machine", "git", "device", "cuda", "workload"},
        )

    def test_environment_omits_git_for_untracked_or_outside_package_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "unrelated-repository"
            repository.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            (repository / ".gitignore").write_text(".venv/\n", encoding="utf-8")
            ignored_source = (
                repository
                / ".venv"
                / "lib"
                / "site-packages"
                / "warp"
                / "examples"
                / "optimizations"
                / "harness"
                / "environment.py"
            )
            ignored_source.parent.mkdir(parents=True)
            ignored_source.write_text("# installed wheel\n", encoding="utf-8")
            outside_source = root / "outside-git" / "warp" / "examples" / "optimizations" / "harness" / "environment.py"
            outside_source.parent.mkdir(parents=True)
            outside_source.write_text("# installed wheel\n", encoding="utf-8")

            for description, source in (
                ("ignored virtual environment", ignored_source),
                ("outside Git", outside_source),
            ):
                with self.subTest(description=description), patch.object(environment_module, "__file__", str(source)):
                    git = capture_environment("cpu", {"size": 1})["git"]
                    self.assertEqual(
                        git,
                        {
                            "revision": None,
                            "repository_dirty": None,
                            "runtime_sources_dirty": None,
                        },
                    )

    def test_classifies_interval_below_parity_as_improved(self):
        summary = PairedSummary(100.0, 80.0, 1.0, 1.0, 0.8, 0.7, 0.9, 10)

        self.assertEqual(classify_summary(summary), "improved")

    def test_classifies_interval_above_parity_as_harmful(self):
        summary = PairedSummary(100.0, 120.0, 1.0, 1.0, 1.2, 1.1, 1.3, 10)

        self.assertEqual(classify_summary(summary), "harmful")

    def test_classifies_interval_crossing_parity_as_inconclusive(self):
        summary = PairedSummary(100.0, 100.0, 1.0, 1.0, 1.0, 0.9, 1.1, 10)

        self.assertEqual(classify_summary(summary), "inconclusive")

    def test_classifies_intervals_touching_parity_as_inconclusive(self):
        summaries = (
            PairedSummary(100.0, 90.0, 1.0, 1.0, 0.9, 0.8, 1.0, 10),
            PairedSummary(100.0, 110.0, 1.0, 1.0, 1.1, 1.0, 1.2, 10),
        )

        for summary in summaries:
            with self.subTest(summary=summary):
                self.assertEqual(classify_summary(summary), "inconclusive")

    def test_evidence_rejects_mismatched_sample_lengths(self):
        manifest = make_valid_manifest()
        record = make_evidence_record()
        record["samples"]["candidate_ns"].pop()

        with self.assertRaisesRegex(ValueError, "matching lengths"):
            validate_evidence_record(record, manifest)

    def test_evidence_rejects_fewer_pairs_than_manifest(self):
        manifest = make_valid_manifest()
        manifest["benchmark"]["pairs"] = 11
        record = make_evidence_record()

        with self.assertRaisesRegex(ValueError, "manifest requires at least 11"):
            validate_evidence_record(record, manifest)

    def test_evidence_accepts_overridden_workload_with_declared_keys(self):
        manifest = make_valid_manifest()
        record = make_evidence_record()
        record["environment"]["workload"] = {"size": 2048, "iterations": 7, "seed": 29}

        try:
            validate_evidence_record(record, manifest)
        except ValueError as error:
            self.fail(str(error))

    def test_evidence_rejects_workload_key_drift(self):
        manifest = make_valid_manifest()
        for workload in (
            {"size": 2048, "iterations": 7},
            {"size": 2048, "iterations": 7, "seed": 29, "extra": True},
        ):
            with self.subTest(workload=workload):
                record = make_evidence_record()
                record["environment"]["workload"] = workload
                with self.assertRaisesRegex(ValueError, "workload keys"):
                    validate_evidence_record(record, manifest)

    def test_recommended_manifest_requires_passing_cuda_evidence(self):
        manifest = make_valid_manifest()
        manifest["status"] = "recommended"
        record = make_evidence_record()

        with self.assertRaisesRegex(ValueError, "non-stale improved CUDA"):
            validate_evidence_document({"schema_version": 1, "records": [record]}, manifest)

    def test_record_validation_does_not_apply_document_claim_gate(self):
        manifest = make_valid_manifest()
        manifest["status"] = "recommended"

        validate_evidence_record(make_evidence_record(), manifest)

    def test_temp_history_validation_accepts_non_supporting_records(self):
        cases = (
            ("recommended", make_evidence_record()),
            (
                "conditional",
                make_evidence_record(
                    device_alias="cuda:0",
                    is_cuda=True,
                    candidate_ns=(90, 92, 94, 96, 98, 102, 104, 106, 108, 110),
                ),
            ),
        )

        for status, record in cases:
            with self.subTest(status=status, result=record["result"]):
                manifest = make_valid_manifest()
                manifest["status"] = status
                try:
                    validate_evidence_document(
                        {"schema_version": 1, "records": [record]},
                        manifest,
                        require_claim_support=False,
                    )
                except (TypeError, ValueError) as error:
                    self.fail(str(error))

    def test_manifest_append_accepts_standalone_inconclusive_cuda_record(self):
        manifest = make_valid_manifest()
        manifest["status"] = "conditional"
        record = make_evidence_record(
            device_alias="cuda:0",
            is_cuda=True,
            candidate_ns=(90, 92, 94, 96, 98, 102, 104, 106, 108, 110),
        )
        self.assertEqual(record["result"], "inconclusive")

        with tempfile.TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "evidence.json"
            evidence_path.write_text('{"schema_version": 1, "records": []}\n', encoding="utf-8")

            try:
                append_evidence(evidence_path, record, manifest=manifest)
            except ValueError as error:
                self.fail(str(error))

            document = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(document["records"], [record])
            with self.assertRaisesRegex(ValueError, "non-stale improved CUDA"):
                validate_evidence_document(document, manifest)

    def test_manifest_append_rejects_invalid_history_atomically(self):
        manifest = make_valid_manifest()
        manifest["status"] = "recommended"
        supporting = make_evidence_record(device_alias="cuda:0", is_cuda=True)
        mixed_card = make_evidence_record()
        mixed_card["example_id"] = "other-synthetic-card"
        duplicate = make_evidence_record()
        invalid_workload = make_evidence_record()
        invalid_workload["environment"]["workload"].pop("seed")
        histories = (
            '{"schema_version": 1, "records": [',
            json.dumps({"schema_version": 1, "records": [mixed_card]}),
            json.dumps({"schema_version": 1, "records": [duplicate, deepcopy(duplicate)]}),
            json.dumps({"schema_version": 1, "records": [invalid_workload]}),
        )

        with tempfile.TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "evidence.json"
            for index, history in enumerate(histories):
                with self.subTest(index=index):
                    evidence_path.write_text(history, encoding="utf-8")
                    before = evidence_path.read_bytes()

                    with self.assertRaises(ValueError):
                        append_evidence(evidence_path, supporting, manifest=manifest)

                    self.assertEqual(evidence_path.read_bytes(), before)

    def test_document_claim_accepts_history_with_one_supporting_record(self):
        manifest = make_valid_manifest()
        manifest["status"] = "recommended"
        manifest["compatibility"]["evidence_max_age_days"] = 30
        now = datetime(2026, 7, 29, tzinfo=timezone.utc)
        records = [
            make_evidence_record(timestamp=now.isoformat()),
            make_evidence_record(
                timestamp="2025-01-01T00:00:00+00:00",
                device_alias="cuda:0",
                is_cuda=True,
            ),
            make_evidence_record(
                timestamp=now.isoformat(),
                device_alias="cuda:0",
                is_cuda=True,
                candidate_ns=(90, 92, 94, 96, 98, 102, 104, 106, 108, 110),
            ),
            make_evidence_record(
                timestamp=now.isoformat(),
                device_alias="cuda:0",
                is_cuda=True,
            ),
        ]

        validate_evidence_document({"schema_version": 1, "records": records}, manifest, now=now)

    def test_append_preserves_first_record_when_adding_second(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "evidence.json"
            first = make_evidence_record(timestamp="2026-07-28T00:00:00+00:00")
            second = make_evidence_record(timestamp="2026-07-29T00:00:00+00:00")

            append_evidence(evidence_path, first)
            first_document = json.loads(evidence_path.read_text(encoding="utf-8"))
            append_evidence(evidence_path, second)
            second_document = json.loads(evidence_path.read_text(encoding="utf-8"))

            self.assertEqual(first_document["records"][0], second_document["records"][0])
            self.assertEqual(
                [record["record_id"] for record in second_document["records"]],
                [first["record_id"], second["record_id"]],
            )

    def test_append_rejects_duplicate_ids_already_in_history(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "evidence.json"
            first = make_evidence_record()
            duplicate_history = {"schema_version": 1, "records": [first, deepcopy(first)]}
            evidence_path.write_text(json.dumps(duplicate_history), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate evidence record ID"):
                append_evidence(evidence_path, make_evidence_record())

    def test_concurrent_appends_preserve_every_record(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "evidence.json"
            evidence_path.write_text('{"schema_version": 1, "records": []}\n', encoding="utf-8")
            context = multiprocessing.get_context("spawn")
            start_event = context.Event()
            result_queue = context.Queue()
            records = [make_evidence_record() for _ in range(8)]
            processes = [
                context.Process(
                    target=append_evidence_in_process,
                    args=(str(evidence_path), record, start_event, result_queue),
                )
                for record in records
            ]
            for process in processes:
                process.start()
            start_event.set()
            for process in processes:
                process.join(timeout=30)

            self.assertTrue(all(not process.is_alive() for process in processes))
            self.assertEqual([result_queue.get(timeout=5) for _ in processes], [None] * len(processes))
            document = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {record["record_id"] for record in document["records"]},
                {record["record_id"] for record in records},
            )

    def test_manifest_append_keeps_exploratory_record_with_supporting_history(self):
        manifest = make_valid_manifest()
        manifest["status"] = "recommended"
        supporting = make_evidence_record(device_alias="cuda:0", is_cuda=True)
        exploratory = make_evidence_record(
            candidate_ns=(90, 92, 94, 96, 98, 102, 104, 106, 108, 110),
        )

        with tempfile.TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "evidence.json"
            evidence_path.write_text(
                json.dumps({"schema_version": 1, "records": [supporting]}),
                encoding="utf-8",
            )

            try:
                append_evidence(evidence_path, exploratory, manifest=manifest)
            except TypeError as error:
                self.fail(str(error))

            document = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {record["record_id"] for record in document["records"]},
                {supporting["record_id"], exploratory["record_id"]},
            )
            validate_evidence_document(document, manifest)

    def test_manifest_append_rejects_mixed_card_history_under_writer_lock(self):
        manifest = make_valid_manifest()
        valid = make_evidence_record()
        other_manifest = deepcopy(manifest)
        other_manifest["id"] = "other-synthetic-card"
        wrong_card = make_evidence_record()
        wrong_card["example_id"] = "other-synthetic-card"

        with tempfile.TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "evidence.json"
            context = multiprocessing.get_context("spawn")
            start_event = context.Event()
            result_queue = context.Queue()
            processes = [
                context.Process(
                    target=append_evidence_for_manifest_in_process,
                    args=(
                        str(evidence_path),
                        record,
                        selected_manifest,
                        start_event,
                        result_queue,
                    ),
                )
                for record, selected_manifest in (
                    (valid, manifest),
                    (wrong_card, other_manifest),
                )
            ]
            for process in processes:
                process.start()
            start_event.set()
            for process in processes:
                process.join(timeout=30)

            self.assertTrue(all(not process.is_alive() for process in processes))
            results = [result_queue.get(timeout=5) for _ in processes]
            self.assertEqual(sorted(results, key=str), [None, "ValueError"])
            document = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(len(document["records"]), 1)
            retained = document["records"][0]
            self.assertIn(retained["record_id"], {valid["record_id"], wrong_card["record_id"]})
            selected_manifest = manifest if retained["record_id"] == valid["record_id"] else other_manifest
            validate_evidence_document(document, selected_manifest)

    def test_evidence_recomputes_and_rejects_tampered_statistics(self):
        manifest = make_valid_manifest()
        record = make_evidence_record()
        record["statistics"]["median_ratio"] = 0.5

        with self.assertRaisesRegex(ValueError, "statistics.median_ratio"):
            validate_evidence_record(record, manifest)

    def test_evidence_rejects_tampered_result_classification(self):
        record = make_evidence_record()
        record["result"] = "harmful"

        with self.assertRaisesRegex(ValueError, "result"):
            validate_evidence_record(record, make_valid_manifest())

    def test_evidence_rejects_negative_correctness_errors(self):
        record = make_evidence_record()
        record["correctness"]["outputs"]["result"]["max_abs"] = -0.1

        with self.assertRaisesRegex(ValueError, "max_abs"):
            validate_evidence_record(record, make_valid_manifest())

    def test_evidence_rejects_passed_nonfinite_output(self):
        record = make_evidence_record()
        record["correctness"]["outputs"]["result"]["finite"] = False

        with self.assertRaisesRegex(ValueError, "finite"):
            validate_evidence_record(record, make_valid_manifest())

    def test_evidence_rejects_invalid_cuda_shape_and_versions(self):
        cases = (
            ("extra key", lambda cuda: cuda.__setitem__("device_path", "/secret"), "exactly"),
            ("string toolkit", lambda cuda: cuda.__setitem__("toolkit", "13.0"), "toolkit"),
            ("boolean version", lambda cuda: cuda.__setitem__("driver", [True, 0]), "driver"),
        )

        for description, mutate, message in cases:
            with self.subTest(description=description):
                record = make_evidence_record()
                mutate(record["environment"]["cuda"])
                with self.assertRaisesRegex(ValueError, message):
                    validate_evidence_record(record, make_valid_manifest())

    def test_evidence_rejects_invalid_device_architecture_type(self):
        for architecture in (True, ["sm_120"]):
            with self.subTest(architecture=architecture):
                record = make_evidence_record()
                record["environment"]["device"]["architecture"] = architecture
                with self.assertRaisesRegex(ValueError, "architecture"):
                    validate_evidence_record(record, make_valid_manifest())

    def test_evidence_rejects_non_scalar_workload_values(self):
        record = make_evidence_record()
        record["environment"]["workload"]["nested"] = {"value": 1}

        with self.assertRaisesRegex(ValueError, "JSON scalar"):
            validate_evidence_record(record, make_valid_manifest())

    def test_evidence_rejects_non_alternating_pair_order(self):
        record = make_evidence_record()
        record["samples"]["order"][1] = "baseline-first"

        with self.assertRaisesRegex(ValueError, "alternate"):
            validate_evidence_record(record, make_valid_manifest())

    def test_evidence_becomes_stale_after_manifest_maximum_age(self):
        manifest = make_valid_manifest()
        manifest["compatibility"]["evidence_max_age_days"] = 30
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        record = make_evidence_record(timestamp=timestamp.isoformat())

        self.assertFalse(is_evidence_stale(record, manifest, now=timestamp + timedelta(days=30)))
        self.assertTrue(is_evidence_stale(record, manifest, now=timestamp + timedelta(days=30, seconds=1)))

    def test_evidence_requires_canonical_utc_timestamp(self):
        for timestamp in ("2026-07-29T00:00:00Z", "2026-07-29T01:00:00+01:00"):
            with self.subTest(timestamp=timestamp):
                record = make_evidence_record()
                record["environment"]["timestamp_utc"] = timestamp
                with self.assertRaisesRegex(ValueError, "canonical UTC"):
                    validate_evidence_record(record, make_valid_manifest())

    def test_evidence_rejects_implausibly_future_timestamp_with_explicit_skew(self):
        now = datetime(2026, 7, 29, tzinfo=timezone.utc)
        record = make_evidence_record(timestamp=(now + timedelta(minutes=6)).isoformat())

        with self.assertRaisesRegex(ValueError, "future"):
            validate_evidence_record(record, make_valid_manifest(), now=now)
        validate_evidence_record(
            record,
            make_valid_manifest(),
            now=now,
            future_skew=timedelta(minutes=10),
        )

    def test_append_sorts_records_by_parsed_timestamp_then_id(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "evidence.json"
            later = make_evidence_record(timestamp="2026-07-29T00:00:01+00:00")
            earlier = make_evidence_record(timestamp="2026-07-29T00:00:00+00:00")
            same_time_first_id = make_evidence_record(timestamp="2026-07-29T00:00:00+00:00")
            same_time_first_id["record_id"] = "00000000000000000000000000000000"

            append_evidence(evidence_path, later)
            append_evidence(evidence_path, earlier)
            append_evidence(evidence_path, same_time_first_id)

            document = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [record["record_id"] for record in document["records"]],
                [same_time_first_id["record_id"], earlier["record_id"], later["record_id"]],
            )

    def test_recommended_manifest_rejects_stale_improved_cuda_evidence(self):
        manifest = make_valid_manifest()
        manifest["status"] = "recommended"
        manifest["compatibility"]["evidence_max_age_days"] = 30
        record = make_evidence_record(
            timestamp="2025-01-01T00:00:00+00:00",
            device_alias="cuda:0",
            is_cuda=True,
        )

        with self.assertRaisesRegex(ValueError, "non-stale improved CUDA"):
            validate_evidence_document({"schema_version": 1, "records": [record]}, manifest)

    def test_recommended_manifest_requires_clean_runtime_sources(self):
        manifest = make_valid_manifest()
        manifest["status"] = "conditional"
        record = make_evidence_record(
            device_alias="cuda:0",
            is_cuda=True,
            runtime_sources_dirty=True,
        )

        with self.assertRaisesRegex(ValueError, "non-stale improved CUDA"):
            validate_evidence_document({"schema_version": 1, "records": [record]}, manifest)


class TestOptimizationRunner(unittest.TestCase):
    def run_runner(self, *arguments, environment=None):
        process_environment = os.environ.copy()
        if environment is not None:
            process_environment.update(environment)
        return subprocess.run(
            [sys.executable, "-m", "warp.examples.optimizations.run", *arguments],
            capture_output=True,
            check=False,
            env=process_environment,
            text=True,
        )

    def test_list_prints_header(self):
        result = self.run_runner("list")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ID", result.stdout)
        self.assertIn("STATUS", result.stdout)

    def test_validate_checks_both_schema_documents(self):
        result = self.run_runner("validate")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("example.schema.json: valid", result.stdout)
        self.assertIn("evidence.schema.json: valid", result.stdout)

    def test_validate_applies_external_deny_patterns(self):
        with tempfile.TemporaryDirectory() as directory:
            pattern_path = Path(directory) / "deny-patterns.txt"
            pattern_path.write_text("OptimizationCase\n", encoding="utf-8")

            result = self.run_runner("validate", "--deny-pattern-file", str(pattern_path))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("prohibited pattern", result.stderr)
        self.assertNotIn("OptimizationCase", result.stderr)

    def test_benchmark_rejects_protocol_values_below_evidence_minimums(self):
        arguments = (
            ("--pairs", "9", "at least 10"),
            ("--resamples", "9999", "at least 10000"),
        )
        for option, value, message in arguments:
            with self.subTest(option=option):
                result = self.run_runner(
                    "benchmark",
                    "--example",
                    "missing-card",
                    "--device",
                    "cpu",
                    option,
                    value,
                    "--output",
                    str(Path(tempfile.gettempdir()) / "unused-optimization-evidence.json"),
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_benchmark_rejects_negative_bootstrap_seed_before_card_selection(self):
        result = self.run_runner(
            "benchmark",
            "--example",
            "missing-card",
            "--device",
            "cpu",
            "--bootstrap-seed",
            "-1",
            "--output",
            str(Path(tempfile.gettempdir()) / "unused-optimization-evidence.json"),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("at least 0", result.stderr)
        self.assertNotIn("unknown optimization example", result.stderr)

    def test_benchmark_requires_an_explicit_output_path(self):
        result = self.run_runner(
            "benchmark",
            "--example",
            "missing-card",
            "--device",
            "cpu",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--output", result.stderr)

    def test_workload_overrides_parse_json_scalars_deterministically(self):
        script = "\n".join(
            (
                "import json",
                "from warp.examples.optimizations.run import _apply_workload_overrides",
                "workload = {'size': 1, 'ratio': 1.0, 'enabled': False, 'label': '', 'optional': 1}",
                "overrides = ['size=4096', 'ratio=1.25', 'enabled=true', 'label=wide', 'optional=null']",
                "print(json.dumps(_apply_workload_overrides(workload, overrides), sort_keys=True))",
            )
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "enabled": True,
                "label": "wide",
                "optional": None,
                "ratio": 1.25,
                "size": 4096,
            },
        )

    def test_workload_overrides_reject_unknown_and_duplicate_keys(self):
        for overrides, message in (
            (["unknown=1"], "unknown workload key"),
            (["size=2", "size=3"], "duplicate workload override"),
        ):
            with self.subTest(overrides=overrides):
                script = "\n".join(
                    (
                        "from warp.examples.optimizations.run import _apply_workload_overrides",
                        f"_apply_workload_overrides({{'size': 1}}, {overrides!r})",
                    )
                )
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    capture_output=True,
                    check=False,
                    text=True,
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_temporary_card_check_succeeds(self):
        with temporary_runner_card() as (_, manifest):
            result = self.run_runner(
                "check",
                "--example",
                manifest["id"],
                "--device",
                "cpu",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Correctness: PASS", result.stdout)
        self.assertIn("result", result.stdout)

    def test_temporary_card_validation_succeeds(self):
        with temporary_runner_card() as (_, manifest):
            result = self.run_runner("validate")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"{manifest['id']}: valid", result.stdout)

    def test_benchmark_override_produces_valid_card_evidence(self):
        with temporary_runner_card() as (card_root, manifest):
            evidence_path = card_root / "evidence.json"
            benchmark = self.run_runner(
                "benchmark",
                "--example",
                manifest["id"],
                "--device",
                "cpu",
                "--set",
                "size=8",
                "--output",
                str(evidence_path),
            )
            document = json.loads(evidence_path.read_text(encoding="utf-8"))
            validation = self.run_runner("validate")

        self.assertEqual(benchmark.returncode, 0, benchmark.stderr)
        self.assertEqual(document["records"][0]["environment"]["workload"], {"size": 8, "scale": 1.0})
        self.assertEqual(validation.returncode, 0, validation.stderr)

    def test_benchmark_accepts_standalone_cpu_evidence_for_recommended_card(self):
        with temporary_runner_card(status="recommended") as (card_root, manifest):
            evidence_path = card_root / "evidence.json"
            first_benchmark = self.run_runner(
                "benchmark",
                "--example",
                manifest["id"],
                "--device",
                "cpu",
                "--output",
                str(evidence_path),
            )
            second_benchmark = self.run_runner(
                "benchmark",
                "--example",
                manifest["id"],
                "--device",
                "cpu",
                "--output",
                str(evidence_path),
            )
            document = json.loads(evidence_path.read_text(encoding="utf-8"))
            validation = self.run_runner("validate")

        self.assertEqual(first_benchmark.returncode, 0, first_benchmark.stderr)
        self.assertEqual(second_benchmark.returncode, 0, second_benchmark.stderr)
        self.assertEqual(len(document["records"]), 2)
        self.assertTrue(all(not record["environment"]["device"]["is_cuda"] for record in document["records"]))
        self.assertNotEqual(validation.returncode, 0)
        self.assertIn("non-stale improved CUDA", validation.stderr)

    def test_case_tolerances_must_match_manifest_before_execution(self):
        with (
            temporary_runner_card(case_atol=2.0e-6) as (_, manifest),
            tempfile.TemporaryDirectory() as directory,
        ):
            marker = Path(directory) / "executed.txt"
            result = self.run_runner(
                "check",
                "--example",
                manifest["id"],
                "--device",
                "cpu",
                environment={"WARP_RUNNER_SENTINEL": str(marker)},
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("tolerances", result.stderr)
            self.assertFalse(marker.exists())

    def test_correctness_result_keys_are_checked_before_paired_timing(self):
        bad_correctness = CorrectnessResult(
            passed=True,
            outputs={
                "unexpected": OutputError(
                    name="unexpected",
                    max_abs=0.0,
                    max_rel=0.0,
                    finite=True,
                    passed=True,
                )
            },
        )
        samples = PairedSamples(
            baseline_ns=(100,) * 10,
            candidate_ns=(90,) * 10,
            order=("baseline-first", "candidate-first") * 5,
        )
        with (
            temporary_runner_card() as (_, manifest),
            tempfile.TemporaryDirectory() as directory,
            patch.object(runner_module, "check_correctness", return_value=bad_correctness),
            patch.object(runner_module, "run_paired", return_value=samples),
            self.assertRaisesRegex(ValueError, "correctness outputs"),
            redirect_stdout(io.StringIO()),
        ):
            runner_module._command_benchmark(
                SimpleNamespace(
                    example=manifest["id"],
                    device="cpu",
                    set=[],
                    warmups=None,
                    pairs=None,
                    bootstrap_seed=None,
                    resamples=None,
                    output=Path(directory) / "evidence.json",
                )
            )

    def test_existing_mixed_card_output_fails_before_execution(self):
        with (
            temporary_runner_card() as (card_root, manifest),
            tempfile.TemporaryDirectory() as directory,
        ):
            evidence_path = card_root / "evidence.json"
            initial = self.run_runner(
                "benchmark",
                "--example",
                manifest["id"],
                "--device",
                "cpu",
                "--output",
                str(evidence_path),
            )
            self.assertEqual(initial.returncode, 0, initial.stderr)
            document = json.loads(evidence_path.read_text(encoding="utf-8"))
            document["records"][0]["example_id"] = "other-synthetic-card"
            evidence_path.write_text(json.dumps(document), encoding="utf-8")

            marker = Path(directory) / "executed.txt"
            result = self.run_runner(
                "benchmark",
                "--example",
                manifest["id"],
                "--device",
                "cpu",
                "--output",
                str(evidence_path),
                environment={"WARP_RUNNER_SENTINEL": str(marker)},
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("example_id does not match", result.stderr)
            self.assertFalse(marker.exists())

    def test_validate_rejects_artifact_symlink_outside_card_root(self):
        with (
            temporary_runner_card() as (card_root, _),
            tempfile.TemporaryDirectory() as directory,
        ):
            outside_evidence = Path(directory) / "evidence.json"
            outside_evidence.write_text('{"schema_version": 1, "records": []}\n', encoding="utf-8")
            evidence_path = card_root / "evidence.json"
            evidence_path.unlink()
            try:
                evidence_path.symlink_to(outside_evidence)
            except OSError as error:
                self.skipTest(f"symbolic links unavailable: {error}")

            result = self.run_runner("validate")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside card root", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
