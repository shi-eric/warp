# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the shared protocol used by runtime-optimization example cards."""

import hashlib
import inspect
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
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import numpy as np

import warp.examples.optimizations.harness.environment as environment_module
import warp.examples.optimizations.harness.evidence as evidence_module
import warp.examples.optimizations.run as runner_module
from warp.examples.optimizations.autodiff.gradient_safe_intermediate_lifetime.benchmark import (
    build_case as build_gradient_safe_intermediate_lifetime_case,
)
from warp.examples.optimizations.harness import (
    UnsupportedWorkload,
    append_evidence,
    classify_summary,
    evidence_staleness_reasons,
    is_evidence_stale,
    validate_evidence_document,
    validate_evidence_record,
)
from warp.examples.optimizations.harness.benchmark import PairedSamples, run_paired
from warp.examples.optimizations.harness.clean_room import scan_prohibited
from warp.examples.optimizations.harness.correctness import CorrectnessResult, OutputError, check_correctness
from warp.examples.optimizations.harness.environment import capture_environment
from warp.examples.optimizations.harness.manifest import load_manifest, validate_manifest
from warp.examples.optimizations.harness.model import OptimizationCase, Tolerance, Variant
from warp.examples.optimizations.harness.registry import discover_examples
from warp.examples.optimizations.harness.statistics import PairedSummary, summarize_paired
from warp.tests.unittest_suites import default_suite
from warp.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

_INITIAL_CORPUS_IDS = {
    "direct-tape-without-checkpointing",
    "device-resident-spectral-transform",
    "device-resident-torch-exchange",
    "expanded-halo-fusion",
    "fused-elementwise-pipeline",
    "gradient-safe-intermediate-lifetime",
    "native-autodiff-rollout",
    "reused-iteration-workspace",
}
_IMPACT_LABELS = {"improved", "neutral", "harmful", "unverified"}
_STANDARD_TEST_MEMORY_LIMIT = 256 * 1024 * 1024


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
        "claims": {
            "cuda": [],
            "cpu": [],
        },
        "compatibility": {
            "warp": ">=1.17",
            "devices": ["cpu", "cuda"],
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
            "equivalence_band": {"low": 0.98, "high": 1.02},
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


def _canonical_digest(value):
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_contract_sources(card_root, manifest):
    card_root.mkdir(parents=True, exist_ok=True)
    for role in ("baseline", "candidate", "benchmark", "correctness"):
        path = card_root / manifest["artifacts"][role]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Synthetic {role} source.\n", encoding="utf-8")


def make_measured_contract(manifest, environment, correctness, protocol, card_root=None):
    artifact_hashes = {}
    for role in ("baseline", "candidate", "benchmark", "correctness"):
        relative_path = manifest["artifacts"][role]
        artifact_hashes[role] = {
            "path": relative_path,
            "sha256": "0" * 64 if card_root is None else _sha256_file(card_root / relative_path),
        }

    if card_root is None:
        shared_hashes = [{"path": "harness/evidence.py", "sha256": "1" * 64}]
    else:
        optimization_root = Path(evidence_module.__file__).resolve().parents[1]
        shared_paths = [*sorted((optimization_root / "harness").glob("*.py")), optimization_root / "run.py"]
        shared_hashes = [
            {
                "path": path.relative_to(optimization_root).as_posix(),
                "sha256": _sha256_file(path),
            }
            for path in shared_paths
        ]

    device = environment["device"]
    cpu = environment["cpu"]
    contract = {
        "example_id": manifest["id"],
        "workload": dict(environment["workload"]),
        "declared_workload": dict(manifest["benchmark"]["workload"]),
        "protocol": dict(protocol),
        "protocol_requirements": {
            name: manifest["benchmark"][name] for name in ("warmups", "pairs", "bootstrap_seed", "resamples")
        },
        "outputs": {
            name: {
                "atol": output["atol"],
                "rtol": output["rtol"],
            }
            for name, output in correctness["outputs"].items()
        },
        "compatibility": {
            "warp": {
                "measured_version": environment["warp"],
                "specifier": manifest["compatibility"]["warp"],
            },
            "devices": list(manifest["compatibility"]["devices"]),
            "equivalence_band": dict(manifest["benchmark"]["equivalence_band"]),
            "device": {
                "class": "cuda" if device["is_cuda"] else "cpu",
                "name": device["name"],
                "architecture": device["architecture"],
                "total_memory_bytes": device["total_memory_bytes"],
                "cpu_model": None if device["is_cuda"] else cpu["model"],
                "logical_cpu_count": None if device["is_cuda"] else cpu["logical_cpu_count"],
                "affinity_cpu_count": None if device["is_cuda"] else cpu["affinity_cpu_count"],
            },
        },
        "source_hashes": {
            "artifacts": artifact_hashes,
            "shared": shared_hashes,
        },
    }
    return {"digest_sha256": _canonical_digest(contract), **contract}


def make_evidence_record(
    *,
    timestamp="2026-07-29T00:00:00+00:00",
    device_alias="cpu",
    is_cuda=False,
    runtime_sources_dirty=False,
    candidate_ns=None,
    manifest=None,
    card_root=None,
    warp_version=None,
):
    if manifest is None:
        manifest = make_valid_manifest()
    if candidate_ns is None:
        candidate_ns = (70, 71, 69, 72, 68, 71, 67, 70, 69, 68)
    samples = PairedSamples(
        baseline_ns=(100, 102, 98, 101, 99, 103, 97, 100, 102, 98),
        candidate_ns=candidate_ns,
        order=("baseline-first", "candidate-first") * 5,
    )
    summary = summarize_paired(samples, bootstrap_seed=17, resamples=10_000)
    correctness = {
        "passed": True,
        "outputs": {
            "result": {
                "name": "result",
                "max_abs": 0.0,
                "max_rel": 0.0,
                "finite": True,
                "atol": 1.0e-6,
                "rtol": 1.0e-5,
                "max_normalized": 0.0,
                "passed": True,
            }
        },
    }
    if warp_version is None:
        warp_version = environment_module.wp.__version__
    environment = {
        "timestamp_utc": timestamp,
        "python": "3.12.0",
        "warp": warp_version,
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
        "cpu": {
            "model": "Synthetic CPU Model",
            "logical_cpu_count": 16,
            "affinity_cpu_count": 8,
        },
        "workload": dict(manifest["benchmark"]["workload"]),
    }
    protocol = {
        "warmups": 3,
        "pairs": summary.pairs,
        "bootstrap_seed": 17,
        "resamples": 10_000,
    }
    return {
        "record_format_version": 2,
        "record_id": uuid4().hex,
        "example_id": manifest["id"],
        "environment": environment,
        "correctness": correctness,
        "protocol": protocol,
        "samples": {
            "baseline_ns": list(samples.baseline_ns),
            "candidate_ns": list(samples.candidate_ns),
            "order": list(samples.order),
        },
        "statistics": summary.as_dict(),
        "result": classify_summary(summary),
        "limitations": list(manifest["compatibility"]["limitations"]),
        "measured_contract": make_measured_contract(
            manifest,
            environment,
            correctness,
            protocol,
            card_root,
        ),
    }


def add_manifest_claim(manifest, platform, record, impact=None, workload=None):
    if impact is None:
        impact = record["result"] if record["result"] != "inconclusive" else "unverified"
    if workload is None:
        workload = record["measured_contract"]["workload"]
    claim = {
        "impact": impact,
        "supporting_record_ids": [record["record_id"]],
        "scope": {
            "workload": dict(workload),
            "device": dict(record["measured_contract"]["compatibility"]["device"]),
        },
    }
    manifest["claims"][platform].append(claim)
    manifest["impact"][platform] = impact
    return claim


def refresh_contract_digest(record):
    contract = record["measured_contract"]
    payload = {name: value for name, value in contract.items() if name != "digest_sha256"}
    contract["digest_sha256"] = _canonical_digest(payload)


@contextmanager
def current_evidence_record(*, manifest=None, **record_arguments):
    if manifest is None:
        manifest = make_valid_manifest()
    with tempfile.TemporaryDirectory(prefix="synthetic_contract_card_") as directory:
        card_root = Path(directory)
        write_contract_sources(card_root, manifest)
        record = make_evidence_record(
            manifest=manifest,
            card_root=card_root,
            **record_arguments,
        )
        yield record, card_root


def append_evidence_in_process(path, record, start_event, result_queue):
    try:
        if not start_event.wait(timeout=30):
            raise TimeoutError("append start event timed out")
        append_evidence(Path(path), record)
    except Exception as error:
        result_queue.put(repr(error))
    else:
        result_queue.put(None)


def append_evidence_for_manifest_in_process(path, record, manifest, card_root, start_event, result_queue):
    try:
        if not start_event.wait(timeout=30):
            raise TimeoutError("append start event timed out")
        append_evidence(Path(path), record, manifest=manifest, card_root=Path(card_root))
    except Exception as error:
        result_queue.put(type(error).__name__)
    else:
        result_queue.put(None)


@contextmanager
def temporary_runner_card(
    *,
    case_atol=1.0e-6,
    case_rtol=1.0e-5,
    case_size_as_boolean=False,
    case_workload_proxy=False,
    status="unverified",
    manifest_pairs=10,
):
    with tempfile.TemporaryDirectory(prefix="synthetic_runner_registry_") as directory:
        registry_root = Path(directory)
        package_name = f"synthetic_runner_card_{uuid4().hex}"
        card_root = registry_root / package_name
        card_root.mkdir()
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
            "pairs": manifest_pairs,
            "bootstrap_seed": 17,
            "resamples": 10000,
            "equivalence_band": {"low": 0.98, "high": 1.02},
        }
        manifest["artifacts"]["python_module"] = f"{package_name}.benchmark"

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
from types import MappingProxyType

import numpy as np

from warp.examples.optimizations.harness.model import (
    JSONScalar,
    OptimizationCase,
    Tolerance,
    UnsupportedWorkload,
    Variant,
)

import_marker = os.environ.get("WARP_RUNNER_IMPORT_SENTINEL")
if import_marker:
    Path(import_marker).write_text("imported\\n", encoding="utf-8")


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

    returned_workload = dict(workload)
    if {case_size_as_boolean!r}:
        returned_workload["size"] = True
    if {case_workload_proxy!r}:
        returned_workload = MappingProxyType(returned_workload)

    return OptimizationCase(
        example_id="synthetic-runner-card",
        workload=returned_workload,
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
        sys.path.insert(0, str(registry_root))
        try:
            yield card_root, manifest, registry_root
        finally:
            sys.path.remove(str(registry_root))
            for module_name in tuple(sys.modules):
                if module_name == package_name or module_name.startswith(f"{package_name}."):
                    del sys.modules[module_name]


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

    def test_correctness_compares_large_integer_outputs_exactly(self):
        case = make_fake_case(
            baseline={"result": np.array([2**53], dtype=np.int64)},
            candidate={"result": np.array([2**53 + 1], dtype=np.int64)},
            tolerances={"result": Tolerance(atol=0.0, rtol=0.0)},
        )

        output = check_correctness(case).outputs["result"]

        self.assertFalse(output.passed)
        self.assertEqual(output.max_abs, 1.0)
        self.assertIsNone(output.max_normalized)

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

    def test_correctness_records_tolerance_and_normalized_margin(self):
        case = make_fake_case(
            baseline={"result": np.array([10.0])},
            candidate={"result": np.array([10.15])},
            tolerances={"result": Tolerance(atol=0.1, rtol=0.01)},
        )

        output = check_correctness(case).outputs["result"]

        self.assertEqual(output.atol, 0.1)
        self.assertEqual(output.rtol, 0.01)
        self.assertAlmostEqual(output.max_normalized, 0.75)
        self.assertTrue(output.passed)

    def test_nonfinite_correctness_failure_serializes_with_null_metrics(self):
        case = make_fake_case(
            baseline={"result": np.array([1.0])},
            candidate={"result": np.array([np.inf])},
        )

        result = check_correctness(case)
        serialized = json.dumps(asdict(result), allow_nan=False)

        self.assertFalse(result.passed)
        self.assertIsNone(result.outputs["result"].max_abs)
        self.assertIsNone(result.outputs["result"].max_rel)
        self.assertIsNone(result.outputs["result"].max_normalized)
        self.assertIn('"max_normalized": null', serialized)

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
                "Windows rooted artifact path",
                lambda manifest: manifest["artifacts"].__setitem__("baseline", r"\\outside.py"),
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
        self.assertIsNone(re.fullmatch(pattern, r"\outside.py"))
        self.assertIsNone(re.fullmatch(pattern, "../before.py"))
        self.assertIsNotNone(re.fullmatch(pattern, "before.py"))

    def test_recommended_claim_must_bind_the_default_workload(self):
        manifest = make_valid_manifest()
        record = make_evidence_record(
            device_alias="cuda:0",
            is_cuda=True,
            manifest=manifest,
        )
        manifest["status"] = "recommended"
        add_manifest_claim(
            manifest,
            "cuda",
            record,
            impact="improved",
            workload={"size": 2048, "iterations": 4, "seed": 17},
        )

        with self.assertRaisesRegex(ValueError, "default workload"):
            validate_manifest(manifest)

    def test_conditional_claim_requires_an_explicit_bounded_scope(self):
        manifest = make_valid_manifest()
        record = make_evidence_record(
            device_alias="cuda:0",
            is_cuda=True,
            manifest=manifest,
        )
        manifest["status"] = "conditional"
        claim = add_manifest_claim(manifest, "cuda", record, impact="improved")
        del claim["scope"]

        with self.assertRaisesRegex(ValueError, "scope"):
            validate_manifest(manifest)

    def test_unverified_manifest_cannot_publish_cuda_claims(self):
        manifest = make_valid_manifest()
        record = make_evidence_record(
            manifest=manifest,
            device_alias="cuda:0",
            is_cuda=True,
        )
        add_manifest_claim(manifest, "cuda", record, impact="improved")

        with self.assertRaisesRegex(ValueError, "unverified"):
            validate_manifest(manifest)

    def test_manifest_claim_device_must_be_declared_compatible(self):
        manifest = make_valid_manifest()
        manifest["status"] = "rejected"
        manifest["compatibility"]["devices"] = ["cuda"]
        record = make_evidence_record(manifest=manifest)
        add_manifest_claim(manifest, "cpu", record, impact="improved")

        with self.assertRaisesRegex(ValueError, "compatibility.devices"):
            validate_manifest(manifest)

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
        self.assertTrue(environment["cpu"]["model"])
        self.assertGreaterEqual(environment["cpu"]["logical_cpu_count"], 1)
        self.assertGreaterEqual(environment["cpu"]["affinity_cpu_count"], 1)
        self.assertEqual(environment["workload"], {"size": 1024, "optional_packages": []})
        self.assertEqual(
            set(environment),
            {"timestamp_utc", "python", "warp", "os", "machine", "git", "device", "cuda", "cpu", "workload"},
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

    def test_legacy_history_remains_intrinsically_valid_but_stale(self):
        evidence_path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "optimizations"
            / "kernel_fusion"
            / "fused_elementwise_pipeline"
            / "evidence.json"
        )
        document = json.loads(evidence_path.read_text(encoding="utf-8"))
        legacy_records = document["records"][:2]
        self.assertEqual(
            _canonical_digest(legacy_records),
            "2fff76a72e65707c6e72f0668d98da602b9e168427a653bea86bf95fb48cb7f1",
        )

        evolved_manifest = make_valid_manifest()
        evolved_manifest["id"] = "fused-elementwise-pipeline"
        evolved_manifest["benchmark"]["workload"]["size"] = 2048
        evolved_manifest["benchmark"]["pairs"] = 20
        for record in legacy_records:
            with self.subTest(record_id=record["record_id"]):
                validate_evidence_record(record)
                self.assertTrue(
                    is_evidence_stale(
                        record,
                        evolved_manifest,
                        card_root=evidence_path.parent,
                    )
                )

    def test_v2_evidence_stales_when_current_source_contract_or_warp_changes(self):
        now = datetime(2026, 7, 29, tzinfo=timezone.utc)
        manifest = make_valid_manifest()
        cases = ("source", "protocol contract", "output contract", "warp")
        for change in cases:
            with (
                self.subTest(change=change),
                current_evidence_record(
                    manifest=manifest,
                    timestamp=now.isoformat(),
                ) as (record, card_root),
            ):
                self.assertFalse(
                    is_evidence_stale(
                        record,
                        manifest,
                        now=now,
                        card_root=card_root,
                        current_warp_version=record["environment"]["warp"],
                    )
                )
                current_manifest = deepcopy(manifest)
                current_warp_version = record["environment"]["warp"]
                if change == "source":
                    (card_root / current_manifest["artifacts"]["baseline"]).write_text(
                        "# Changed baseline source.\n",
                        encoding="utf-8",
                    )
                elif change == "protocol contract":
                    current_manifest["benchmark"]["pairs"] += 1
                elif change == "output contract":
                    current_manifest["semantics"]["tolerance"]["absolute"] *= 2.0
                else:
                    current_warp_version = "999.0"

                self.assertTrue(
                    is_evidence_stale(
                        record,
                        current_manifest,
                        now=now,
                        card_root=card_root,
                        current_warp_version=current_warp_version,
                    )
                )

    def test_currentness_apis_require_card_root_after_hashed_source_changes(self):
        now = datetime(2026, 7, 29, tzinfo=timezone.utc)
        manifest = make_valid_manifest()
        with current_evidence_record(
            manifest=manifest,
            timestamp=now.isoformat(),
        ) as (record, card_root):
            (card_root / manifest["artifacts"]["baseline"]).write_text(
                "# Changed baseline source.\n",
                encoding="utf-8",
            )

            for currentness_operation in (evidence_staleness_reasons, is_evidence_stale):
                with (
                    self.subTest(operation=currentness_operation.__name__),
                    self.assertRaisesRegex(TypeError, "card_root"),
                ):
                    currentness_operation(
                        record,
                        manifest,
                        now=now,
                        current_warp_version=record["environment"]["warp"],
                    )

            self.assertIn(
                "current source hashes changed",
                evidence_staleness_reasons(
                    record,
                    manifest,
                    now=now,
                    card_root=card_root,
                    current_warp_version=record["environment"]["warp"],
                ),
            )

    def test_manifest_append_without_card_root_rejects_before_mutation(self):
        manifest = make_valid_manifest()
        with current_evidence_record(manifest=manifest) as (record, card_root):
            (card_root / manifest["artifacts"]["baseline"]).write_text(
                "# Changed baseline source.\n",
                encoding="utf-8",
            )
            with tempfile.TemporaryDirectory() as directory:
                evidence_path = Path(directory) / "evidence.json"

                with self.assertRaisesRegex(ValueError, "card_root"):
                    append_evidence(evidence_path, record, manifest=manifest)

                self.assertFalse(evidence_path.exists())
                self.assertFalse(evidence_path.with_name(f".{evidence_path.name}.lock").exists())

    def test_record_validation_is_intrinsic_for_prior_example_ids(self):
        evidence_path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "optimizations"
            / "kernel_fusion"
            / "fused_elementwise_pipeline"
            / "evidence.json"
        )
        legacy = json.loads(evidence_path.read_text(encoding="utf-8"))["records"][0]
        old_v2 = make_evidence_record()

        validate_evidence_record(legacy)
        validate_evidence_record(old_v2)
        self.assertNotIn("manifest", inspect.signature(validate_evidence_record).parameters)

    def test_v2_evidence_rejects_tampered_contract_digest(self):
        record = make_evidence_record()
        record["measured_contract"]["protocol"]["pairs"] += 1

        with self.assertRaisesRegex(ValueError, "digest"):
            validate_evidence_record(record)

    def test_v2_correctness_pass_is_derived_from_normalized_error(self):
        mutations = (
            (
                lambda output: output.__setitem__("passed", False),
                "passed",
            ),
            (
                lambda output: (
                    output.__setitem__("max_normalized", 1.5),
                    output.__setitem__("passed", True),
                ),
                "passed",
            ),
            (
                lambda output: output.__setitem__("atol", 2.0e-6),
                "contract",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                record = make_evidence_record()
                mutate(record["correctness"]["outputs"]["result"])
                with self.assertRaisesRegex(ValueError, message):
                    validate_evidence_record(record)

    def test_publication_rejects_contradictory_cuda_and_cpu_labels(self):
        now = datetime(2026, 7, 29, tzinfo=timezone.utc)
        for platform, record_arguments in (
            ("cuda", {"device_alias": "cuda:0", "is_cuda": True}),
            ("cpu", {}),
        ):
            with self.subTest(platform=platform):
                manifest = make_valid_manifest()
                manifest["status"] = "rejected"
                with current_evidence_record(
                    manifest=manifest,
                    timestamp=now.isoformat(),
                    **record_arguments,
                ) as (record, card_root):
                    add_manifest_claim(
                        manifest,
                        platform,
                        record,
                        impact="harmful",
                    )
                    with self.assertRaisesRegex(ValueError, "harmful"):
                        validate_evidence_document(
                            {"schema_version": 1, "records": [record]},
                            manifest,
                            now=now,
                            card_root=card_root,
                        )

    def test_cpu_claim_is_independent_of_unverified_cuda_status(self):
        now = datetime(2026, 7, 29, tzinfo=timezone.utc)
        manifest = make_valid_manifest()
        with current_evidence_record(
            manifest=manifest,
            timestamp=now.isoformat(),
        ) as (record, card_root):
            add_manifest_claim(manifest, "cpu", record, impact="improved")

            validate_evidence_document(
                {"schema_version": 1, "records": [record]},
                manifest,
                now=now,
                card_root=card_root,
            )

    def test_evidence_rejects_mismatched_sample_lengths(self):
        manifest = make_valid_manifest()
        record = make_evidence_record()
        record["samples"]["candidate_ns"].pop()

        with self.assertRaisesRegex(ValueError, "matching lengths"):
            validate_evidence_record(record)

    def test_intrinsic_validation_does_not_reinterpret_record_with_current_protocol(self):
        record = make_evidence_record()

        try:
            validate_evidence_record(record)
        except ValueError as error:
            self.fail(str(error))

    def test_evidence_accepts_overridden_workload_with_declared_keys(self):
        manifest = make_valid_manifest()
        record = make_evidence_record()
        record["environment"]["workload"] = {"size": 2048, "iterations": 7, "seed": 29}
        record["measured_contract"]["workload"] = dict(record["environment"]["workload"])
        refresh_contract_digest(record)

        try:
            validate_evidence_record(record)
        except ValueError as error:
            self.fail(str(error))

    def test_v2_contract_rejects_workload_key_or_protocol_requirement_drift(self):
        mutations = (
            (
                lambda record: (
                    record["environment"]["workload"].pop("seed"),
                    record["measured_contract"]["workload"].pop("seed"),
                ),
                "workload keys",
            ),
            (
                lambda record: record["measured_contract"]["protocol_requirements"].__setitem__(
                    "pairs",
                    11,
                ),
                "protocol.pairs",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                record = make_evidence_record()
                mutate(record)
                refresh_contract_digest(record)
                with self.assertRaisesRegex(ValueError, message):
                    validate_evidence_record(record)

    def test_workload_contract_equality_distinguishes_booleans_from_integers(self):
        manifest = make_valid_manifest()
        manifest["benchmark"]["workload"]["size"] = 1
        record = make_evidence_record(manifest=manifest)
        record["environment"]["workload"]["size"] = True

        with self.assertRaisesRegex(ValueError, "workload"):
            validate_evidence_record(record)

    def test_workload_type_evolution_marks_evidence_stale(self):
        manifest = make_valid_manifest()
        manifest["benchmark"]["workload"]["size"] = 1
        with current_evidence_record(manifest=manifest) as (record, card_root):
            evolved_manifest = deepcopy(manifest)
            evolved_manifest["benchmark"]["workload"]["size"] = True

            self.assertTrue(
                is_evidence_stale(
                    record,
                    evolved_manifest,
                    card_root=card_root,
                    current_warp_version=record["environment"]["warp"],
                )
            )

    def test_claim_scope_equality_distinguishes_booleans_from_integers(self):
        now = datetime(2026, 7, 29, tzinfo=timezone.utc)
        manifest = make_valid_manifest()
        manifest["benchmark"]["workload"]["size"] = 1
        manifest["status"] = "recommended"
        with current_evidence_record(
            manifest=manifest,
            timestamp=now.isoformat(),
            device_alias="cuda:0",
            is_cuda=True,
        ) as (record, card_root):
            claim = add_manifest_claim(manifest, "cuda", record, impact="improved")
            claim["scope"]["workload"]["size"] = True

            with self.assertRaisesRegex(ValueError, "workload"):
                validate_evidence_document(
                    {"schema_version": 1, "records": [record]},
                    manifest,
                    now=now,
                    card_root=card_root,
                )

    def test_evidence_rejects_workload_key_drift(self):
        manifest = make_valid_manifest()
        for workload in (
            {"size": 2048, "iterations": 7},
            {"size": 2048, "iterations": 7, "seed": 29, "extra": True},
        ):
            with self.subTest(workload=workload):
                record = make_evidence_record()
                record["environment"]["workload"] = workload
                with self.assertRaisesRegex(ValueError, "measured contract"):
                    validate_evidence_record(record)

    def test_recommended_manifest_requires_passing_cuda_evidence(self):
        manifest = make_valid_manifest()
        manifest["status"] = "recommended"
        now = datetime(2026, 7, 29, tzinfo=timezone.utc)
        with current_evidence_record(
            manifest=manifest,
            timestamp=now.isoformat(),
            device_alias="cuda:0",
            is_cuda=True,
        ) as (record, card_root):
            output = record["correctness"]["outputs"]["result"]
            output["max_normalized"] = 1.5
            output["passed"] = False
            record["correctness"]["passed"] = False
            add_manifest_claim(manifest, "cuda", record, impact="improved")
            with self.assertRaisesRegex(ValueError, "passing correctness"):
                validate_evidence_document(
                    {"schema_version": 1, "records": [record]},
                    manifest,
                    now=now,
                    card_root=card_root,
                )

    def test_record_validation_does_not_apply_document_claim_gate(self):
        manifest = make_valid_manifest()
        manifest["status"] = "recommended"
        record = make_evidence_record(device_alias="cuda:0", is_cuda=True)
        add_manifest_claim(manifest, "cuda", record, impact="improved")

        validate_evidence_record(record)

    def test_temp_history_validation_accepts_non_supporting_records(self):
        records = (
            make_evidence_record(),
            make_evidence_record(
                device_alias="cuda:0",
                is_cuda=True,
                candidate_ns=(90, 92, 94, 96, 98, 102, 104, 106, 108, 110),
            ),
        )

        for record in records:
            with self.subTest(result=record["result"]):
                manifest = make_valid_manifest()
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
        with current_evidence_record(
            manifest=manifest,
            device_alias="cuda:0",
            is_cuda=True,
            candidate_ns=(90, 92, 94, 96, 98, 102, 104, 106, 108, 110),
        ) as (record, card_root):
            self.assertEqual(record["result"], "inconclusive")

            with tempfile.TemporaryDirectory() as directory:
                evidence_path = Path(directory) / "evidence.json"
                evidence_path.write_text('{"schema_version": 1, "records": []}\n', encoding="utf-8")

                try:
                    append_evidence(
                        evidence_path,
                        record,
                        manifest=manifest,
                        card_root=card_root,
                    )
                except ValueError as error:
                    self.fail(str(error))

                document = json.loads(evidence_path.read_text(encoding="utf-8"))
                self.assertEqual(document["records"], [record])
                validate_evidence_document(
                    document,
                    manifest,
                    require_claim_support=False,
                )

    def test_manifest_append_rejects_invalid_history_atomically(self):
        manifest = make_valid_manifest()
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
            card_root = Path(directory) / "card"
            write_contract_sources(card_root, manifest)
            supporting = make_evidence_record(
                manifest=manifest,
                card_root=card_root,
                device_alias="cuda:0",
                is_cuda=True,
            )
            evidence_path = Path(directory) / "evidence.json"
            for index, history in enumerate(histories):
                with self.subTest(index=index):
                    evidence_path.write_text(history, encoding="utf-8")
                    before = evidence_path.read_bytes()

                    with self.assertRaises(ValueError):
                        append_evidence(
                            evidence_path,
                            supporting,
                            manifest=manifest,
                            card_root=card_root,
                        )

                    self.assertEqual(evidence_path.read_bytes(), before)

    def test_document_claim_accepts_history_with_one_supporting_record(self):
        manifest = make_valid_manifest()
        manifest["status"] = "recommended"
        manifest["compatibility"]["evidence_max_age_days"] = 30
        now = datetime(2026, 7, 29, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory(prefix="synthetic_contract_card_") as directory:
            card_root = Path(directory)
            write_contract_sources(card_root, manifest)
            records = [
                make_evidence_record(
                    manifest=manifest,
                    card_root=card_root,
                    timestamp=now.isoformat(),
                ),
                make_evidence_record(
                    manifest=manifest,
                    card_root=card_root,
                    timestamp="2025-01-01T00:00:00+00:00",
                    device_alias="cuda:0",
                    is_cuda=True,
                ),
                make_evidence_record(
                    manifest=manifest,
                    card_root=card_root,
                    timestamp=now.isoformat(),
                    device_alias="cuda:0",
                    is_cuda=True,
                    candidate_ns=(90, 92, 94, 96, 98, 102, 104, 106, 108, 110),
                ),
                make_evidence_record(
                    manifest=manifest,
                    card_root=card_root,
                    timestamp=now.isoformat(),
                    device_alias="cuda:0",
                    is_cuda=True,
                ),
            ]
            add_manifest_claim(manifest, "cuda", records[-1], impact="improved")

            validate_evidence_document(
                {"schema_version": 1, "records": records},
                manifest,
                now=now,
                card_root=card_root,
            )

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

    def test_evolved_example_id_retains_legacy_and_v2_history_as_stale(self):
        evidence_path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "optimizations"
            / "kernel_fusion"
            / "fused_elementwise_pipeline"
            / "evidence.json"
        )
        legacy = json.loads(evidence_path.read_text(encoding="utf-8"))["records"][0]
        old_manifest = make_valid_manifest()
        evolved_manifest = deepcopy(old_manifest)
        evolved_manifest["id"] = "renamed-synthetic-card"

        with tempfile.TemporaryDirectory() as directory:
            card_root = Path(directory) / "card"
            write_contract_sources(card_root, old_manifest)
            old_v2 = make_evidence_record(
                manifest=old_manifest,
                card_root=card_root,
            )
            new_v2 = make_evidence_record(
                manifest=evolved_manifest,
                card_root=card_root,
            )
            retained = {"schema_version": 1, "records": [legacy, old_v2]}
            validate_evidence_document(
                retained,
                evolved_manifest,
                require_claim_support=False,
            )
            self.assertTrue(
                is_evidence_stale(
                    legacy,
                    evolved_manifest,
                    card_root=card_root,
                )
            )
            self.assertIn(
                "example ID differs from the current manifest",
                evidence_staleness_reasons(
                    old_v2,
                    evolved_manifest,
                    card_root=card_root,
                    current_warp_version=old_v2["environment"]["warp"],
                ),
            )

            output_path = Path(directory) / "evidence.json"
            output_path.write_text(json.dumps(retained), encoding="utf-8")
            before = output_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "example_id"):
                append_evidence(
                    output_path,
                    old_v2,
                    manifest=evolved_manifest,
                    card_root=card_root,
                )
            self.assertEqual(output_path.read_bytes(), before)

            append_evidence(
                output_path,
                new_v2,
                manifest=evolved_manifest,
                card_root=card_root,
            )
            updated = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(updated["records"], [legacy, old_v2, new_v2])

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

        with tempfile.TemporaryDirectory() as directory:
            card_root = Path(directory) / "card"
            write_contract_sources(card_root, manifest)
            supporting = make_evidence_record(
                manifest=manifest,
                card_root=card_root,
                device_alias="cuda:0",
                is_cuda=True,
            )
            exploratory = make_evidence_record(
                manifest=manifest,
                card_root=card_root,
                candidate_ns=(90, 92, 94, 96, 98, 102, 104, 106, 108, 110),
            )
            evidence_path = Path(directory) / "evidence.json"
            evidence_path.write_text(
                json.dumps({"schema_version": 1, "records": [supporting]}),
                encoding="utf-8",
            )

            try:
                append_evidence(
                    evidence_path,
                    exploratory,
                    manifest=manifest,
                    card_root=card_root,
                )
            except TypeError as error:
                self.fail(str(error))

            document = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {record["record_id"] for record in document["records"]},
                {supporting["record_id"], exploratory["record_id"]},
            )
            validate_evidence_document(
                document,
                manifest,
                require_claim_support=False,
            )

    def test_manifest_append_rejects_mixed_card_history_under_writer_lock(self):
        manifest = make_valid_manifest()

        with tempfile.TemporaryDirectory() as directory:
            card_root = Path(directory) / "card"
            write_contract_sources(card_root, manifest)
            valid = make_evidence_record(
                manifest=manifest,
                card_root=card_root,
            )
            wrong_card = make_evidence_record(
                manifest=manifest,
                card_root=card_root,
            )
            wrong_card["example_id"] = "other-synthetic-card"
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
                        manifest,
                        str(card_root),
                        start_event,
                        result_queue,
                    ),
                )
                for record in (valid, wrong_card)
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
            self.assertEqual(retained["record_id"], valid["record_id"])
            validate_evidence_document(
                document,
                manifest,
                require_claim_support=False,
            )

    def test_evidence_recomputes_and_rejects_tampered_statistics(self):
        manifest = make_valid_manifest()
        record = make_evidence_record()
        record["statistics"]["median_ratio"] = 0.5

        with self.assertRaisesRegex(ValueError, "statistics.median_ratio"):
            validate_evidence_record(record)

    def test_evidence_rejects_tampered_result_classification(self):
        record = make_evidence_record()
        record["result"] = "harmful"

        with self.assertRaisesRegex(ValueError, "result"):
            validate_evidence_record(record)

    def test_evidence_rejects_negative_correctness_errors(self):
        record = make_evidence_record()
        record["correctness"]["outputs"]["result"]["max_abs"] = -0.1

        with self.assertRaisesRegex(ValueError, "max_abs"):
            validate_evidence_record(record)

    def test_evidence_rejects_passed_nonfinite_output(self):
        record = make_evidence_record()
        record["correctness"]["outputs"]["result"]["finite"] = False

        with self.assertRaisesRegex(ValueError, "finite"):
            validate_evidence_record(record)

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
                    validate_evidence_record(record)

    def test_evidence_rejects_invalid_device_architecture_type(self):
        for architecture in (True, ["sm_120"]):
            with self.subTest(architecture=architecture):
                record = make_evidence_record()
                record["environment"]["device"]["architecture"] = architecture
                with self.assertRaisesRegex(ValueError, "architecture"):
                    validate_evidence_record(record)

    def test_evidence_rejects_non_scalar_workload_values(self):
        record = make_evidence_record()
        record["environment"]["workload"]["nested"] = {"value": 1}

        with self.assertRaisesRegex(ValueError, "JSON scalar"):
            validate_evidence_record(record)

    def test_evidence_rejects_non_alternating_pair_order(self):
        record = make_evidence_record()
        record["samples"]["order"][1] = "baseline-first"

        with self.assertRaisesRegex(ValueError, "alternate"):
            validate_evidence_record(record)

    def test_evidence_becomes_stale_after_manifest_maximum_age(self):
        manifest = make_valid_manifest()
        manifest["compatibility"]["evidence_max_age_days"] = 30
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with current_evidence_record(
            manifest=manifest,
            timestamp=timestamp.isoformat(),
        ) as (record, card_root):
            self.assertFalse(
                is_evidence_stale(
                    record,
                    manifest,
                    now=timestamp + timedelta(days=30),
                    card_root=card_root,
                    current_warp_version=record["environment"]["warp"],
                )
            )
            self.assertTrue(
                is_evidence_stale(
                    record,
                    manifest,
                    now=timestamp + timedelta(days=30, seconds=1),
                    card_root=card_root,
                    current_warp_version=record["environment"]["warp"],
                )
            )

    def test_evidence_requires_canonical_utc_timestamp(self):
        for timestamp in ("2026-07-29T00:00:00Z", "2026-07-29T01:00:00+01:00"):
            with self.subTest(timestamp=timestamp):
                record = make_evidence_record()
                record["environment"]["timestamp_utc"] = timestamp
                with self.assertRaisesRegex(ValueError, "canonical UTC"):
                    validate_evidence_record(record)

    def test_evidence_rejects_implausibly_future_timestamp_with_explicit_skew(self):
        now = datetime(2026, 7, 29, tzinfo=timezone.utc)
        record = make_evidence_record(timestamp=(now + timedelta(minutes=6)).isoformat())

        with self.assertRaisesRegex(ValueError, "future"):
            validate_evidence_record(record, now=now)
        validate_evidence_record(
            record,
            now=now,
            future_skew=timedelta(minutes=10),
        )

    def test_append_retains_insertion_order_for_immutable_history(self):
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
                [later["record_id"], earlier["record_id"], same_time_first_id["record_id"]],
            )

    def test_recommended_manifest_rejects_stale_improved_cuda_evidence(self):
        manifest = make_valid_manifest()
        manifest["status"] = "recommended"
        manifest["compatibility"]["evidence_max_age_days"] = 30
        now = datetime(2026, 7, 29, tzinfo=timezone.utc)
        with current_evidence_record(
            manifest=manifest,
            timestamp="2025-01-01T00:00:00+00:00",
            device_alias="cuda:0",
            is_cuda=True,
        ) as (record, card_root):
            add_manifest_claim(manifest, "cuda", record, impact="improved")
            with self.assertRaisesRegex(ValueError, "stale"):
                validate_evidence_document(
                    {"schema_version": 1, "records": [record]},
                    manifest,
                    now=now,
                    card_root=card_root,
                )

    def test_recommended_manifest_requires_clean_runtime_sources(self):
        manifest = make_valid_manifest()
        manifest["status"] = "conditional"
        now = datetime(2026, 7, 29, tzinfo=timezone.utc)
        with current_evidence_record(
            manifest=manifest,
            timestamp=now.isoformat(),
            device_alias="cuda:0",
            is_cuda=True,
            runtime_sources_dirty=True,
        ) as (record, card_root):
            add_manifest_claim(manifest, "cuda", record, impact="improved")
            with self.assertRaisesRegex(ValueError, "clean runtime sources"):
                validate_evidence_document(
                    {"schema_version": 1, "records": [record]},
                    manifest,
                    now=now,
                    card_root=card_root,
                )


class TestOptimizationRunner(unittest.TestCase):
    def run_runner(self, *arguments, environment=None, registry_root=None):
        process_environment = os.environ.copy()
        if environment is not None:
            process_environment.update(environment)
        command = [sys.executable, "-m", "warp.examples.optimizations.run"]
        if registry_root is not None:
            command.extend(("--registry-root", str(registry_root)))
            python_path = process_environment.get("PYTHONPATH")
            process_environment["PYTHONPATH"] = (
                str(registry_root) if not python_path else f"{registry_root}{os.pathsep}{python_path}"
            )
        command.extend(arguments)
        return subprocess.run(
            command,
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

    def test_selected_manifest_protocol_under_ride_fails_before_card_import(self):
        with (
            temporary_runner_card(manifest_pairs=20) as (_, manifest, registry_root),
            tempfile.TemporaryDirectory() as directory,
        ):
            marker = Path(directory) / "imported.txt"
            result = self.run_runner(
                "benchmark",
                "--example",
                manifest["id"],
                "--device",
                "cpu",
                "--pairs",
                "10",
                "--output",
                str(Path(directory) / "evidence.json"),
                environment={"WARP_RUNNER_IMPORT_SENTINEL": str(marker)},
                registry_root=registry_root,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("manifest requires at least 20", result.stderr)
        self.assertFalse(marker.exists())

    def test_invalid_output_parent_fails_before_card_import(self):
        with (
            temporary_runner_card() as (_, manifest, registry_root),
            tempfile.TemporaryDirectory() as directory,
        ):
            marker = Path(directory) / "imported.txt"
            output_path = Path(directory) / "missing-parent" / "evidence.json"
            result = self.run_runner(
                "benchmark",
                "--example",
                manifest["id"],
                "--device",
                "cpu",
                "--output",
                str(output_path),
                environment={"WARP_RUNNER_IMPORT_SENTINEL": str(marker)},
                registry_root=registry_root,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("output parent", result.stderr)
        self.assertFalse(marker.exists())

    def test_python_module_must_resolve_to_hashed_benchmark_artifact(self):
        with (
            temporary_runner_card() as (card_root, manifest, registry_root),
            tempfile.TemporaryDirectory() as directory,
        ):
            marker = Path(directory) / "imported.txt"
            external_package = registry_root / f"external_benchmark_{uuid4().hex}"
            external_package.mkdir()
            (external_package / "__init__.py").write_text(
                """
import os
from pathlib import Path

marker = os.environ.get("WARP_RUNNER_IMPORT_SENTINEL")
if marker:
    Path(marker).write_text("imported\\n", encoding="utf-8")
""".lstrip(),
                encoding="utf-8",
            )
            (external_package / "benchmark.py").write_text(
                "# This external target must never be imported.\n",
                encoding="utf-8",
            )
            manifest["artifacts"]["python_module"] = f"{external_package.name}.benchmark"
            (card_root / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            for command in (
                ("check", "--example", manifest["id"], "--device", "cpu"),
                ("validate",),
            ):
                with self.subTest(command=command):
                    result = self.run_runner(
                        *command,
                        environment={"WARP_RUNNER_IMPORT_SENTINEL": str(marker)},
                        registry_root=registry_root,
                    )

                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("hashed benchmark artifact", result.stderr)
                    self.assertFalse(marker.exists())

    def test_build_case_workload_equality_distinguishes_booleans_from_integers(self):
        with temporary_runner_card(case_size_as_boolean=True) as (_, manifest, registry_root):
            result = self.run_runner(
                "check",
                "--example",
                manifest["id"],
                "--device",
                "cpu",
                registry_root=registry_root,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("changed the requested workload", result.stderr)

    def test_build_case_accepts_a_generic_workload_mapping(self):
        with temporary_runner_card(case_workload_proxy=True) as (_, manifest, registry_root):
            result = self.run_runner(
                "check",
                "--example",
                manifest["id"],
                "--device",
                "cpu",
                registry_root=registry_root,
            )

        self.assertEqual(result.returncode, 0, result.stderr)

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
        with temporary_runner_card() as (_, manifest, registry_root):
            result = self.run_runner(
                "check",
                "--example",
                manifest["id"],
                "--device",
                "cpu",
                registry_root=registry_root,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Correctness: PASS", result.stdout)
        self.assertIn("result", result.stdout)

    def test_temporary_card_validation_succeeds(self):
        with temporary_runner_card() as (_, manifest, registry_root):
            result = self.run_runner("validate", registry_root=registry_root)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"{manifest['id']}: valid", result.stdout)

    def test_injected_registry_does_not_leak_fixture_into_production_discovery(self):
        with temporary_runner_card() as (_, manifest, registry_root):
            fixture_list = self.run_runner("list", registry_root=registry_root)
            production_list = self.run_runner("list")

        self.assertEqual(fixture_list.returncode, 0, fixture_list.stderr)
        self.assertIn(manifest["id"], fixture_list.stdout)
        self.assertEqual(production_list.returncode, 0, production_list.stderr)
        self.assertNotIn(manifest["id"], production_list.stdout)
        self.assertIn("fused-elementwise-pipeline", production_list.stdout)

    def test_benchmark_override_produces_valid_card_evidence(self):
        with temporary_runner_card() as (card_root, manifest, registry_root):
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
                registry_root=registry_root,
            )
            document = json.loads(evidence_path.read_text(encoding="utf-8"))
            validation = self.run_runner("validate", registry_root=registry_root)

        self.assertEqual(benchmark.returncode, 0, benchmark.stderr)
        self.assertEqual(document["records"][0]["environment"]["workload"], {"size": 8, "scale": 1.0})
        self.assertEqual(validation.returncode, 0, validation.stderr)

    def test_benchmark_output_accepts_standalone_exploratory_cpu_evidence(self):
        with temporary_runner_card() as (card_root, manifest, registry_root):
            evidence_path = card_root / "evidence.json"
            first_benchmark = self.run_runner(
                "benchmark",
                "--example",
                manifest["id"],
                "--device",
                "cpu",
                "--output",
                str(evidence_path),
                registry_root=registry_root,
            )
            second_benchmark = self.run_runner(
                "benchmark",
                "--example",
                manifest["id"],
                "--device",
                "cpu",
                "--output",
                str(evidence_path),
                registry_root=registry_root,
            )
            document = json.loads(evidence_path.read_text(encoding="utf-8"))
            validation = self.run_runner("validate", registry_root=registry_root)

        self.assertEqual(first_benchmark.returncode, 0, first_benchmark.stderr)
        self.assertEqual(second_benchmark.returncode, 0, second_benchmark.stderr)
        self.assertEqual(len(document["records"]), 2)
        self.assertTrue(all(not record["environment"]["device"]["is_cuda"] for record in document["records"]))
        self.assertEqual(validation.returncode, 0, validation.stderr)

    def test_case_tolerances_must_match_manifest_before_execution(self):
        with (
            temporary_runner_card(case_atol=2.0e-6) as (_, manifest, registry_root),
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
                registry_root=registry_root,
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
                    atol=1.0e-6,
                    rtol=1.0e-5,
                    max_normalized=0.0,
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
            temporary_runner_card() as (_, manifest, registry_root),
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
                    registry_root=registry_root,
                )
            )

    def test_existing_prior_id_history_is_retained_when_appending(self):
        with (
            temporary_runner_card() as (card_root, manifest, registry_root),
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
                registry_root=registry_root,
            )
            self.assertEqual(initial.returncode, 0, initial.stderr)
            document = json.loads(evidence_path.read_text(encoding="utf-8"))
            retained = document["records"][0]
            retained["example_id"] = "other-synthetic-card"
            retained["measured_contract"]["example_id"] = "other-synthetic-card"
            refresh_contract_digest(retained)
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
                registry_root=registry_root,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(marker.exists())
            updated = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(len(updated["records"]), 2)
            self.assertEqual(updated["records"][0], retained)
            self.assertEqual(updated["records"][1]["example_id"], manifest["id"])
            self.assertTrue(is_evidence_stale(retained, manifest, card_root=card_root))
            validate_evidence_document(updated, manifest, card_root=card_root)

    def test_validate_rejects_artifact_symlink_outside_card_root(self):
        with (
            temporary_runner_card() as (card_root, _, registry_root),
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

            result = self.run_runner("validate", registry_root=registry_root)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside card root", result.stderr)


class TestRuntimeOptimizationExamples(unittest.TestCase):
    def test_initial_corpus_satisfies_publication_invariants(self):
        examples = discover_examples()

        self.assertEqual(set(examples), _INITIAL_CORPUS_IDS)
        for example_id, example in examples.items():
            with self.subTest(example_id=example_id):
                manifest = example.manifest
                evidence_path = example.root / manifest["artifacts"]["evidence"]
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

                validate_manifest(manifest, example.root / "manifest.json")
                validate_evidence_document(evidence, manifest, card_root=example.root)
                for name, relative_path in manifest["artifacts"].items():
                    if name != "python_module":
                        self.assertTrue((example.root / relative_path).is_file(), name)

                self.assertIn(manifest["impact"]["cuda"], _IMPACT_LABELS)
                self.assertIn(manifest["impact"]["cpu"], _IMPACT_LABELS)
                self.assertIs(manifest["clean_room"]["synthetic"], True)
                self.assertIs(manifest["clean_room"]["derived_from_private_source"], False)
                self.assertLess(
                    manifest["benchmark"]["estimated_peak_bytes"],
                    _STANDARD_TEST_MEMORY_LIMIT,
                )

                records = evidence["records"]
                for measured in records:
                    self.assertGreaterEqual(measured["protocol"]["pairs"], 10)
                    self.assertGreaterEqual(measured["protocol"]["resamples"], 10_000)

                if manifest["status"] in {"recommended", "conditional"}:
                    current_improved_cuda = [
                        measured
                        for measured in records
                        if measured["environment"]["device"]["is_cuda"]
                        and measured["correctness"]["passed"]
                        and measured["result"] == "improved"
                        and not is_evidence_stale(measured, manifest, card_root=example.root)
                    ]
                    self.assertTrue(current_improved_cuda)
                elif manifest["status"] == "rejected":
                    self.assertTrue(manifest["impact"]["mechanism"])
                    rejected_claims = manifest["claims"]["cuda"]
                    self.assertTrue(rejected_claims)
                    records_by_id = {measured["record_id"]: measured for measured in records}
                    for claim in rejected_claims:
                        self.assertEqual(claim["impact"], manifest["impact"]["cuda"])
                        self.assertIn(claim["impact"], {"neutral", "harmful"})
                        for record_id in claim["supporting_record_ids"]:
                            measured = records_by_id[record_id]
                            self.assertTrue(measured["environment"]["device"]["is_cuda"])
                            self.assertTrue(measured["correctness"]["passed"])
                            self.assertIs(measured["environment"]["git"]["runtime_sources_dirty"], False)
                            self.assertEqual(
                                measured["measured_contract"]["workload"],
                                claim["scope"]["workload"],
                            )
                            self.assertEqual(
                                measured["measured_contract"]["compatibility"]["device"],
                                claim["scope"]["device"],
                            )
                            self.assertFalse(
                                is_evidence_stale(measured, manifest, card_root=example.root),
                            )
                            if claim["impact"] == "harmful":
                                self.assertEqual(measured["result"], "harmful")
                            else:
                                band = measured["measured_contract"]["compatibility"]["equivalence_band"]
                                self.assertGreaterEqual(measured["statistics"]["ratio_ci_low"], band["low"])
                                self.assertLessEqual(measured["statistics"]["ratio_ci_high"], band["high"])

    def test_fused_elementwise_pipeline_card_is_registered(self):
        examples = discover_examples()

        self.assertEqual(set(examples), _INITIAL_CORPUS_IDS)
        record = examples["fused-elementwise-pipeline"]
        validate_manifest(record.manifest, record.root / "manifest.json")
        for name, relative_path in record.manifest["artifacts"].items():
            if name != "python_module":
                self.assertTrue((record.root / relative_path).is_file(), name)

    def test_device_resident_torch_exchange_card_is_registered(self):
        examples = discover_examples()

        record = examples["device-resident-torch-exchange"]
        validate_manifest(record.manifest, record.root / "manifest.json")
        self.assertEqual(record.manifest["compatibility"]["devices"], ["cuda"])
        self.assertEqual(record.manifest["semantics"]["observable_outputs"], ["values"])
        self.assertEqual(
            record.manifest["benchmark"]["workload"],
            {
                "iterations": 20,
                "seed": 20260730,
                "size": 1_048_576,
            },
        )
        for name, relative_path in record.manifest["artifacts"].items():
            if name != "python_module":
                self.assertTrue((record.root / relative_path).is_file(), name)

    def test_device_resident_torch_exchange_rejects_invalid_workloads(self):
        from warp.examples.optimizations.interoperability.device_resident_torch_exchange.benchmark import (  # noqa: PLC0415
            build_case,
        )

        valid = {
            "iterations": 2,
            "seed": 20260730,
            "size": 4096,
        }
        invalid = (
            ({name: value for name, value in valid.items() if name != "size"}, "exactly"),
            ({**valid, "extra": 1}, "exactly"),
            ({**valid, "size": True}, "size"),
            ({**valid, "size": 0}, "size"),
            ({**valid, "iterations": 0}, "iterations"),
            ({**valid, "seed": True}, "seed"),
            ({**valid, "seed": -1}, "seed"),
        )
        for workload, message in invalid:
            with (
                self.subTest(workload=workload),
                self.assertRaisesRegex(UnsupportedWorkload, message),
            ):
                build_case("cuda:0", workload)

        with self.assertRaisesRegex(UnsupportedWorkload, "CUDA"):
            build_case("cpu", valid)

    def test_device_resident_torch_exchange_rejection_does_not_import_torch(self):
        script = """
import builtins

from warp.examples.optimizations.harness import UnsupportedWorkload

original_import = builtins.__import__


def reject_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("PyTorch was imported before workload rejection")
    return original_import(name, *args, **kwargs)


builtins.__import__ = reject_torch
from warp.examples.optimizations.interoperability.device_resident_torch_exchange.benchmark import build_case

valid = {"iterations": 2, "seed": 20260730, "size": 4096}
for device, workload in (
    ("cuda:0", {"iterations": 2, "seed": 20260730}),
    ("cpu", valid),
):
    try:
        build_case(device, workload)
    except UnsupportedWorkload:
        pass
    else:
        raise AssertionError("workload should have been rejected")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")

    def test_expanded_halo_fusion_card_is_registered(self):
        examples = discover_examples()

        record = examples["expanded-halo-fusion"]
        validate_manifest(record.manifest, record.root / "manifest.json")
        self.assertIn(
            "fusion_expands_halo",
            record.manifest["recognition"]["signals"],
        )
        for name, relative_path in record.manifest["artifacts"].items():
            if name != "python_module":
                self.assertTrue((record.root / relative_path).is_file(), name)

    def test_expanded_halo_fusion_rejects_retuned_followup_workloads(self):
        from warp.examples.optimizations.kernel_fusion.expanded_halo_fusion.benchmark import (  # noqa: PLC0415
            build_case,
        )

        followup = {
            "height": 2048,
            "iterations": 20,
            "radius": 4,
            "seed": 20260730,
            "width": 2048,
        }
        for name, value in (
            ("height", 1024),
            ("iterations", 10),
            ("seed", 17),
            ("width", 1024),
        ):
            workload = {**followup, name: value}
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(UnsupportedWorkload, "predeclared"),
            ):
                build_case("cuda:0", workload)

    def test_expanded_halo_fusion_classification_matches_evidence(self):
        record = discover_examples()["expanded-halo-fusion"]
        evidence = json.loads((record.root / record.manifest["artifacts"]["evidence"]).read_text(encoding="utf-8"))
        measured = evidence["records"][-1]

        self.assertGreater(measured["statistics"]["ratio_ci_low"], 1.0)
        self.assertEqual(measured["result"], "harmful")
        self.assertEqual(record.manifest["status"], "rejected")
        self.assertEqual(record.manifest["impact"]["cuda"], "harmful")
        self.assertEqual(record.manifest["impact"]["cpu"], "unverified")
        self.assertEqual(
            record.manifest["claims"]["cuda"],
            [
                {
                    "impact": "harmful",
                    "scope": {
                        "device": measured["measured_contract"]["compatibility"]["device"],
                        "workload": measured["measured_contract"]["workload"],
                    },
                    "supporting_record_ids": [measured["record_id"]],
                }
            ],
        )

    def test_device_resident_spectral_transform_card_is_registered(self):
        examples = discover_examples()

        record = examples["device-resident-spectral-transform"]
        validate_manifest(record.manifest, record.root / "manifest.json")
        self.assertEqual(
            set(record.manifest["recognition"]["signals"]),
            {
                "device_to_host_copy_inside_iteration",
                "host_to_device_copy_inside_iteration",
                "host_transform_between_device_kernels",
            },
        )
        for name, relative_path in record.manifest["artifacts"].items():
            if name != "python_module":
                self.assertTrue((record.root / relative_path).is_file(), name)

    def test_reused_iteration_workspace_card_is_registered(self):
        examples = discover_examples()

        record = examples["reused-iteration-workspace"]
        validate_manifest(record.manifest, record.root / "manifest.json")
        self.assertIn(
            "scratch_shape_dtype_and_device_are_fixed",
            record.manifest["applicability"]["preconditions"],
        )
        self.assertIn(
            "calls_using_the_same_workspace_do_not_overlap",
            record.manifest["applicability"]["preconditions"],
        )
        for name, relative_path in record.manifest["artifacts"].items():
            if name != "python_module":
                self.assertTrue((record.root / relative_path).is_file(), name)

    def test_native_autodiff_rollout_card_is_registered(self):
        examples = discover_examples()

        record = examples["native-autodiff-rollout"]
        validate_manifest(record.manifest, record.root / "manifest.json")
        self.assertEqual(
            set(record.manifest["semantics"]["observable_outputs"]),
            {"final_state", "input_gradient"},
        )
        for name, relative_path in record.manifest["artifacts"].items():
            if name != "python_module":
                self.assertTrue((record.root / relative_path).is_file(), name)

    def test_gradient_safe_intermediate_lifetime_card_is_registered(self):
        examples = discover_examples()

        record = examples["gradient-safe-intermediate-lifetime"]
        validate_manifest(record.manifest, record.root / "manifest.json")
        self.assertEqual(
            set(record.manifest["semantics"]["observable_outputs"]),
            {"final_state", "input_gradient"},
        )
        self.assertIn(
            "step_derivative_is_independent_of_overwritten_primal_values",
            record.manifest["applicability"]["preconditions"],
        )
        for name, relative_path in record.manifest["artifacts"].items():
            if name != "python_module":
                self.assertTrue((record.root / relative_path).is_file(), name)

    def test_gradient_safe_intermediate_lifetime_rejects_state_dependent_derivative(self):
        with self.assertRaisesRegex(UnsupportedWorkload, "derivative.*state"):
            build_gradient_safe_intermediate_lifetime_case(
                "cpu",
                {
                    "derivative_depends_on_state": True,
                    "seed": 20260730,
                    "size": 16,
                    "steps": 2,
                },
            )

    def test_direct_tape_without_checkpointing_card_is_registered(self):
        examples = discover_examples()

        record = examples["direct-tape-without-checkpointing"]
        validate_manifest(record.manifest, record.root / "manifest.json")
        self.assertEqual(
            set(record.manifest["semantics"]["observable_outputs"]),
            {"final_state", "input_gradient"},
        )
        self.assertEqual(record.manifest["benchmark"]["workload"]["segment_length"], 8)
        self.assertIn(
            "full_tape_exceeds_memory_budget",
            record.manifest["applicability"]["contraindications"],
        )
        for name, relative_path in record.manifest["artifacts"].items():
            if name != "python_module":
                self.assertTrue((record.root / relative_path).is_file(), name)

    def test_direct_tape_without_checkpointing_capacity_guard_counts_gradients_before_allocation(self):
        from warp.examples.optimizations.memory_tradeoffs.direct_tape_without_checkpointing.benchmark import (  # noqa: PLC0415
            build_case,
        )

        constrained_device = SimpleNamespace(is_cuda=True, free_memory=4000)
        workload = {
            "seed": 20260730,
            "segment_length": 2,
            "size": 16,
            "steps": 8,
        }
        with (
            patch(
                "warp.examples.optimizations.memory_tradeoffs."
                "direct_tape_without_checkpointing.benchmark.wp.get_device",
                return_value=constrained_device,
            ),
            self.assertRaisesRegex(UnsupportedWorkload, "25%.*free memory"),
        ):
            build_case("cuda:0", workload)

    def test_direct_tape_without_checkpointing_rejects_partial_segments(self):
        from warp.examples.optimizations.memory_tradeoffs.direct_tape_without_checkpointing.benchmark import (  # noqa: PLC0415
            build_case,
        )

        roomy_device = SimpleNamespace(is_cuda=True, free_memory=1 << 30)
        workload = {
            "seed": 20260730,
            "segment_length": 3,
            "size": 16,
            "steps": 8,
        }
        with (
            patch(
                "warp.examples.optimizations.memory_tradeoffs."
                "direct_tape_without_checkpointing.benchmark.wp.get_device",
                return_value=roomy_device,
            ),
            self.assertRaisesRegex(UnsupportedWorkload, "segment_length.*divide steps"),
        ):
            build_case("cuda:0", workload)

    def test_native_autodiff_rollout_torch_runtime_errors_are_not_skipped(self):
        original_import = __import__

        def fail_torch_import(name, *args, **kwargs):
            if name == "torch":
                raise RuntimeError("broken CUDA runtime")
            return original_import(name, *args, **kwargs)

        fake_test = SimpleNamespace(
            skipTest=lambda reason: (_ for _ in ()).throw(AssertionError(f"unexpected skip: {reason}"))
        )
        with (
            patch("builtins.__import__", side_effect=fail_torch_import),
            self.assertRaisesRegex(RuntimeError, "broken CUDA runtime"),
        ):
            test_native_autodiff_rollout_correctness(fake_test, "cuda:0", torch_required=True)

    def test_default_suite_registers_optimization_evidence(self):
        loader = unittest.TestLoader()
        loader.testNamePatterns = ["*TestOptimizationEvidence*"]
        suite = default_suite(loader)

        def iter_tests(test_suite):
            for test in test_suite:
                if isinstance(test, unittest.TestSuite):
                    yield from iter_tests(test)
                else:
                    yield test

        test_ids = [test.id() for test in iter_tests(suite)]

        self.assertTrue(any(".TestOptimizationEvidence." in test_id for test_id in test_ids))


def test_fused_elementwise_pipeline_correctness(test, device):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "warp.examples.optimizations.kernel_fusion.fused_elementwise_pipeline.test_correctness",
            "--device",
            str(device),
            "--size",
            "1",
            "--size",
            "4096",
            "--size",
            "4097",
            "--iterations",
            "2",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    test.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
    for size in (1, 4096, 4097):
        test.assertIn(f"size={size}: PASS", result.stdout)


def test_device_resident_spectral_transform_correctness(test, device):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "warp.examples.optimizations.transfer_elimination.device_resident_spectral_transform.test_correctness",
            "--device",
            str(device),
            "--size",
            "256",
            "--batch",
            "4",
            "--iterations",
            "1",
            "--iterations",
            "2",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    test.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
    for iterations in (1, 2):
        test.assertIn(
            f"size=256 batch=4 iterations={iterations}: PASS",
            result.stdout,
        )


def test_reused_iteration_workspace_correctness(test, device):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "warp.examples.optimizations.allocation_reuse.reused_iteration_workspace.test_correctness",
            "--device",
            str(device),
            "--size",
            "4096",
            "--iterations",
            "3",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    test.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
    test.assertIn("size=4096 iterations=3: PASS", result.stdout)


def _require_torch_cuda(test):
    try:
        import torch  # noqa: PLC0415
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        test.skipTest(f"{error}")
    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA support is required")


def test_device_resident_torch_exchange_correctness(test, device, torch_required):
    if torch_required:
        _require_torch_cuda(test)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "warp.examples.optimizations.interoperability.device_resident_torch_exchange.test_correctness",
            "--device",
            str(device),
            "--size",
            "4096",
            "--iterations",
            "2",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    test.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
    test.assertIn("size=4096 iterations=2: PASS", result.stdout)
    test.assertIn("storage_alias=PASS", result.stdout)


def test_device_resident_torch_exchange_non_default_stream(test, device, torch_required):
    if torch_required:
        _require_torch_cuda(test)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "warp.examples.optimizations.interoperability.device_resident_torch_exchange.test_correctness",
            "--device",
            str(device),
            "--size",
            "4096",
            "--iterations",
            "2",
            "--verify-ordering",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    test.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
    test.assertIn("stream=non-default", result.stdout)
    test.assertIn("producer_warp_consumer=PASS", result.stdout)


def test_native_autodiff_rollout_correctness(test, device, torch_required):
    if torch_required:
        _require_torch_cuda(test)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "warp.examples.optimizations.autodiff.native_autodiff_rollout.test_correctness",
            "--device",
            str(device),
            "--size",
            "1024",
            "--steps",
            "4",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    test.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
    test.assertIn("size=1024 steps=4: PASS", result.stdout)


def test_native_autodiff_rollout_non_default_stream(test, device, torch_required):
    if torch_required:
        try:
            import torch  # noqa: PLC0415
        except ModuleNotFoundError as error:
            if error.name != "torch":
                raise
            test.skipTest(f"{error}")
        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA support is required")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "warp.examples.optimizations.autodiff.native_autodiff_rollout.test_correctness",
            "--device",
            str(device),
            "--size",
            "1024",
            "--steps",
            "4",
            "--non-default-stream",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    test.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
    test.assertIn("size=1024 steps=4: PASS", result.stdout)
    test.assertIn("stream=non-default", result.stdout)


def test_gradient_safe_intermediate_lifetime_correctness(test, device):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "warp.examples.optimizations.autodiff.gradient_safe_intermediate_lifetime.test_correctness",
            "--device",
            str(device),
            "--size",
            "1024",
            "--steps",
            "4",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    test.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
    test.assertIn("size=1024 steps=4: PASS", result.stdout)
    test.assertIn("nonlinear_counterexample: PASS", result.stdout)


def test_direct_tape_without_checkpointing_correctness(test, device):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "warp.examples.optimizations.memory_tradeoffs.direct_tape_without_checkpointing.test_correctness",
            "--device",
            str(device),
            "--size",
            "1024",
            "--steps",
            "8",
            "--segment-length",
            "2",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    test.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
    test.assertIn("size=1024 steps=8 segment_length=2: PASS", result.stdout)


def test_expanded_halo_fusion_correctness(test, device):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "warp.examples.optimizations.kernel_fusion.expanded_halo_fusion.test_correctness",
            "--device",
            str(device),
            "--shape",
            "1",
            "1",
            "--shape",
            "7",
            "9",
            "--iterations",
            "2",
            "--radius",
            "2",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    test.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
    for height, width in ((1, 1), (7, 9)):
        test.assertIn(
            f"shape={height}x{width} iterations=2 radius=2: PASS",
            result.stdout,
        )


add_function_test(
    TestRuntimeOptimizationExamples,
    "test_fused_elementwise_pipeline_correctness",
    test_fused_elementwise_pipeline_correctness,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestRuntimeOptimizationExamples,
    "test_device_resident_spectral_transform_correctness",
    test_device_resident_spectral_transform_correctness,
    devices=get_cuda_test_devices(mode="basic"),
)
add_function_test(
    TestRuntimeOptimizationExamples,
    "test_reused_iteration_workspace_correctness",
    test_reused_iteration_workspace_correctness,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestRuntimeOptimizationExamples,
    "test_device_resident_torch_exchange_correctness",
    test_device_resident_torch_exchange_correctness,
    devices=get_cuda_test_devices(mode="basic"),
    torch_required=True,
)
add_function_test(
    TestRuntimeOptimizationExamples,
    "test_device_resident_torch_exchange_non_default_stream",
    test_device_resident_torch_exchange_non_default_stream,
    devices=get_cuda_test_devices(mode="basic"),
    torch_required=True,
)
add_function_test(
    TestRuntimeOptimizationExamples,
    "test_native_autodiff_rollout_correctness",
    test_native_autodiff_rollout_correctness,
    devices=get_cuda_test_devices(mode="basic"),
    torch_required=True,
)
add_function_test(
    TestRuntimeOptimizationExamples,
    "test_native_autodiff_rollout_non_default_stream",
    test_native_autodiff_rollout_non_default_stream,
    devices=get_cuda_test_devices(mode="basic"),
    torch_required=True,
)
add_function_test(
    TestRuntimeOptimizationExamples,
    "test_gradient_safe_intermediate_lifetime_correctness",
    test_gradient_safe_intermediate_lifetime_correctness,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestRuntimeOptimizationExamples,
    "test_direct_tape_without_checkpointing_correctness",
    test_direct_tape_without_checkpointing_correctness,
    devices=get_cuda_test_devices(mode="basic"),
)
add_function_test(
    TestRuntimeOptimizationExamples,
    "test_expanded_halo_fusion_correctness",
    test_expanded_halo_fusion_correctness,
    devices=get_cuda_test_devices(mode="basic"),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
