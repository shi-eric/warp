# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build, validate, and append auditable optimization evidence."""

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from uuid import uuid4

import warp as wp
from warp.examples.optimizations.harness.benchmark import PairedSamples
from warp.examples.optimizations.harness.correctness import CorrectnessResult
from warp.examples.optimizations.harness.json_utils import json_values_equal, load_json
from warp.examples.optimizations.harness.manifest import validate_manifest
from warp.examples.optimizations.harness.statistics import PairedSummary, summarize_paired

_LEGACY_RECORD_KEYS = {
    "record_id",
    "example_id",
    "environment",
    "correctness",
    "protocol",
    "samples",
    "statistics",
    "result",
    "limitations",
}
_V2_RECORD_KEYS = _LEGACY_RECORD_KEYS | {"record_format_version", "measured_contract"}
_STATISTIC_KEYS = {
    "baseline_median_ns",
    "candidate_median_ns",
    "baseline_mad_ns",
    "candidate_mad_ns",
    "median_ratio",
    "ratio_ci_low",
    "ratio_ci_high",
    "pairs",
}
_PROTOCOL_KEYS = {"warmups", "pairs", "bootstrap_seed", "resamples"}
_ARTIFACT_SOURCE_ROLES = ("baseline", "candidate", "benchmark", "correctness")
_APPEND_LOCK_TIMEOUT_SECONDS = 30.0
_DEFAULT_FUTURE_SKEW = timedelta(minutes=5)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_VERSION_CLAUSE_PATTERN = re.compile(r"^(>=|<=|==|>|<)\s*(\d+(?:\.\d+)*)$")


def classify_summary(summary: PairedSummary) -> str:
    """Classify a paired interval relative to runtime parity."""

    if summary.ratio_ci_high < 1.0:
        return "improved"
    if summary.ratio_ci_low > 1.0:
        return "harmful"
    return "inconclusive"


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_source(card_root: Path, relative_path: str) -> Path:
    root = card_root.resolve()
    source_path = (root / relative_path).resolve(strict=True)
    try:
        source_path.relative_to(root)
    except ValueError:
        raise ValueError(f"source path resolves outside card root: {relative_path}") from None
    if not source_path.is_file():
        raise ValueError(f"source path is not a file: {relative_path}")
    return source_path


def _shared_source_paths(optimization_root: Path) -> list[Path]:
    paths = sorted((optimization_root / "harness").glob("*.py"))
    paths.append(optimization_root / "run.py")
    return paths


def _source_hash_snapshot(
    manifest: Mapping[str, Any],
    card_root: Path,
    optimization_root: Path,
) -> dict[str, Any]:
    artifact_hashes = {}
    for role in _ARTIFACT_SOURCE_ROLES:
        relative_path = manifest["artifacts"][role]
        artifact_hashes[role] = {
            "path": relative_path,
            "sha256": _sha256_file(_resolve_source(card_root, relative_path)),
        }

    optimization_root = optimization_root.resolve()
    shared_hashes = []
    for source_path in _shared_source_paths(optimization_root):
        resolved_source_path = source_path.resolve(strict=True)
        shared_hashes.append(
            {
                "path": resolved_source_path.relative_to(optimization_root).as_posix(),
                "sha256": _sha256_file(resolved_source_path),
            }
        )
    return {
        "artifacts": artifact_hashes,
        "shared": shared_hashes,
    }


def _correctness_mapping(correctness: CorrectnessResult | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(correctness, CorrectnessResult):
        return asdict(correctness)
    return correctness


def _device_contract(environment: Mapping[str, Any]) -> dict[str, Any]:
    device = environment["device"]
    cpu = environment["cpu"]
    is_cuda = device["is_cuda"]
    return {
        "class": "cuda" if is_cuda else "cpu",
        "name": device["name"],
        "architecture": device["architecture"],
        "total_memory_bytes": device["total_memory_bytes"],
        "cpu_model": None if is_cuda else cpu["model"],
        "logical_cpu_count": None if is_cuda else cpu["logical_cpu_count"],
        "affinity_cpu_count": None if is_cuda else cpu["affinity_cpu_count"],
    }


def build_measured_contract(
    *,
    manifest: Mapping[str, Any],
    card_root: Path,
    environment: Mapping[str, Any],
    correctness: CorrectnessResult | Mapping[str, Any],
    warmups: int,
    pairs: int,
    bootstrap_seed: int,
    resamples: int,
    optimization_root: Path | None = None,
) -> dict[str, Any]:
    """Build the immutable workload, compatibility, and source contract for a run."""

    validate_manifest(manifest)
    correctness_data = _correctness_mapping(correctness)
    if set(environment["workload"]) != set(manifest["benchmark"]["workload"]):
        raise ValueError("measured workload keys do not match the declared workload")
    output_names = set(correctness_data["outputs"])
    declared_outputs = set(manifest["semantics"]["observable_outputs"])
    if output_names != declared_outputs:
        raise ValueError("correctness outputs do not match manifest semantics")
    tolerance = manifest["semantics"]["tolerance"]
    outputs = {}
    for name, output in correctness_data["outputs"].items():
        if output["atol"] != tolerance["absolute"] or output["rtol"] != tolerance["relative"]:
            raise ValueError("correctness tolerances do not match manifest semantics")
        outputs[name] = {
            "atol": output["atol"],
            "rtol": output["rtol"],
        }

    if optimization_root is None:
        optimization_root = Path(__file__).resolve().parents[1]
    source_hashes = _source_hash_snapshot(manifest, card_root, optimization_root)

    protocol = {
        "warmups": warmups,
        "pairs": pairs,
        "bootstrap_seed": bootstrap_seed,
        "resamples": resamples,
    }
    contract = {
        "example_id": manifest["id"],
        "workload": dict(environment["workload"]),
        "declared_workload": dict(manifest["benchmark"]["workload"]),
        "protocol": protocol,
        "protocol_requirements": {name: manifest["benchmark"][name] for name in _PROTOCOL_KEYS},
        "outputs": outputs,
        "compatibility": {
            "warp": {
                "measured_version": environment["warp"],
                "specifier": manifest["compatibility"]["warp"],
            },
            "devices": list(manifest["compatibility"]["devices"]),
            "equivalence_band": dict(manifest["benchmark"]["equivalence_band"]),
            "device": _device_contract(environment),
        },
        "source_hashes": source_hashes,
    }
    return {"digest_sha256": _canonical_digest(contract), **contract}


def build_evidence_record(
    *,
    example_id: str,
    environment: Mapping[str, Any],
    correctness: CorrectnessResult,
    samples: PairedSamples,
    summary: PairedSummary,
    warmups: int,
    bootstrap_seed: int,
    resamples: int,
    limitations: Sequence[str],
    measured_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize one correctness and paired-benchmark run as version 2 evidence."""

    record = {
        "record_format_version": 2,
        "record_id": uuid4().hex,
        "example_id": example_id,
        "environment": dict(environment),
        "correctness": asdict(correctness),
        "protocol": {
            "warmups": warmups,
            "pairs": summary.pairs,
            "bootstrap_seed": bootstrap_seed,
            "resamples": resamples,
        },
        "samples": {
            "baseline_ns": list(samples.baseline_ns),
            "candidate_ns": list(samples.candidate_ns),
            "order": list(samples.order),
        },
        "statistics": summary.as_dict(),
        "result": classify_summary(summary),
        "limitations": list(limitations),
        "measured_contract": dict(measured_contract),
    }
    _validate_record_integrity(record)
    return record


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _require_exact_keys(data: Mapping[str, Any], keys: set[str], field: str) -> None:
    if set(data) != keys:
        raise ValueError(f"{field} must contain exactly: {', '.join(sorted(keys))}")


def _require_integer(value: Any, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _require_json_number(value: Any, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return float(value)


def _require_nullable_json_number(
    value: Any,
    field: str,
    minimum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _require_json_number(value, field, minimum)


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_sha256(value: Any, field: str) -> str:
    value = _require_string(value, field)
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("environment.timestamp_utc must be an ISO 8601 string")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("environment.timestamp_utc must be an ISO 8601 timestamp") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("environment.timestamp_utc must be canonical UTC")
    canonical = timestamp.astimezone(timezone.utc).isoformat()
    if timestamp.utcoffset() != timedelta(0) or value != canonical or not value.endswith("+00:00"):
        raise ValueError("environment.timestamp_utc must be canonical UTC")
    return timestamp


def _resolve_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must include a UTC offset")
    return now.astimezone(timezone.utc)


def _validate_future_timestamp(
    record: Mapping[str, Any],
    now: datetime,
    future_skew: timedelta,
) -> None:
    if not isinstance(future_skew, timedelta) or future_skew < timedelta(0):
        raise ValueError("future_skew must be a non-negative timedelta")
    timestamp = _parse_timestamp(record["environment"]["timestamp_utc"])
    if timestamp > now + future_skew:
        raise ValueError("environment.timestamp_utc is implausibly far in the future")


def _validate_json_scalar_mapping(value: Any, field: str) -> Mapping[str, Any]:
    mapping = _require_mapping(value, field)
    for name, item in mapping.items():
        _require_string(name, f"{field} key")
        if item is None or isinstance(item, (str, bool)):
            continue
        if isinstance(item, int) and not isinstance(item, bool):
            continue
        if isinstance(item, float) and math.isfinite(item):
            continue
        raise ValueError(f"{field}.{name} must be a JSON scalar")
    return mapping


def _validate_protocol(value: Any, field: str = "protocol") -> Mapping[str, int]:
    protocol = _require_mapping(value, field)
    _require_exact_keys(protocol, _PROTOCOL_KEYS, field)
    _require_integer(protocol["warmups"], f"{field}.warmups", minimum=3)
    _require_integer(protocol["pairs"], f"{field}.pairs", minimum=10)
    _require_integer(protocol["bootstrap_seed"], f"{field}.bootstrap_seed", minimum=0)
    _require_integer(protocol["resamples"], f"{field}.resamples", minimum=10_000)
    return protocol


def _samples_from_record(record: Mapping[str, Any]) -> PairedSamples:
    samples = _require_mapping(record["samples"], "samples")
    _require_exact_keys(samples, {"baseline_ns", "candidate_ns", "order"}, "samples")
    baseline = samples["baseline_ns"]
    candidate = samples["candidate_ns"]
    order = samples["order"]
    if not isinstance(baseline, list) or not isinstance(candidate, list) or not isinstance(order, list):
        raise ValueError("baseline, candidate, and order samples must be arrays")
    if len(baseline) != len(candidate) or len(baseline) != len(order):
        raise ValueError("baseline, candidate, and order samples must have matching lengths")
    for field, values in (("samples.baseline_ns", baseline), ("samples.candidate_ns", candidate)):
        for value in values:
            _require_integer(value, field, minimum=1)
    if any(not isinstance(value, str) for value in order):
        raise ValueError("samples.order entries must be strings")
    expected_order = tuple(
        "baseline-first" if pair_index % 2 == 0 else "candidate-first" for pair_index in range(len(order))
    )
    if tuple(order) != expected_order:
        raise ValueError("samples.order must alternate baseline-first and candidate-first")
    return PairedSamples(tuple(baseline), tuple(candidate), tuple(order))


def _validate_legacy_correctness(value: Any) -> Mapping[str, Any]:
    correctness = _require_mapping(value, "correctness")
    _require_exact_keys(correctness, {"passed", "outputs"}, "correctness")
    if not isinstance(correctness["passed"], bool):
        raise ValueError("correctness.passed must be a boolean")
    outputs = _require_mapping(correctness["outputs"], "correctness.outputs")
    if not outputs:
        raise ValueError("correctness.outputs must not be empty")
    output_keys = {"name", "max_abs", "max_rel", "finite", "passed"}
    for name, value in outputs.items():
        output = _require_mapping(value, f"correctness.outputs.{name}")
        _require_exact_keys(output, output_keys, f"correctness.outputs.{name}")
        if output["name"] != name:
            raise ValueError(f"correctness.outputs.{name}.name must match its output key")
        for metric in ("max_abs", "max_rel"):
            _require_json_number(output[metric], f"correctness.outputs.{name}.{metric}", minimum=0.0)
        if not isinstance(output["finite"], bool) or not isinstance(output["passed"], bool):
            raise ValueError(f"correctness.outputs.{name} flags must be booleans")
        if output["passed"] and not output["finite"]:
            raise ValueError(f"correctness.outputs.{name}.passed requires finite output")
    if correctness["passed"] != all(output["passed"] for output in outputs.values()):
        raise ValueError("correctness.passed must agree with its output results")
    return correctness


def _validate_v2_correctness(value: Any) -> Mapping[str, Any]:
    correctness = _require_mapping(value, "correctness")
    _require_exact_keys(correctness, {"passed", "outputs"}, "correctness")
    if not isinstance(correctness["passed"], bool):
        raise ValueError("correctness.passed must be a boolean")
    outputs = _require_mapping(correctness["outputs"], "correctness.outputs")
    if not outputs:
        raise ValueError("correctness.outputs must not be empty")
    output_keys = {
        "name",
        "max_abs",
        "max_rel",
        "finite",
        "atol",
        "rtol",
        "max_normalized",
        "passed",
    }
    for name, value in outputs.items():
        output = _require_mapping(value, f"correctness.outputs.{name}")
        _require_exact_keys(output, output_keys, f"correctness.outputs.{name}")
        if output["name"] != name:
            raise ValueError(f"correctness.outputs.{name}.name must match its output key")
        max_abs = _require_nullable_json_number(
            output["max_abs"],
            f"correctness.outputs.{name}.max_abs",
            minimum=0.0,
        )
        max_rel = _require_nullable_json_number(
            output["max_rel"],
            f"correctness.outputs.{name}.max_rel",
            minimum=0.0,
        )
        max_normalized = _require_nullable_json_number(
            output["max_normalized"],
            f"correctness.outputs.{name}.max_normalized",
            minimum=0.0,
        )
        _require_json_number(output["atol"], f"correctness.outputs.{name}.atol", minimum=0.0)
        _require_json_number(output["rtol"], f"correctness.outputs.{name}.rtol", minimum=0.0)
        if not isinstance(output["finite"], bool) or not isinstance(output["passed"], bool):
            raise ValueError(f"correctness.outputs.{name} flags must be booleans")
        if not output["finite"] and any(metric is not None for metric in (max_abs, max_rel, max_normalized)):
            raise ValueError(f"correctness.outputs.{name} non-finite metrics must be null")
        derived_pass = output["finite"] and max_normalized is not None and max_normalized <= 1.0
        if output["passed"] != derived_pass:
            raise ValueError(f"correctness.outputs.{name}.passed must be derived from finite output and max_normalized")
    if correctness["passed"] != all(output["passed"] for output in outputs.values()):
        raise ValueError("correctness.passed must agree with its derived output results")
    return correctness


def _validate_environment(value: Any, *, legacy: bool) -> Mapping[str, Any]:
    environment = _require_mapping(value, "environment")
    keys = {
        "timestamp_utc",
        "python",
        "warp",
        "os",
        "machine",
        "git",
        "device",
        "cuda",
        "workload",
    }
    if not legacy:
        keys.add("cpu")
    _require_exact_keys(environment, keys, "environment")
    _parse_timestamp(environment["timestamp_utc"])
    for field in ("python", "warp", "os", "machine"):
        _require_string(environment[field], f"environment.{field}")

    git = _require_mapping(environment["git"], "environment.git")
    _require_exact_keys(git, {"revision", "repository_dirty", "runtime_sources_dirty"}, "environment.git")
    if git["revision"] is not None and not isinstance(git["revision"], str):
        raise ValueError("environment.git.revision must be a string or null")
    for field in ("repository_dirty", "runtime_sources_dirty"):
        if git[field] is not None and not isinstance(git[field], bool):
            raise ValueError(f"environment.git.{field} must be a boolean or null")

    device = _require_mapping(environment["device"], "environment.device")
    _require_exact_keys(
        device,
        {"alias", "name", "is_cuda", "architecture", "total_memory_bytes"},
        "environment.device",
    )
    if not isinstance(device["alias"], str) or not isinstance(device["name"], str):
        raise ValueError("environment.device alias and name must be strings")
    if not isinstance(device["is_cuda"], bool):
        raise ValueError("environment.device.is_cuda must be a boolean")
    architecture = device["architecture"]
    if isinstance(architecture, bool) or (architecture is not None and not isinstance(architecture, (int, str))):
        raise ValueError("environment.device.architecture must be an integer, string, or null")
    _require_integer(device["total_memory_bytes"], "environment.device.total_memory_bytes", minimum=0)

    cuda = _require_mapping(environment["cuda"], "environment.cuda")
    _require_exact_keys(cuda, {"toolkit", "driver"}, "environment.cuda")
    for field in ("toolkit", "driver"):
        version = cuda[field]
        if not isinstance(version, (list, tuple)) or len(version) != 2:
            raise ValueError(f"environment.cuda.{field} must contain two version integers")
        for component in version:
            _require_integer(component, f"environment.cuda.{field}", minimum=0)

    if not legacy:
        cpu = _require_mapping(environment["cpu"], "environment.cpu")
        _require_exact_keys(
            cpu,
            {"model", "logical_cpu_count", "affinity_cpu_count"},
            "environment.cpu",
        )
        _require_string(cpu["model"], "environment.cpu.model")
        _require_integer(cpu["logical_cpu_count"], "environment.cpu.logical_cpu_count", minimum=1)
        _require_integer(cpu["affinity_cpu_count"], "environment.cpu.affinity_cpu_count", minimum=1)

    _validate_json_scalar_mapping(environment["workload"], "environment.workload")
    return environment


def _validate_source_entry(value: Any, field: str) -> None:
    source = _require_mapping(value, field)
    _require_exact_keys(source, {"path", "sha256"}, field)
    path = _require_string(source["path"], f"{field}.path")
    posix_path = PurePosixPath(path)
    windows_path = PureWindowsPath(path)
    if (
        path.startswith(("/", "\\"))
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    ):
        raise ValueError(f"{field}.path must be a safe relative path")
    _require_sha256(source["sha256"], f"{field}.sha256")


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", value)
    if match is None:
        raise ValueError(f"unsupported Warp version: {value}")
    return tuple(int(component) for component in match.group(1).split("."))


def _padded_version_pair(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)), right + (0,) * (width - len(right))


def _version_satisfies(version: str, specifier: str) -> bool:
    measured = _version_tuple(version)
    for raw_clause in specifier.split(","):
        clause = raw_clause.strip()
        match = _VERSION_CLAUSE_PATTERN.fullmatch(clause)
        if match is None:
            raise ValueError(f"unsupported Warp compatibility specifier: {specifier}")
        operator, required_text = match.groups()
        left, right = _padded_version_pair(measured, _version_tuple(required_text))
        comparisons = {
            ">=": left >= right,
            "<=": left <= right,
            "==": left == right,
            ">": left > right,
            "<": left < right,
        }
        if not comparisons[operator]:
            return False
    return True


def _validate_contract_device(value: Any) -> Mapping[str, Any]:
    device = _require_mapping(value, "measured_contract.compatibility.device")
    keys = {
        "class",
        "name",
        "architecture",
        "total_memory_bytes",
        "cpu_model",
        "logical_cpu_count",
        "affinity_cpu_count",
    }
    _require_exact_keys(device, keys, "measured_contract.compatibility.device")
    device_class = device["class"]
    if device_class not in {"cuda", "cpu"}:
        raise ValueError("measured_contract.compatibility.device.class must be cuda or cpu")
    _require_string(device["name"], "measured_contract.compatibility.device.name")
    architecture = device["architecture"]
    if isinstance(architecture, bool) or (architecture is not None and not isinstance(architecture, (int, str))):
        raise ValueError("measured_contract.compatibility.device.architecture is invalid")
    _require_integer(
        device["total_memory_bytes"],
        "measured_contract.compatibility.device.total_memory_bytes",
        minimum=0,
    )
    if device_class == "cuda":
        for field in ("cpu_model", "logical_cpu_count", "affinity_cpu_count"):
            if device[field] is not None:
                raise ValueError(f"measured_contract.compatibility.device.{field} must be null for CUDA")
    else:
        _require_string(device["cpu_model"], "measured_contract.compatibility.device.cpu_model")
        _require_integer(
            device["logical_cpu_count"],
            "measured_contract.compatibility.device.logical_cpu_count",
            minimum=1,
        )
        _require_integer(
            device["affinity_cpu_count"],
            "measured_contract.compatibility.device.affinity_cpu_count",
            minimum=1,
        )
    return device


def _validate_measured_contract(
    value: Any,
    record: Mapping[str, Any],
    correctness: Mapping[str, Any],
) -> Mapping[str, Any]:
    contract = _require_mapping(value, "measured_contract")
    keys = {
        "digest_sha256",
        "example_id",
        "workload",
        "declared_workload",
        "protocol",
        "protocol_requirements",
        "outputs",
        "compatibility",
        "source_hashes",
    }
    _require_exact_keys(contract, keys, "measured_contract")
    stored_digest = _require_sha256(contract["digest_sha256"], "measured_contract.digest_sha256")
    payload = {name: item for name, item in contract.items() if name != "digest_sha256"}
    if stored_digest != _canonical_digest(payload):
        raise ValueError("measured_contract digest does not match its canonical snapshot")

    if contract["example_id"] != record["example_id"]:
        raise ValueError("example_id does not match the measured contract")
    workload = _validate_json_scalar_mapping(contract["workload"], "measured_contract.workload")
    declared_workload = _validate_json_scalar_mapping(
        contract["declared_workload"],
        "measured_contract.declared_workload",
    )
    if set(workload) != set(declared_workload):
        raise ValueError("measured_contract workload keys must match the declared workload")
    protocol = _validate_protocol(contract["protocol"], "measured_contract.protocol")
    protocol_requirements = _validate_protocol(
        contract["protocol_requirements"],
        "measured_contract.protocol_requirements",
    )
    for name in _PROTOCOL_KEYS:
        if protocol[name] < protocol_requirements[name]:
            raise ValueError(f"measured_contract.protocol.{name} is below its stored requirement")
    outputs = _require_mapping(contract["outputs"], "measured_contract.outputs")
    if not outputs:
        raise ValueError("measured_contract.outputs must not be empty")
    for name, value in outputs.items():
        output = _require_mapping(value, f"measured_contract.outputs.{name}")
        _require_exact_keys(output, {"atol", "rtol"}, f"measured_contract.outputs.{name}")
        _require_json_number(output["atol"], f"measured_contract.outputs.{name}.atol", minimum=0.0)
        _require_json_number(output["rtol"], f"measured_contract.outputs.{name}.rtol", minimum=0.0)

    compatibility = _require_mapping(contract["compatibility"], "measured_contract.compatibility")
    _require_exact_keys(
        compatibility,
        {"warp", "devices", "equivalence_band", "device"},
        "measured_contract.compatibility",
    )
    warp_contract = _require_mapping(compatibility["warp"], "measured_contract.compatibility.warp")
    _require_exact_keys(
        warp_contract,
        {"measured_version", "specifier"},
        "measured_contract.compatibility.warp",
    )
    measured_version = _require_string(
        warp_contract["measured_version"],
        "measured_contract.compatibility.warp.measured_version",
    )
    specifier = _require_string(
        warp_contract["specifier"],
        "measured_contract.compatibility.warp.specifier",
    )
    if not _version_satisfies(measured_version, specifier):
        raise ValueError("measured Warp version is outside the stored compatibility specifier")
    devices = compatibility["devices"]
    if not isinstance(devices, list) or not devices:
        raise ValueError("measured_contract.compatibility.devices must be a non-empty array")
    for device_class in devices:
        if device_class not in {"cpu", "cuda"}:
            raise ValueError("measured_contract.compatibility.devices must contain only cpu or cuda")
    if len(devices) != len(set(devices)):
        raise ValueError("measured_contract.compatibility.devices must not contain duplicates")
    band = _require_mapping(
        compatibility["equivalence_band"],
        "measured_contract.compatibility.equivalence_band",
    )
    _require_exact_keys(
        band,
        {"low", "high"},
        "measured_contract.compatibility.equivalence_band",
    )
    low = _require_json_number(
        band["low"],
        "measured_contract.compatibility.equivalence_band.low",
        minimum=0.0,
    )
    high = _require_json_number(
        band["high"],
        "measured_contract.compatibility.equivalence_band.high",
        minimum=0.0,
    )
    if not low < high or not low <= 1.0 <= high:
        raise ValueError("measured_contract compatibility equivalence band must contain parity")
    device = _validate_contract_device(compatibility["device"])
    if device["class"] not in devices:
        raise ValueError("measured device class is outside the stored compatibility contract")

    source_hashes = _require_mapping(contract["source_hashes"], "measured_contract.source_hashes")
    _require_exact_keys(
        source_hashes,
        {"artifacts", "shared"},
        "measured_contract.source_hashes",
    )
    artifacts = _require_mapping(
        source_hashes["artifacts"],
        "measured_contract.source_hashes.artifacts",
    )
    _require_exact_keys(
        artifacts,
        set(_ARTIFACT_SOURCE_ROLES),
        "measured_contract.source_hashes.artifacts",
    )
    for role, source in artifacts.items():
        _validate_source_entry(source, f"measured_contract.source_hashes.artifacts.{role}")
    shared = source_hashes["shared"]
    if not isinstance(shared, list) or not shared:
        raise ValueError("measured_contract.source_hashes.shared must be a non-empty array")
    shared_paths = set()
    for index, source in enumerate(shared):
        _validate_source_entry(source, f"measured_contract.source_hashes.shared[{index}]")
        source_path = source["path"]
        if source_path in shared_paths:
            raise ValueError("measured_contract.source_hashes.shared contains duplicate paths")
        shared_paths.add(source_path)

    if not json_values_equal(record["environment"]["workload"], workload):
        raise ValueError("environment workload does not match the measured contract")
    if record["protocol"] != protocol:
        raise ValueError("protocol does not match the measured contract")
    if record["environment"]["warp"] != measured_version:
        raise ValueError("environment Warp version does not match the measured contract")
    if _device_contract(record["environment"]) != device:
        raise ValueError("environment device does not match the measured contract")
    if set(correctness["outputs"]) != set(outputs):
        raise ValueError("correctness outputs do not match the measured contract")
    for name, output in correctness["outputs"].items():
        if output["atol"] != outputs[name]["atol"] or output["rtol"] != outputs[name]["rtol"]:
            raise ValueError(f"correctness output {name!r} tolerances do not match the measured contract")
    return contract


def _validate_statistics_and_result(
    record: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[PairedSamples, PairedSummary]:
    samples = _samples_from_record(record)
    if protocol["pairs"] != len(samples.baseline_ns):
        raise ValueError("protocol.pairs must match the raw sample lengths")
    statistics = _require_mapping(record["statistics"], "statistics")
    _require_exact_keys(statistics, _STATISTIC_KEYS, "statistics")
    recomputed = summarize_paired(
        samples,
        bootstrap_seed=protocol["bootstrap_seed"],
        resamples=protocol["resamples"],
    )
    for field in _STATISTIC_KEYS - {"pairs"}:
        stored = statistics[field]
        if isinstance(stored, bool) or not isinstance(stored, (int, float)) or not math.isfinite(stored):
            raise ValueError(f"statistics.{field} must be a finite number")
        if not math.isclose(
            float(stored),
            getattr(recomputed, field),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"statistics.{field} does not match the raw samples")
    if statistics["pairs"] != recomputed.pairs:
        raise ValueError("statistics.pairs does not match the raw samples")
    if record["result"] != classify_summary(recomputed):
        raise ValueError("result does not match the recomputed statistics")
    limitations = record["limitations"]
    if not isinstance(limitations, list) or any(not isinstance(item, str) or not item for item in limitations):
        raise ValueError("limitations must be an array of non-empty strings")
    return samples, recomputed


def _validate_record_integrity(record: Mapping[str, Any]) -> tuple[PairedSamples, PairedSummary]:
    record = _require_mapping(record, "record")
    is_v2 = "record_format_version" in record
    _require_exact_keys(record, _V2_RECORD_KEYS if is_v2 else _LEGACY_RECORD_KEYS, "record")
    if is_v2 and record["record_format_version"] != 2:
        raise ValueError("record_format_version must be 2")
    _require_string(record["record_id"], "record_id")
    _require_string(record["example_id"], "example_id")
    _validate_environment(record["environment"], legacy=not is_v2)
    correctness = (
        _validate_v2_correctness(record["correctness"])
        if is_v2
        else _validate_legacy_correctness(record["correctness"])
    )
    protocol = _validate_protocol(record["protocol"])
    samples, summary = _validate_statistics_and_result(record, protocol)
    if is_v2:
        _validate_measured_contract(record["measured_contract"], record, correctness)
    return samples, summary


def validate_evidence_record(
    record: Mapping[str, Any],
    *,
    now: datetime | None = None,
    future_skew: timedelta = _DEFAULT_FUTURE_SKEW,
) -> None:
    """Validate one record intrinsically against its stored measured contract."""

    _validate_record_integrity(record)
    _validate_future_timestamp(record, _resolve_now(now), future_skew)


def _current_output_contract(manifest: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    tolerance = manifest["semantics"]["tolerance"]
    return {
        name: {
            "atol": tolerance["absolute"],
            "rtol": tolerance["relative"],
        }
        for name in manifest["semantics"]["observable_outputs"]
    }


def evidence_staleness_reasons(
    record: Mapping[str, Any],
    manifest: Mapping[str, Any],
    now: datetime | None = None,
    *,
    card_root: Path,
    current_warp_version: str | None = None,
) -> tuple[str, ...]:
    """Return current compatibility reasons that prevent a historical record from supporting a claim."""

    validate_manifest(manifest)
    _validate_record_integrity(record)
    resolved_now = _resolve_now(now)
    timestamp = _parse_timestamp(record["environment"]["timestamp_utc"])
    maximum_age = manifest["compatibility"]["evidence_max_age_days"]
    reasons = []
    if resolved_now > timestamp + timedelta(days=maximum_age):
        reasons.append("evidence age exceeds the current maximum")
    if record["example_id"] != manifest["id"]:
        reasons.append("example ID differs from the current manifest")
    if "record_format_version" not in record:
        reasons.append("legacy record has no measured-contract envelope")
        return tuple(reasons)

    contract = record["measured_contract"]
    if not json_values_equal(contract["declared_workload"], manifest["benchmark"]["workload"]):
        reasons.append("declared workload changed")
    current_protocol_requirements = {name: manifest["benchmark"][name] for name in _PROTOCOL_KEYS}
    if contract["protocol_requirements"] != current_protocol_requirements:
        reasons.append("protocol requirements changed")
    if contract["outputs"] != _current_output_contract(manifest):
        reasons.append("observable outputs or tolerances changed")
    compatibility = contract["compatibility"]
    if compatibility["warp"]["specifier"] != manifest["compatibility"]["warp"]:
        reasons.append("Warp compatibility specifier changed")
    if compatibility["devices"] != manifest["compatibility"]["devices"]:
        reasons.append("supported device classes changed")
    if compatibility["equivalence_band"] != manifest["benchmark"]["equivalence_band"]:
        reasons.append("equivalence band changed")
    if current_warp_version is None:
        current_warp_version = wp.__version__
    if compatibility["warp"]["measured_version"] != current_warp_version:
        reasons.append("current Warp version differs from the measured version")

    try:
        current_source_hashes = _source_hash_snapshot(
            manifest,
            card_root,
            Path(__file__).resolve().parents[1],
        )
    except (OSError, ValueError):
        reasons.append("current source contract cannot be resolved")
    else:
        if current_source_hashes != contract["source_hashes"]:
            reasons.append("current source hashes changed")
    return tuple(reasons)


def is_evidence_stale(
    record: Mapping[str, Any],
    manifest: Mapping[str, Any],
    now: datetime | None = None,
    *,
    card_root: Path,
    current_warp_version: str | None = None,
) -> bool:
    """Return whether a record is incompatible with the current source contract."""

    return bool(
        evidence_staleness_reasons(
            record,
            manifest,
            now,
            card_root=card_root,
            current_warp_version=current_warp_version,
        )
    )


def _validate_claim_record(
    record: Mapping[str, Any],
    claim: Mapping[str, Any],
    platform: str,
    manifest: Mapping[str, Any],
    now: datetime,
    card_root: Path,
) -> None:
    reasons = evidence_staleness_reasons(
        record,
        manifest,
        now,
        card_root=card_root,
    )
    if reasons:
        raise ValueError(f"{platform} claim references stale record {record['record_id']}: {'; '.join(reasons)}")
    if record["environment"]["git"]["runtime_sources_dirty"] is not False:
        raise ValueError(f"{platform} claim requires clean runtime sources")
    contract = record["measured_contract"]
    if contract["compatibility"]["device"]["class"] != platform:
        raise ValueError(f"{platform.upper()} claim references a non-{platform} record")
    if not json_values_equal(claim["scope"]["workload"], contract["workload"]):
        raise ValueError(f"{platform} claim workload scope does not match its supporting record")
    if claim["scope"]["device"] != contract["compatibility"]["device"]:
        raise ValueError(f"{platform} claim device scope does not match its supporting record")
    if not record["correctness"]["passed"]:
        raise ValueError(f"{platform} claim requires passing correctness")

    impact = claim["impact"]
    if impact in {"improved", "harmful"}:
        if record["result"] != impact:
            raise ValueError(f"{platform} {impact} claim requires a matching {impact} record classification")
    else:
        band = contract["compatibility"]["equivalence_band"]
        statistics = record["statistics"]
        if statistics["ratio_ci_low"] < band["low"] or statistics["ratio_ci_high"] > band["high"]:
            raise ValueError(
                f"{platform} neutral claim requires the complete interval inside the predeclared equivalence band"
            )


def validate_evidence_document(
    document: Mapping[str, Any],
    manifest: Mapping[str, Any],
    now: datetime | None = None,
    future_skew: timedelta = _DEFAULT_FUTURE_SKEW,
    *,
    require_claim_support: bool = True,
    card_root: Path | None = None,
) -> None:
    """Validate retained history and, for publication, every structured claim."""

    validate_manifest(manifest)
    document = _require_mapping(document, "evidence document")
    _require_exact_keys(document, {"schema_version", "records"}, "evidence document")
    if document["schema_version"] != 1:
        raise ValueError("evidence document schema_version must be 1")
    records = document["records"]
    if not isinstance(records, list):
        raise ValueError("evidence document records must be an array")
    resolved_now = _resolve_now(now)
    record_by_id = {}
    for record in records:
        validate_evidence_record(record, now=resolved_now, future_skew=future_skew)
        record_id = record["record_id"]
        if record_id in record_by_id:
            raise ValueError(f"duplicate evidence record ID: {record_id}")
        record_by_id[record_id] = record

    if not require_claim_support:
        return
    claims = manifest["claims"]
    if any(claims[platform] for platform in ("cuda", "cpu")) and card_root is None:
        raise ValueError("publication claim validation requires the card root")
    for platform in ("cuda", "cpu"):
        for claim in claims[platform]:
            for record_id in claim["supporting_record_ids"]:
                try:
                    record = record_by_id[record_id]
                except KeyError:
                    raise ValueError(f"{platform} claim references missing evidence record {record_id}") from None
                _validate_claim_record(
                    record,
                    claim,
                    platform,
                    manifest,
                    resolved_now,
                    card_root,
                )


@contextmanager
def _single_writer(path: Path):
    lock_path = path.with_name(f".{path.name}.lock")
    deadline = time.monotonic() + _APPEND_LOCK_TIMEOUT_SECONDS
    descriptor = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for evidence lock: {lock_path.name}") from None
            time.sleep(0.01)
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode())
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def append_evidence(
    path: Path,
    record: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
    card_root: Path | None = None,
) -> None:
    """Atomically append a unique record after immutable retained history."""

    if manifest is not None and card_root is None:
        raise ValueError("manifest-aware append requires card_root")
    resolved_now = _resolve_now(None)
    validate_evidence_record(record, now=resolved_now)
    if manifest is not None:
        validate_manifest(manifest)
        if record["example_id"] != manifest["id"]:
            raise ValueError("new evidence example_id does not match the manifest")
        if evidence_staleness_reasons(
            record,
            manifest,
            resolved_now,
            card_root=card_root,
        ):
            raise ValueError("new evidence does not match the current measured contract")
    with _single_writer(path):
        if path.exists():
            document = load_json(path)
        else:
            document = {"schema_version": 1, "records": []}
        document = _require_mapping(document, "evidence document")
        _require_exact_keys(document, {"schema_version", "records"}, "evidence document")
        if document["schema_version"] != 1:
            raise ValueError("evidence document schema_version must be 1")
        records = document["records"]
        if not isinstance(records, list):
            raise ValueError("evidence document records must be an array")
        record_ids = set()
        for existing in records:
            _validate_record_integrity(existing)
            if existing["record_id"] in record_ids:
                raise ValueError(f"duplicate evidence record ID: {existing['record_id']}")
            record_ids.add(existing["record_id"])
        if record["record_id"] in record_ids:
            raise ValueError(f"duplicate evidence record ID: {record['record_id']}")

        updated_document = {
            "schema_version": 1,
            "records": [*records, dict(record)],
        }
        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary_path.write_text(
                json.dumps(
                    updated_document,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
