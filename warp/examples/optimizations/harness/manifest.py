# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free validation for optimization example manifests."""

import math
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from warp.examples.optimizations.harness.json_utils import json_values_equal, load_json

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "id",
        "title",
        "category",
        "status",
        "summary",
        "recognition",
        "applicability",
        "semantics",
        "impact",
        "claims",
        "compatibility",
        "artifacts",
        "benchmark",
        "clean_room",
    }
)
_CATEGORIES = frozenset(
    {
        "host-device-transfer-elimination",
        "synchronization-avoidance",
        "kernel-fusion",
        "launch-amortization",
        "allocation-reuse",
        "autodiff-strategy",
        "data-layout",
        "shared-memory-register-reuse",
        "work-decomposition",
        "device-native-substitution",
        "interoperability",
        "memory-runtime-tradeoff",
    }
)
_STATUSES = frozenset({"unverified", "recommended", "conditional", "rejected"})
_IMPACTS = frozenset({"improved", "neutral", "harmful", "unverified"})
_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _fail(field: str, message: str, path: Path | None) -> None:
    prefix = f"{path}: " if path is not None else ""
    raise ValueError(f"{prefix}{field}: {message}")


def _require_mapping(value: Any, field: str, path: Path | None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(field, "must be an object", path)
    return value


def _require_exact_keys(data: Mapping[str, Any], required: set[str], field: str, path: Path | None) -> None:
    missing = required - set(data)
    unknown = set(data) - required
    if missing:
        _fail(field, f"is missing required keys: {', '.join(sorted(missing))}", path)
    if unknown:
        _fail(field, f"contains unknown keys: {', '.join(sorted(unknown))}", path)


def _require_string(value: Any, field: str, path: Path | None) -> str:
    if not isinstance(value, str) or not value:
        _fail(field, "must be a non-empty string", path)
    return value


def _require_string_list(value: Any, field: str, path: Path | None) -> None:
    if not isinstance(value, list) or not value:
        _fail(field, "must be a non-empty list", path)
    for item in value:
        _require_string(item, field, path)


def _require_integer(value: Any, field: str, minimum: int | None, path: Path | None) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(field, "must be an integer", path)
    if minimum is not None and value < minimum:
        _fail(field, f"must be at least {minimum}", path)


def _require_nonnegative_number(value: Any, field: str, path: Path | None) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        _fail(field, "must be a finite non-negative number", path)


def _validate_relative_path(value: Any, field: str, path: Path | None) -> None:
    artifact_path = _require_string(value, field, path)
    posix_path = PurePosixPath(artifact_path)
    windows_path = PureWindowsPath(artifact_path)
    if (
        artifact_path.startswith(("/", "\\"))
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    ):
        _fail(field, "must be relative and cannot contain '..'", path)


def _validate_json_scalar_mapping(
    value: Any,
    field: str,
    path: Path | None,
    *,
    require_nonempty: bool = False,
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, field, path)
    if require_nonempty and not mapping:
        _fail(field, "must not be empty", path)
    for name, item in mapping.items():
        _require_string(name, f"{field} key", path)
        if item is None or isinstance(item, (str, bool)):
            continue
        if isinstance(item, int) and not isinstance(item, bool):
            continue
        if isinstance(item, float) and math.isfinite(item):
            continue
        _fail(f"{field}.{name}", "must be a JSON scalar", path)
    return mapping


def _validate_claim_device(value: Any, platform: str, field: str, path: Path | None) -> None:
    device = _require_mapping(value, field, path)
    keys = {
        "class",
        "name",
        "architecture",
        "total_memory_bytes",
        "cpu_model",
        "logical_cpu_count",
        "affinity_cpu_count",
    }
    _require_exact_keys(device, keys, field, path)
    if device["class"] != platform:
        _fail(f"{field}.class", f"must be {platform!r}", path)
    _require_string(device["name"], f"{field}.name", path)
    architecture = device["architecture"]
    if isinstance(architecture, bool) or (architecture is not None and not isinstance(architecture, (int, str))):
        _fail(f"{field}.architecture", "must be an integer, string, or null", path)
    _require_integer(device["total_memory_bytes"], f"{field}.total_memory_bytes", 0, path)
    if platform == "cuda":
        for name in ("cpu_model", "logical_cpu_count", "affinity_cpu_count"):
            if device[name] is not None:
                _fail(f"{field}.{name}", "must be null for CUDA claims", path)
    else:
        _require_string(device["cpu_model"], f"{field}.cpu_model", path)
        _require_integer(device["logical_cpu_count"], f"{field}.logical_cpu_count", 1, path)
        _require_integer(device["affinity_cpu_count"], f"{field}.affinity_cpu_count", 1, path)


def _validate_claims(
    value: Any,
    impact: Mapping[str, Any],
    status: str,
    default_workload: Mapping[str, Any],
    compatible_devices: set[str],
    path: Path | None,
) -> None:
    claims = _require_mapping(value, "claims", path)
    _require_exact_keys(claims, {"cuda", "cpu"}, "claims", path)
    for platform in ("cuda", "cpu"):
        platform_claims = claims[platform]
        if not isinstance(platform_claims, list):
            _fail(f"claims.{platform}", "must be an array", path)
        if platform_claims and platform not in compatible_devices:
            _fail(
                "compatibility.devices",
                f"must include {platform!r} when claims.{platform} is non-empty",
                path,
            )
        seen_record_ids = set()
        for index, value in enumerate(platform_claims):
            field = f"claims.{platform}[{index}]"
            claim = _require_mapping(value, field, path)
            _require_exact_keys(claim, {"impact", "supporting_record_ids", "scope"}, field, path)
            claim_impact = _require_string(claim["impact"], f"{field}.impact", path)
            if claim_impact not in {"improved", "neutral", "harmful"}:
                _fail(f"{field}.impact", "must be improved, neutral, or harmful", path)
            record_ids = claim["supporting_record_ids"]
            if not isinstance(record_ids, list) or not record_ids:
                _fail(f"{field}.supporting_record_ids", "must be a non-empty array", path)
            for record_id in record_ids:
                _require_string(record_id, f"{field}.supporting_record_ids", path)
                if record_id in seen_record_ids:
                    _fail(f"{field}.supporting_record_ids", f"duplicates record ID {record_id!r}", path)
                seen_record_ids.add(record_id)

            scope = _require_mapping(claim["scope"], f"{field}.scope", path)
            _require_exact_keys(scope, {"workload", "device"}, f"{field}.scope", path)
            _validate_json_scalar_mapping(
                scope["workload"],
                f"{field}.scope.workload",
                path,
                require_nonempty=True,
            )
            _validate_claim_device(scope["device"], platform, f"{field}.scope.device", path)

        declared_impact = impact[platform]
        if declared_impact == "unverified":
            if platform_claims:
                _fail(f"claims.{platform}", "must be empty when impact is unverified", path)
        elif not platform_claims:
            _fail(f"claims.{platform}", f"is required when impact is {declared_impact}", path)
        elif any(claim["impact"] != declared_impact for claim in platform_claims):
            _fail(f"impact.{platform}", "must match every structured claim", path)

    cuda_claims = claims["cuda"]
    if status == "unverified":
        if cuda_claims or impact["cuda"] != "unverified":
            _fail("status", "unverified cards cannot publish CUDA claims or impact labels", path)
    elif status in {"recommended", "conditional"}:
        if impact["cuda"] != "improved" or not cuda_claims:
            _fail("status", f"{status} requires an improved CUDA claim", path)
        if status == "recommended" and not any(
            json_values_equal(claim["scope"]["workload"], default_workload) for claim in cuda_claims
        ):
            _fail("status", "recommended claims must match the declared default workload", path)


def validate_manifest(data: Mapping[str, Any], path: Path | None = None) -> None:
    """Validate a manifest's stable metadata contract without dependencies."""

    manifest = _require_mapping(data, "manifest", path)
    _require_exact_keys(manifest, set(_TOP_LEVEL_KEYS), "manifest", path)

    schema_version = manifest["schema_version"]
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version != 1:
        _fail("schema_version", "must be 1", path)
    example_id = _require_string(manifest["id"], "id", path)
    if _ID_PATTERN.fullmatch(example_id) is None:
        _fail("id", "must use lowercase hyphen-separated words", path)
    _require_string(manifest["title"], "title", path)
    category = _require_string(manifest["category"], "category", path)
    if category not in _CATEGORIES:
        _fail("category", "is not supported", path)
    status = _require_string(manifest["status"], "status", path)
    if status not in _STATUSES:
        _fail("status", "is not supported", path)
    _require_string(manifest["summary"], "summary", path)

    recognition = _require_mapping(manifest["recognition"], "recognition", path)
    _require_exact_keys(recognition, {"signals"}, "recognition", path)
    _require_string_list(recognition["signals"], "recognition.signals", path)

    applicability = _require_mapping(manifest["applicability"], "applicability", path)
    _require_exact_keys(applicability, {"preconditions", "contraindications"}, "applicability", path)
    _require_string_list(applicability["preconditions"], "applicability.preconditions", path)
    _require_string_list(applicability["contraindications"], "applicability.contraindications", path)

    semantics = _require_mapping(manifest["semantics"], "semantics", path)
    _require_exact_keys(semantics, {"observable_outputs", "tolerance"}, "semantics", path)
    _require_string_list(semantics["observable_outputs"], "semantics.observable_outputs", path)
    tolerance = _require_mapping(semantics["tolerance"], "semantics.tolerance", path)
    _require_exact_keys(tolerance, {"relative", "absolute"}, "semantics.tolerance", path)
    _require_nonnegative_number(tolerance["relative"], "semantics.tolerance.relative", path)
    _require_nonnegative_number(tolerance["absolute"], "semantics.tolerance.absolute", path)

    impact = _require_mapping(manifest["impact"], "impact", path)
    _require_exact_keys(impact, {"cuda", "cpu", "mechanism"}, "impact", path)
    cuda_impact = _require_string(impact["cuda"], "impact.cuda", path)
    if cuda_impact not in _IMPACTS:
        _fail("impact.cuda", "is not supported", path)
    cpu_impact = _require_string(impact["cpu"], "impact.cpu", path)
    if cpu_impact not in _IMPACTS:
        _fail("impact.cpu", "is not supported", path)
    _require_string_list(impact["mechanism"], "impact.mechanism", path)

    compatibility = _require_mapping(manifest["compatibility"], "compatibility", path)
    _require_exact_keys(
        compatibility, {"warp", "devices", "evidence_max_age_days", "limitations"}, "compatibility", path
    )
    _require_string(compatibility["warp"], "compatibility.warp", path)
    devices = compatibility["devices"]
    _require_string_list(devices, "compatibility.devices", path)
    if any(device not in {"cpu", "cuda"} for device in devices):
        _fail("compatibility.devices", "must contain only cpu or cuda", path)
    if len(devices) != len(set(devices)):
        _fail("compatibility.devices", "must not contain duplicates", path)
    _require_integer(compatibility["evidence_max_age_days"], "compatibility.evidence_max_age_days", 1, path)
    _require_string_list(compatibility["limitations"], "compatibility.limitations", path)

    artifacts = _require_mapping(manifest["artifacts"], "artifacts", path)
    artifact_keys = {"python_module", "baseline", "candidate", "correctness", "benchmark", "explanation", "evidence"}
    _require_exact_keys(artifacts, artifact_keys, "artifacts", path)
    _require_string(artifacts["python_module"], "artifacts.python_module", path)
    for name in artifact_keys - {"python_module"}:
        _validate_relative_path(artifacts[name], f"artifacts.{name}", path)

    benchmark = _require_mapping(manifest["benchmark"], "benchmark", path)
    benchmark_keys = {
        "workload",
        "estimated_peak_bytes",
        "warmups",
        "pairs",
        "bootstrap_seed",
        "resamples",
        "equivalence_band",
    }
    _require_exact_keys(benchmark, benchmark_keys, "benchmark", path)
    workload = _validate_json_scalar_mapping(
        benchmark["workload"],
        "benchmark.workload",
        path,
        require_nonempty=True,
    )
    _require_integer(benchmark["estimated_peak_bytes"], "benchmark.estimated_peak_bytes", 0, path)
    _require_integer(benchmark["warmups"], "benchmark.warmups", 3, path)
    _require_integer(benchmark["pairs"], "benchmark.pairs", 10, path)
    _require_integer(benchmark["bootstrap_seed"], "benchmark.bootstrap_seed", 0, path)
    _require_integer(benchmark["resamples"], "benchmark.resamples", 10000, path)
    equivalence_band = _require_mapping(
        benchmark["equivalence_band"],
        "benchmark.equivalence_band",
        path,
    )
    _require_exact_keys(equivalence_band, {"low", "high"}, "benchmark.equivalence_band", path)
    _require_nonnegative_number(equivalence_band["low"], "benchmark.equivalence_band.low", path)
    _require_nonnegative_number(equivalence_band["high"], "benchmark.equivalence_band.high", path)
    if not equivalence_band["low"] < equivalence_band["high"]:
        _fail("benchmark.equivalence_band", "low must be below high", path)
    if not equivalence_band["low"] <= 1.0 <= equivalence_band["high"]:
        _fail("benchmark.equivalence_band", "must contain runtime parity", path)

    _validate_claims(manifest["claims"], impact, status, workload, set(devices), path)

    clean_room = _require_mapping(manifest["clean_room"], "clean_room", path)
    _require_exact_keys(clean_room, {"synthetic", "derived_from_private_source", "declaration"}, "clean_room", path)
    if clean_room["synthetic"] is not True:
        _fail("clean_room.synthetic", "must be true", path)
    if clean_room["derived_from_private_source"] is not False:
        _fail("clean_room.derived_from_private_source", "must be false", path)
    _require_string(clean_room["declaration"], "clean_room.declaration", path)


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate one JSON manifest from *path*."""

    data = load_json(path)
    validate_manifest(data, path)
    return data
