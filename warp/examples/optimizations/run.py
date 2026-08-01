# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line runner for runtime-optimization example cards."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from warp.examples.optimizations.harness.benchmark import run_paired
from warp.examples.optimizations.harness.clean_room import scan_prohibited
from warp.examples.optimizations.harness.correctness import CorrectnessResult, check_correctness
from warp.examples.optimizations.harness.environment import capture_environment
from warp.examples.optimizations.harness.evidence import (
    append_evidence,
    build_evidence_record,
    build_measured_contract,
    is_evidence_stale,
    validate_evidence_document,
)
from warp.examples.optimizations.harness.json_utils import json_values_equal, load_json
from warp.examples.optimizations.harness.manifest import validate_manifest
from warp.examples.optimizations.harness.model import (
    JSONScalar,
    OptimizationCase,
    Tolerance,
    UnsupportedWorkload,
)
from warp.examples.optimizations.harness.registry import ExampleRecord, discover_examples
from warp.examples.optimizations.harness.statistics import summarize_paired

_OPTIMIZATION_ROOT = Path(__file__).resolve().parent
_SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
_SCHEMA_TYPES = frozenset({"array", "boolean", "integer", "null", "number", "object", "string"})
_SCHEMA_KEYWORDS = frozenset(
    {
        "$defs",
        "$ref",
        "$schema",
        "additionalProperties",
        "allOf",
        "anyOf",
        "const",
        "enum",
        "exclusiveMinimum",
        "format",
        "if",
        "items",
        "maxItems",
        "minItems",
        "minLength",
        "minProperties",
        "minimum",
        "pattern",
        "properties",
        "required",
        "then",
        "type",
        "uniqueItems",
    }
)


def _schema_sequence(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty array")
    return value


def _schema_nonnegative_integer(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")


def _validate_schema_node(node: Any, location: str, references: list[tuple[str, str]]) -> None:
    if isinstance(node, bool):
        return
    if not isinstance(node, Mapping):
        raise ValueError(f"{location} must be a schema object or boolean")

    unknown = set(node) - _SCHEMA_KEYWORDS
    if unknown:
        raise ValueError(f"{location} contains unsupported schema keywords: {', '.join(sorted(unknown))}")

    schema_type = node.get("type")
    if schema_type is not None:
        types = [schema_type] if isinstance(schema_type, str) else _schema_sequence(schema_type, f"{location}.type")
        if any(not isinstance(item, str) or item not in _SCHEMA_TYPES for item in types):
            raise ValueError(f"{location}.type contains an unsupported JSON type")
        if len(types) != len(set(types)):
            raise ValueError(f"{location}.type contains duplicate entries")

    properties = node.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping):
            raise ValueError(f"{location}.properties must be an object")
        for name, child in properties.items():
            _validate_schema_node(child, f"{location}.properties.{name}", references)

    required = node.get("required")
    if required is not None:
        required = _schema_sequence(required, f"{location}.required")
        if any(not isinstance(item, str) or not item for item in required):
            raise ValueError(f"{location}.required entries must be non-empty strings")
        if len(required) != len(set(required)):
            raise ValueError(f"{location}.required contains duplicate entries")
        if properties is not None and not set(required).issubset(properties):
            raise ValueError(f"{location}.required names must exist in properties")

    definitions = node.get("$defs")
    if definitions is not None:
        if not isinstance(definitions, Mapping):
            raise ValueError(f"{location}.$defs must be an object")
        for name, child in definitions.items():
            _validate_schema_node(child, f"{location}.$defs.{name}", references)

    additional = node.get("additionalProperties")
    if additional is not None:
        _validate_schema_node(additional, f"{location}.additionalProperties", references)

    items = node.get("items")
    if items is not None:
        _validate_schema_node(items, f"{location}.items", references)

    for keyword in ("allOf", "anyOf"):
        if keyword in node:
            for index, child in enumerate(_schema_sequence(node[keyword], f"{location}.{keyword}")):
                _validate_schema_node(child, f"{location}.{keyword}[{index}]", references)

    for keyword in ("if", "then"):
        if keyword in node:
            _validate_schema_node(node[keyword], f"{location}.{keyword}", references)

    reference = node.get("$ref")
    if reference is not None:
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"{location}.$ref must be a non-empty string")
        references.append((location, reference))

    pattern = node.get("pattern")
    if pattern is not None:
        if not isinstance(pattern, str):
            raise ValueError(f"{location}.pattern must be a string")
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(f"{location}.pattern is invalid: {error}") from error

    for keyword in ("minItems", "maxItems", "minLength", "minProperties"):
        if keyword in node:
            _schema_nonnegative_integer(node[keyword], f"{location}.{keyword}")
    if "minItems" in node and "maxItems" in node and node["minItems"] > node["maxItems"]:
        raise ValueError(f"{location}.minItems cannot exceed maxItems")
    if "uniqueItems" in node and not isinstance(node["uniqueItems"], bool):
        raise ValueError(f"{location}.uniqueItems must be a boolean")

    for keyword in ("minimum", "exclusiveMinimum"):
        if keyword in node:
            value = node[keyword]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{location}.{keyword} must be a finite number")

    if "enum" in node:
        values = _schema_sequence(node["enum"], f"{location}.enum")
        serialized = [json.dumps(value, sort_keys=True, allow_nan=False) for value in values]
        if len(serialized) != len(set(serialized)):
            raise ValueError(f"{location}.enum contains duplicate entries")

    schema_format = node.get("format")
    if schema_format is not None and (not isinstance(schema_format, str) or not schema_format):
        raise ValueError(f"{location}.format must be a non-empty string")


def _resolve_schema_reference(document: Mapping[str, Any], location: str, reference: str) -> None:
    if not reference.startswith("#/"):
        raise ValueError(f"{location}.$ref must be a local JSON Pointer")
    target: Any = document
    for encoded_part in reference[2:].split("/"):
        part = encoded_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(target, Mapping) or part not in target:
            raise ValueError(f"{location}.$ref does not resolve: {reference}")
        target = target[part]


def _validate_schema_document(path: Path) -> None:
    document = load_json(path)
    if not isinstance(document, Mapping):
        raise ValueError(f"{path}: schema document must be an object")
    if document.get("$schema") != _SCHEMA_URI:
        raise ValueError(f"{path}: $schema must declare JSON Schema draft 2020-12")
    if document.get("type") != "object":
        raise ValueError(f"{path}: root schema type must be object")

    references: list[tuple[str, str]] = []
    _validate_schema_node(document, path.name, references)
    for location, reference in references:
        _resolve_schema_reference(document, location, reference)


def _parse_json_scalar(text: str) -> JSONScalar:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(value, (dict, list)):
        raise ValueError("workload override values must be JSON scalars")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("workload override numbers must be finite")
    return value


def _apply_workload_overrides(
    workload: Mapping[str, JSONScalar],
    assignments: Sequence[str],
) -> dict[str, JSONScalar]:
    updated = dict(workload)
    overridden = set()
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"workload override must use KEY=VALUE: {assignment}")
        key, value = assignment.split("=", 1)
        if not key:
            raise ValueError("workload override key must not be empty")
        if key not in updated:
            raise ValueError(f"unknown workload key: {key}")
        if key in overridden:
            raise ValueError(f"duplicate workload override: {key}")
        updated[key] = _parse_json_scalar(value)
        overridden.add(key)
    return updated


def _integer_at_least(minimum: int):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be an integer") from error
        if parsed < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}")
        return parsed

    return parse


def _select_example(example_id: str, registry_root: Path | None = None) -> ExampleRecord:
    examples = discover_examples(registry_root)
    try:
        return examples[example_id]
    except KeyError:
        available = ", ".join(examples) if examples else "none"
        raise ValueError(f"unknown optimization example '{example_id}' (available: {available})") from None


def _expected_benchmark_module_path(record: ExampleRecord) -> Path:
    card_root = record.root.resolve()
    relative_path = record.manifest["artifacts"]["benchmark"]
    benchmark_path = (card_root / relative_path).resolve(strict=True)
    try:
        benchmark_path.relative_to(card_root)
    except ValueError:
        raise ValueError(f"{record.example_id}: hashed benchmark artifact resolves outside the card root") from None
    if not benchmark_path.is_file():
        raise ValueError(f"{record.example_id}: hashed benchmark artifact is not a file")
    return benchmark_path


def _expected_benchmark_module_name(record: ExampleRecord, benchmark_path: Path) -> str:
    card_root = record.root.resolve()
    package_parts = []
    package_root = card_root
    while (package_root / "__init__.py").is_file():
        if not package_root.name.isidentifier():
            raise ValueError(f"{record.example_id}: hashed benchmark artifact is not in an importable package")
        package_parts.append(package_root.name)
        package_root = package_root.parent
    if not package_parts:
        raise ValueError(f"{record.example_id}: hashed benchmark artifact is not in an importable package")

    relative_path = benchmark_path.relative_to(card_root)
    nested_parts = list(relative_path.parts[:-1])
    nested_root = card_root
    for part in nested_parts:
        nested_root /= part
        if not part.isidentifier() or not (nested_root / "__init__.py").is_file():
            raise ValueError(f"{record.example_id}: hashed benchmark artifact is not in an importable package")

    module_stem = relative_path.stem
    if relative_path.suffix != ".py" or module_stem == "__init__" or not module_stem.isidentifier():
        raise ValueError(f"{record.example_id}: hashed benchmark artifact is not an importable module")
    return ".".join([*reversed(package_parts), *nested_parts, module_stem])


def _validate_benchmark_module_binding(record: ExampleRecord) -> str:
    expected_module_path = _expected_benchmark_module_path(record)
    expected_module_name = _expected_benchmark_module_name(record, expected_module_path)
    module_name = record.manifest["artifacts"]["python_module"]
    if module_name != expected_module_name:
        raise ValueError(f"{module_name} must name the hashed benchmark artifact")

    module_spec = importlib.util.find_spec(module_name)
    if module_spec is None or module_spec.origin is None:
        raise ValueError(f"{module_name} must resolve to the hashed benchmark artifact")
    try:
        resolved_module_path = Path(module_spec.origin).resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{module_name} must resolve to the hashed benchmark artifact") from error
    if resolved_module_path != expected_module_path:
        raise ValueError(f"{module_name} must resolve to the hashed benchmark artifact")
    return module_name


def _build_case(record: ExampleRecord, device: str, workload: Mapping[str, JSONScalar]) -> OptimizationCase:
    module_name = _validate_benchmark_module_binding(record)

    module = importlib.import_module(module_name)
    build_case = getattr(module, "build_case", None)
    if not callable(build_case):
        raise ValueError(f"{module_name} must export build_case(device, workload)")
    case = build_case(device, workload)
    if not isinstance(case, OptimizationCase):
        raise ValueError(f"{module_name}.build_case() must return OptimizationCase")
    if case.example_id != record.example_id:
        raise ValueError(f"{module_name}.build_case() returned the wrong example ID")
    if not json_values_equal(dict(case.workload), dict(workload)):
        raise ValueError(f"{module_name}.build_case() changed the requested workload")
    expected_outputs = set(record.manifest["semantics"]["observable_outputs"])
    if set(case.tolerances) != expected_outputs:
        raise ValueError(f"{module_name}.build_case() tolerances do not match manifest output keys")
    declared = record.manifest["semantics"]["tolerance"]
    expected_tolerance = Tolerance(
        atol=declared["absolute"],
        rtol=declared["relative"],
    )
    for name in sorted(expected_outputs):
        if case.tolerances[name] != expected_tolerance:
            raise ValueError(f"{module_name}.build_case() tolerances do not match manifest semantics")
    return case


def _validate_correctness_result(result: CorrectnessResult, manifest: Mapping[str, Any]) -> None:
    expected_outputs = set(manifest["semantics"]["observable_outputs"])
    if set(result.outputs) != expected_outputs:
        raise ValueError("correctness outputs do not match manifest semantics")


def _print_correctness(result: CorrectnessResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"Correctness: {status}")
    print("OUTPUT  PASSED  FINITE  MAX_ABS  MAX_REL  MAX_NORMALIZED")
    for name in sorted(result.outputs):
        output = result.outputs[name]
        max_abs = "null" if output.max_abs is None else f"{output.max_abs:.12g}"
        max_rel = "null" if output.max_rel is None else f"{output.max_rel:.12g}"
        max_normalized = "null" if output.max_normalized is None else f"{output.max_normalized:.12g}"
        print(
            f"{name}  {'yes' if output.passed else 'no'}  "
            f"{'yes' if output.finite else 'no'}  {max_abs}  {max_rel}  {max_normalized}"
        )


def _command_list(_arguments: argparse.Namespace) -> int:
    examples = discover_examples(_arguments.registry_root)
    print("ID  STATUS  CUDA  CPU  CATEGORY  TITLE")
    for example_id, record in examples.items():
        manifest = record.manifest
        print(
            f"{example_id}  {manifest['status']}  {manifest['impact']['cuda']}  "
            f"{manifest['impact']['cpu']}  {manifest['category']}  {manifest['title']}"
        )
    return 0


def _command_check(arguments: argparse.Namespace) -> int:
    record = _select_example(arguments.example, arguments.registry_root)
    workload = dict(record.manifest["benchmark"]["workload"])
    case = _build_case(record, arguments.device, workload)
    result = check_correctness(case)
    _validate_correctness_result(result, record.manifest)
    _print_correctness(result)
    return 0 if result.passed else 1


def _command_benchmark(arguments: argparse.Namespace) -> int:
    record = _select_example(arguments.example, arguments.registry_root)
    manifest = record.manifest
    workload = _apply_workload_overrides(manifest["benchmark"]["workload"], arguments.set)
    protocol = {
        name: (manifest["benchmark"][name] if getattr(arguments, name) is None else getattr(arguments, name))
        for name in ("warmups", "pairs", "bootstrap_seed", "resamples")
    }
    for name, value in protocol.items():
        required = manifest["benchmark"][name]
        if value < required:
            raise ValueError(f"{name} is {value}, but the selected manifest requires at least {required}")

    output_parent = arguments.output.parent
    if not output_parent.exists() or not output_parent.is_dir():
        raise ValueError(f"output parent is not an existing directory: {output_parent}")
    if arguments.output.exists() and not arguments.output.is_file():
        raise ValueError(f"output path is not a file: {arguments.output}")
    if arguments.output.exists():
        validate_evidence_document(
            load_json(arguments.output),
            manifest,
            require_claim_support=False,
        )
    case = _build_case(record, arguments.device, workload)

    correctness = check_correctness(case)
    _validate_correctness_result(correctness, manifest)
    _print_correctness(correctness)
    if not correctness.passed:
        raise ValueError("correctness failed; benchmark was not run")

    warmups = protocol["warmups"]
    pairs = protocol["pairs"]
    bootstrap_seed = protocol["bootstrap_seed"]
    resamples = protocol["resamples"]
    samples = run_paired(case, warmups=warmups, pairs=pairs)
    summary = summarize_paired(samples, bootstrap_seed=bootstrap_seed, resamples=resamples)
    environment = capture_environment(arguments.device, workload)
    measured_contract = build_measured_contract(
        manifest=manifest,
        card_root=record.root,
        environment=environment,
        correctness=correctness,
        warmups=warmups,
        pairs=pairs,
        bootstrap_seed=bootstrap_seed,
        resamples=resamples,
    )
    evidence = build_evidence_record(
        example_id=record.example_id,
        environment=environment,
        correctness=correctness,
        samples=samples,
        summary=summary,
        warmups=warmups,
        bootstrap_seed=bootstrap_seed,
        resamples=resamples,
        limitations=manifest["compatibility"]["limitations"],
        measured_contract=measured_contract,
    )
    append_evidence(
        arguments.output,
        evidence,
        manifest=manifest,
        card_root=record.root,
    )

    print(f"Baseline median: {summary.baseline_median_ns:.12g} ns")
    print(f"Candidate median: {summary.candidate_median_ns:.12g} ns")
    print(f"Paired candidate/baseline ratio: {summary.median_ratio:.12g}")
    print(f"Paired 95% CI: [{summary.ratio_ci_low:.12g}, {summary.ratio_ci_high:.12g}]")
    print(f"Classification: {evidence['result']}")
    print(f"Evidence: {arguments.output}")
    return 0


def _load_deny_patterns(path: Path) -> list[str]:
    patterns = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not patterns:
        raise ValueError(f"{path}: deny-pattern file contains no non-empty patterns")
    return patterns


def _resolve_artifacts(record: ExampleRecord) -> dict[str, Path]:
    card_root = record.root.resolve()
    resolved_artifacts = {}
    for name, relative_path in record.manifest["artifacts"].items():
        if name == "python_module":
            continue
        artifact_path = (card_root / relative_path).resolve(strict=True)
        try:
            artifact_path.relative_to(card_root)
        except ValueError:
            raise ValueError(f"{record.example_id}: {name} artifact resolves outside card root") from None
        if not artifact_path.is_file():
            raise ValueError(f"{record.example_id}: missing {name} artifact: {artifact_path}")
        resolved_artifacts[name] = artifact_path
    return resolved_artifacts


def _command_validate(arguments: argparse.Namespace) -> int:
    schema_root = _OPTIMIZATION_ROOT / "schema"
    schema_paths = (schema_root / "example.schema.json", schema_root / "evidence.schema.json")
    for schema_path in schema_paths:
        _validate_schema_document(schema_path)
        print(f"{schema_path.name}: valid")

    examples = discover_examples(arguments.registry_root)
    for example_id, record in examples.items():
        manifest_path = record.root / "manifest.json"
        manifest = record.manifest
        validate_manifest(manifest, manifest_path)
        artifacts = _resolve_artifacts(record)
        _validate_benchmark_module_binding(record)

        evidence = load_json(artifacts["evidence"])
        validate_evidence_document(
            evidence,
            manifest,
            card_root=record.root,
        )
        stale_records = sum(is_evidence_stale(item, manifest, card_root=record.root) for item in evidence["records"])
        print(f"{example_id}: valid ({len(evidence['records'])} evidence records, {stale_records} stale)")

    if arguments.deny_pattern_file is not None:
        patterns = _load_deny_patterns(arguments.deny_pattern_file)
        scan_root = _OPTIMIZATION_ROOT if arguments.registry_root is None else arguments.registry_root
        findings = scan_prohibited(scan_root, patterns)
        if findings:
            first = findings[0]
            relative_path = first.path.relative_to(scan_root)
            pattern_index = patterns.index(first.pattern) + 1
            pattern_hash = hashlib.sha256(first.pattern.encode()).hexdigest()[:12]
            raise ValueError(
                f"prohibited pattern #{pattern_index} ({pattern_hash}) at "
                f"{relative_path}:{first.line} ({len(findings)} findings)"
            )
        print(f"Clean-room scan: valid ({len(patterns)} patterns)")

    print(f"Examples: {len(examples)} valid")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-root",
        type=Path,
        help=argparse.SUPPRESS,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available optimization cards.")
    list_parser.set_defaults(handler=_command_list)

    check_parser = subparsers.add_parser("check", help="Check one card's declared outputs.")
    check_parser.add_argument("--example", required=True, help="Optimization card ID.")
    check_parser.add_argument("--device", required=True, help="Warp device alias.")
    check_parser.set_defaults(handler=_command_check)

    benchmark_parser = subparsers.add_parser("benchmark", help="Generate paired runtime evidence.")
    benchmark_parser.add_argument("--example", required=True, help="Optimization card ID.")
    benchmark_parser.add_argument("--device", required=True, help="Warp device alias.")
    benchmark_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one declared workload value with a JSON scalar.",
    )
    benchmark_parser.add_argument("--warmups", type=_integer_at_least(3))
    benchmark_parser.add_argument("--pairs", type=_integer_at_least(10))
    benchmark_parser.add_argument("--bootstrap-seed", type=_integer_at_least(0))
    benchmark_parser.add_argument("--resamples", type=_integer_at_least(10_000))
    benchmark_parser.add_argument("--output", required=True, type=Path, help="Explicit evidence output path.")
    benchmark_parser.set_defaults(handler=_command_benchmark)

    validate_parser = subparsers.add_parser("validate", help="Validate schemas, cards, and evidence.")
    validate_parser.add_argument(
        "--deny-pattern-file",
        type=Path,
        help="Local newline-separated clean-room deny-pattern file.",
    )
    validate_parser.set_defaults(handler=_command_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the optimization corpus command-line interface."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        return arguments.handler(arguments)
    except (ImportError, OSError, RuntimeError, UnsupportedWorkload, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
