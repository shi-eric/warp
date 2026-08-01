# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery for runtime-optimization example manifests."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from warp.examples.optimizations.harness.manifest import load_manifest


@dataclass(frozen=True)
class ExampleRecord:
    """A validated optimization example and the directory that contains it."""

    example_id: str
    root: Path
    manifest: Mapping[str, Any]


def discover_examples(root: Path | None = None) -> dict[str, ExampleRecord]:
    """Discover validated manifests below *root*, indexed in ID order."""

    examples_root = root if root is not None else Path(__file__).resolve().parent.parent
    records: dict[str, ExampleRecord] = {}
    for manifest_path in sorted(examples_root.rglob("manifest.json")):
        manifest = load_manifest(manifest_path)
        example_id = manifest["id"]
        if example_id in records:
            raise ValueError(f"duplicate optimization example id: {example_id}")
        records[example_id] = ExampleRecord(example_id, manifest_path.parent, manifest)

    return {example_id: records[example_id] for example_id in sorted(records)}
