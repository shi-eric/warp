# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller-configured clean-room scans for optimization examples."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Finding:
    """One prohibited literal found in a checked-in text file."""

    path: Path
    line: int
    pattern: str


def scan_prohibited(root: Path, patterns: Sequence[str]) -> list[Finding]:
    """Return case-insensitive literal matches in Python, Markdown, and JSON files."""

    normalized_patterns = [(pattern, pattern.lower()) for pattern in patterns]
    findings: list[Finding] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".md", ".json"}:
            continue
        if any(part in {".git", "__pycache__"} for part in path.relative_to(root).parts):
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            normalized_line = line.lower()
            for pattern, normalized_pattern in normalized_patterns:
                if normalized_pattern in normalized_line:
                    findings.append(Finding(path, line_number, pattern))
    return findings
