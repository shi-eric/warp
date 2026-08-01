# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public, reproducible environment capture for optimization evidence."""

import os
import platform
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import warp as wp


def _cpu_model() -> str:
    cpuinfo_path = Path("/proc/cpuinfo")
    try:
        cpuinfo = cpuinfo_path.read_text(encoding="utf-8")
    except OSError:
        cpuinfo = ""
    for preferred_key in ("model name", "hardware", "processor"):
        for line in cpuinfo.splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() == preferred_key and value.strip():
                return value.strip()

    processor = platform.processor().strip()
    if processor:
        return processor
    return platform.machine() or "unknown"


def _capture_cpu() -> dict[str, str | int]:
    logical_cpu_count = os.cpu_count()
    try:
        affinity_cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_cpu_count = logical_cpu_count
    if logical_cpu_count is None:
        logical_cpu_count = affinity_cpu_count
    if logical_cpu_count is None or affinity_cpu_count is None:
        raise RuntimeError("CPU topology is unavailable")
    return {
        "model": _cpu_model(),
        "logical_cpu_count": logical_cpu_count,
        "affinity_cpu_count": affinity_cpu_count,
    }


def _run_git(arguments: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _capture_git() -> dict[str, str | bool | None]:
    source_path = Path(__file__).resolve()
    optimization_root = source_path.parents[1]
    repository_root_text = _run_git(["rev-parse", "--show-toplevel"], optimization_root)
    if repository_root_text is None:
        return {
            "revision": None,
            "repository_dirty": None,
            "runtime_sources_dirty": None,
        }

    repository_root = Path(repository_root_text)
    try:
        tracked_source_path = source_path.relative_to(repository_root)
    except ValueError:
        tracked_source = None
    else:
        tracked_source = _run_git(
            ["ls-files", "--error-unmatch", "--", tracked_source_path.as_posix()],
            repository_root,
        )
    if tracked_source is None:
        return {
            "revision": None,
            "repository_dirty": None,
            "runtime_sources_dirty": None,
        }

    revision = _run_git(["rev-parse", "HEAD"], repository_root)
    repository_status = _run_git(["status", "--porcelain", "--untracked-files=normal"], repository_root)
    try:
        runtime_path = optimization_root.relative_to(repository_root)
    except ValueError:
        runtime_status = None
    else:
        runtime_status = _run_git(
            ["status", "--porcelain", "--untracked-files=normal", "--", runtime_path.as_posix()],
            repository_root,
        )

    return {
        "revision": revision,
        "repository_dirty": None if repository_status is None else bool(repository_status),
        "runtime_sources_dirty": None if runtime_status is None else bool(runtime_status),
    }


def capture_environment(device: str, workload: Mapping[str, Any]) -> dict[str, Any]:
    """Capture public facts needed to reproduce one optimization measurement."""

    resolved_device = wp.get_device(device)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "warp": wp.__version__,
        "os": platform.platform(),
        "machine": platform.machine(),
        "git": _capture_git(),
        "device": {
            "alias": str(device),
            "name": resolved_device.name,
            "is_cuda": resolved_device.is_cuda,
            "architecture": resolved_device.arch,
            "total_memory_bytes": resolved_device.total_memory,
        },
        "cuda": {
            "toolkit": wp.get_cuda_toolkit_version(),
            "driver": wp.get_cuda_driver_version(),
        },
        "cpu": _capture_cpu(),
        "workload": dict(workload),
    }
