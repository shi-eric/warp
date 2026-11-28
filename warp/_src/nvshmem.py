# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVSHMEM support that needs neither CUDA, nvshmem4py, nor Warp's runtime.

Kernels that call NVSHMEM built-ins are linked against ``libnvshmem_device.ltoir.fatbin`` from the NVSHMEM
installation that owns the host library loaded in the process. This module locates that library by looking at the
file system and the installed distributions. It also classifies the NVSHMEM built-ins that a kernel calls, which
decides how the adjoint launches of the kernel are synchronized and checked.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from collections.abc import Sequence
from typing import NamedTuple

DEVICE_LIBRARY_ENV = "WARP_NVSHMEM_DEVICE_LIBRARY"
"""Environment variable naming the device library to link instead of the discovered one."""

DEVICE_LIBRARY_FILENAME = "libnvshmem_device.ltoir.fatbin"
"""File name of the multi-architecture LTOIR fatbin that NVSHMEM installs next to its host library."""

_WHEEL_LIBRARY_RELPATH = os.path.join("nvidia", "nvshmem", "lib", DEVICE_LIBRARY_FILENAME)


class DeviceLibrary(NamedTuple):
    """A device library selected for linking."""

    path: str
    """Absolute path of the file."""

    kind: str
    """``"fatbin"`` for a multi-architecture LTOIR fatbin or ``"ltoir"`` for a raw LTOIR image."""


def _kind_from_path(path: str) -> str:
    return "fatbin" if path.endswith(".fatbin") else "ltoir"


def find_distribution_library(cuda_major: int, search_path: list[str] | None = None) -> str | None:
    """Find the device library in the installed ``nvidia-nvshmem-cu<major>`` distribution.

    Args:
        cuda_major: CUDA major version that selects the distribution name.
        search_path: Directories to search for distributions. Defaults to ``sys.path``.

    Returns:
        The library path, or ``None`` when no installed distribution ships it.
    """
    for dist in _distributions(cuda_major, search_path):
        candidate = str(dist.locate_file(_WHEEL_LIBRARY_RELPATH))
        if os.path.isfile(candidate):
            return candidate
    return None


def _distributions(cuda_major: int, search_path: list[str] | None):
    kwargs = {} if search_path is None else {"path": list(search_path)}
    return list(importlib.metadata.Distribution.discover(name=_distribution_name(cuda_major), **kwargs))


def _distribution_name(cuda_major: int) -> str:
    return f"nvidia-nvshmem-cu{cuda_major}"


def find_device_library(
    host_library_dir: str | None,
    cuda_major: int,
    search_path: list[str] | None = None,
) -> DeviceLibrary:
    """Select the NVSHMEM device library to link.

    The override environment variable wins. Otherwise the library next to the loaded host library is used, then the
    one in the installed ``nvidia-nvshmem-cu<major>`` distribution.

    Args:
        host_library_dir: Directory of the loaded ``libnvshmem_host.so``, or ``None`` when none is loaded.
        cuda_major: CUDA major version of Warp's toolkit, which selects the distribution name.
        search_path: Directories to search for distributions. Defaults to ``sys.path``.

    Raises:
        RuntimeError: If the override names a missing file, or no device library is found.
    """
    override = os.environ.get(DEVICE_LIBRARY_ENV)
    if override:
        if not os.path.isfile(override):
            raise RuntimeError(f"{DEVICE_LIBRARY_ENV} names '{override}', which is not a file.")
        return DeviceLibrary(override, _kind_from_path(override))

    tried = []
    if host_library_dir:
        candidate = os.path.join(host_library_dir, DEVICE_LIBRARY_FILENAME)
        if os.path.isfile(candidate):
            return DeviceLibrary(candidate, "fatbin")
        tried.append(candidate)

    found = find_distribution_library(cuda_major, search_path)
    if found is not None:
        return DeviceLibrary(found, "fatbin")

    name = _distribution_name(cuda_major)
    installed = [dist.version for dist in _distributions(cuda_major, search_path)]
    if installed:
        tried.append(f"{_WHEEL_LIBRARY_RELPATH} in {name} {', '.join(installed)}, which does not ship it")
    else:
        tried.append(f"{_WHEEL_LIBRARY_RELPATH} in {name}, which is not installed")

    raise RuntimeError(
        "Cannot find the NVSHMEM device library needed to compile a module that uses NVSHMEM built-ins. "
        f"Tried: {'; '.join(tried)}. Install NVSHMEM 3.9 or newer, whose wheels ship {DEVICE_LIBRARY_FILENAME}, "
        f"or set {DEVICE_LIBRARY_ENV} to a {DEVICE_LIBRARY_FILENAME} or raw .ltoir file."
    )


_contents_cache: dict[tuple[str, int, int], tuple[bytes, str]] = {}


def read_device_library(lib: DeviceLibrary) -> tuple[bytes, str]:
    """Read a device library and return its bytes and their SHA-256 hex digest.

    Results are cached per process on the file's path, size, and modification time.
    """
    stat = os.stat(lib.path)
    key = (lib.path, stat.st_size, stat.st_mtime_ns)
    cached = _contents_cache.get(key)
    if cached is None:
        with open(lib.path, "rb") as f:
            contents = f.read()
        cached = (contents, hashlib.sha256(contents).hexdigest())
        _contents_cache.clear()
        _contents_cache[key] = cached
    return cached


_ADJOINT_REMOTE_ACCESS = frozenset({"nvshmem_p", "nvshmem_put", "nvshmem_put_signal", "nvshmem_g"})
"""Built-ins that leave a remote access in the adjoint kernel.

The push built-ins pull the destination gradient from the remote PE, and ``nvshmem_g`` is replayed.
"""

_PULL = frozenset({"nvshmem_g", "nvshmem_get"})
"""Built-ins that read remote memory and have no adjoint."""

_REMOTE_WRITE = frozenset({"nvshmem_p", "nvshmem_put", "nvshmem_put_signal", "nvshmem_signal_op"})
"""Built-ins that write remote memory."""

_SIGNAL_WAIT = "nvshmem_signal_wait_until"


class AdjointInfo(NamedTuple):
    """What the adjoint launches of a kernel need to know about the NVSHMEM built-ins that the kernel calls."""

    remote_access: bool
    """Whether the adjoint kernel accesses remote memory, so its launches must be ordered against the other PEs."""

    pull_builtins: tuple[str, ...]
    """Sorted keys of the pull built-ins the kernel calls, which have no adjoint."""

    fused_shape: bool
    """Whether the kernel writes to a remote PE before waiting on a signal, which autodiff does not support."""


def classify_adjoint(calls: Sequence[str]) -> AdjointInfo:
    """Summarize the NVSHMEM built-ins that a kernel calls for the adjoint launches of the kernel.

    Args:
        calls: Keys of the built-ins that the kernel and its callees call, in source order, as code generation
            records them.
    """
    remote_write_seen = False
    fused_shape = False
    for key in calls:
        if key == _SIGNAL_WAIT:
            fused_shape |= remote_write_seen
        elif key in _REMOTE_WRITE:
            remote_write_seen = True
    return AdjointInfo(
        remote_access=not _ADJOINT_REMOTE_ACCESS.isdisjoint(calls),
        pull_builtins=tuple(sorted(_PULL.intersection(calls))),
        fused_shape=fused_shape,
    )
