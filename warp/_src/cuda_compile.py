# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable CUDA compilation records used by targeted export.

This module deliberately has no Warp imports. It can therefore describe a
captured CUDA compilation without loading the runtime or a native library.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import weakref
from dataclasses import asdict, dataclass, field, fields
from typing import Any

CUDA_COMPILE_RECORD_VERSION = 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _require_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _require_non_negative_int(value: object, name: str) -> int:
    value = _require_int(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hexadecimal digest")
    return value


def _sha256_file(path: str) -> str:
    with open(path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


@dataclass(frozen=True)
class CompiledKernel:
    """Kernel entry points and launch requirements for a compiled CUDA module."""

    forward_name: str
    backward_name: str
    forward_smem_bytes: int
    backward_smem_bytes: int
    cluster_dim: int = 1


@dataclass(frozen=True)
class CudaNativeOptions:
    """The runtime-independent subset of options passed to ``build_cuda()``."""

    mode: str
    optimization_level: int
    verify_fp: bool
    fast_math: bool
    fuse_fp: bool
    lineinfo: bool
    compile_time_trace: bool
    llvm_cuda: bool
    use_precompiled_headers: bool
    extra_cuda_include_dirs: tuple[str, ...]
    cuda_arch_suffix: str


@dataclass(frozen=True)
class _CudaSource:
    """Weak-referenceable owner of immutable source bytes shared by live records."""

    data: bytes


_cuda_sources: weakref.WeakValueDictionary[str, _CudaSource] = weakref.WeakValueDictionary()
_cuda_sources_lock = threading.Lock()


def _share_cuda_source(source: bytes, digest: str) -> _CudaSource:
    with _cuda_sources_lock:
        snapshot = _cuda_sources.get(digest)
        if snapshot is None:
            snapshot = _CudaSource(source)
            _cuda_sources[digest] = snapshot
        elif snapshot.data != source:
            raise ValueError("CUDA source digest collision")
        return snapshot


@dataclass(frozen=True)
class CudaCompileRecord:
    """Canonical description of a CUDA compilation with non-serialized source bytes."""

    version: int
    module_hash: str
    block_dim: int
    source_basename: str
    source_sha256: str
    native_options: CudaNativeOptions
    kernels: tuple[CompiledKernel, ...]
    dependencies: tuple[tuple[str, str], ...]
    _source: _CudaSource = field(repr=False, metadata={"serialize": False})

    @property
    def source(self) -> bytes:
        """Return source bytes shared across records for as long as they are live."""
        return self._source.data

    @classmethod
    def create(
        cls,
        *,
        module_hash: bytes | str,
        block_dim: int,
        source: bytes,
        source_basename: str,
        native_options: CudaNativeOptions,
        kernels: tuple[CompiledKernel, ...],
        dependencies: tuple[str | os.PathLike[str] | tuple[str, str], ...],
    ) -> CudaCompileRecord:
        """Capture source and dependency digests into a canonical record."""
        if isinstance(module_hash, bytes):
            if len(module_hash) != 32:
                raise ValueError("module_hash must contain 32 bytes")
            module_hash = module_hash.hex()
        _require_sha256(module_hash, "module_hash")
        _require_non_negative_int(block_dim, "block_dim")
        if block_dim == 0:
            raise ValueError("block_dim must be greater than zero")
        if not isinstance(native_options, CudaNativeOptions):
            raise TypeError("native_options must be a CudaNativeOptions instance")
        _validate_native_options(native_options)

        _validate_source_basename(source_basename)
        if type(source) is not bytes:
            raise TypeError("source must be immutable bytes")
        source_sha256 = hashlib.sha256(source).hexdigest()

        canonical_kernels = _canonical_kernels(kernels)
        canonical_dependencies = _canonical_dependencies(dependencies)
        return cls(
            CUDA_COMPILE_RECORD_VERSION,
            module_hash,
            block_dim,
            source_basename,
            source_sha256,
            native_options,
            canonical_kernels,
            canonical_dependencies,
            _share_cuda_source(source, source_sha256),
        )

    def to_json(self) -> dict[str, Any]:
        """Return the versioned record payload without its in-memory source."""
        return {
            "version": self.version,
            "module_hash": self.module_hash,
            "block_dim": self.block_dim,
            "source_basename": self.source_basename,
            "source_sha256": self.source_sha256,
            "native_options": _native_options_to_json(self.native_options),
            "kernels": [asdict(kernel) for kernel in self.kernels],
            "dependencies": [list(dependency) for dependency in self.dependencies],
        }

    def _canonical_json_bytes(self) -> bytes:
        """Return the versioned record payload in its unique JSON representation."""
        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":")).encode()

    def fingerprint(self) -> str:
        """Return the SHA-256 fingerprint of the canonical JSON payload."""
        return hashlib.sha256(self._canonical_json_bytes()).hexdigest()

    @classmethod
    def from_json(cls, payload: object, source: bytes) -> CudaCompileRecord:
        """Restore a record from a cache manifest and its immutable source bytes."""
        try:
            return _record_from_json(payload, source)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid CUDA compile record: {error}") from error

    def validate(self, module_name: str) -> None:
        """Ensure the frozen source and external dependencies remain valid."""
        if type(module_name) is not str:
            raise TypeError("module_name must be a string")
        if not isinstance(self.source, bytes) or hashlib.sha256(self.source).hexdigest() != self.source_sha256:
            raise RuntimeError(f"CUDA compile record for module {module_name!r}: frozen source is invalid")

        for dependency_path, dependency_sha256 in self.dependencies:
            try:
                current_sha256 = _sha256_file(dependency_path)
            except OSError as error:
                raise RuntimeError(
                    f"CUDA compile record for module {module_name!r}: dependency {dependency_path!r} is missing"
                ) from error
            if current_sha256 != dependency_sha256:
                raise RuntimeError(
                    f"CUDA compile record for module {module_name!r}: dependency {dependency_path!r} changed after capture"
                )


def _canonical_kernels(kernels: object) -> tuple[CompiledKernel, ...]:
    if not isinstance(kernels, tuple):
        raise TypeError("kernels must be a tuple")
    seen = set()
    for kernel in kernels:
        if not isinstance(kernel, CompiledKernel):
            raise TypeError("kernels must contain CompiledKernel instances")
        if type(kernel.forward_name) is not str or not kernel.forward_name or type(kernel.backward_name) is not str:
            raise ValueError("Kernel names must be strings and the forward name must be non-empty")
        for name in (kernel.forward_name, kernel.backward_name):
            if name:
                if name in seen:
                    raise ValueError(f"Duplicate kernel symbol {name!r}")
                seen.add(name)
        for name, value in (
            ("forward_smem_bytes", kernel.forward_smem_bytes),
            ("backward_smem_bytes", kernel.backward_smem_bytes),
        ):
            _require_non_negative_int(value, name)
        if _require_non_negative_int(kernel.cluster_dim, "cluster_dim") == 0:
            raise ValueError("cluster_dim must be greater than zero")
    return tuple(sorted(kernels, key=lambda kernel: kernel.forward_name))


def _kernel_from_json(payload: object) -> CompiledKernel:
    if not isinstance(payload, dict):
        raise ValueError("kernels must be objects")
    _require_exact_keys(payload, {item.name for item in fields(CompiledKernel)}, "kernel")
    return CompiledKernel(**payload)


def _canonical_dependencies(dependencies: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(dependencies, tuple):
        raise TypeError("dependencies must be a tuple")
    result: dict[str, str] = {}
    for dependency in dependencies:
        if isinstance(dependency, tuple):
            if len(dependency) != 2 or type(dependency[0]) is not str:
                raise TypeError("dependency entries must be paths or (path, sha256) pairs")
            path, digest = dependency
            if not os.path.isabs(path):
                raise ValueError(f"dependency {path!r} must be absolute")
            path = os.path.realpath(path)
            _require_sha256(digest, "dependency sha256")
        else:
            path = os.fspath(dependency)
            if not os.path.isabs(path):
                raise ValueError(f"dependency {path!r} must be absolute")
            path = os.path.realpath(path)
            if not os.path.isfile(path):
                raise ValueError(f"dependency {path!r} must name an existing file")
            digest = _sha256_file(path)
        if path in result and result[path] != digest:
            raise ValueError(f"Conflicting dependency digests for {path!r}")
        result[path] = digest
    return tuple(sorted(result.items()))


def _native_options_to_json(options: CudaNativeOptions) -> dict[str, Any]:
    _validate_native_options(options)
    return asdict(options) | {"extra_cuda_include_dirs": list(options.extra_cuda_include_dirs)}


def _native_options_from_json(payload: object) -> CudaNativeOptions:
    if not isinstance(payload, dict):
        raise ValueError("native_options must be an object")
    _require_exact_keys(payload, {field.name for field in fields(CudaNativeOptions)}, "native_options")
    include_dirs = payload["extra_cuda_include_dirs"]
    if not isinstance(include_dirs, list) or any(type(path) is not str for path in include_dirs):
        raise ValueError("extra_cuda_include_dirs must be a list of strings")
    options = CudaNativeOptions(**(payload | {"extra_cuda_include_dirs": tuple(include_dirs)}))
    _validate_native_options(options)
    return options


def _validate_native_options(options: CudaNativeOptions) -> None:
    if type(options.mode) is not str or not options.mode:
        raise ValueError("native_options mode must be a non-empty string")
    _require_non_negative_int(options.optimization_level, "native_options optimization_level")
    for name in (
        "verify_fp",
        "fast_math",
        "fuse_fp",
        "lineinfo",
        "compile_time_trace",
        "llvm_cuda",
        "use_precompiled_headers",
    ):
        if type(getattr(options, name)) is not bool:
            raise ValueError(f"native_options {name} must be a bool")
    if not isinstance(options.extra_cuda_include_dirs, tuple) or any(
        type(path) is not str for path in options.extra_cuda_include_dirs
    ):
        raise ValueError("native_options extra_cuda_include_dirs must be a tuple of strings")
    if type(options.cuda_arch_suffix) is not str:
        raise ValueError("native_options cuda_arch_suffix must be a string")


def _require_exact_keys(payload: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} fields are invalid (missing={missing}, extra={extra})")


def _validate_source_basename(source_basename: object) -> None:
    if (
        type(source_basename) is not str
        or not source_basename
        or source_basename in (".", "..")
        or os.path.basename(source_basename) != source_basename
        or "/" in source_basename
        or "\\" in source_basename
    ):
        raise ValueError("source_basename must not contain path separators or traversal")


def _record_from_json(payload: object, source: bytes) -> CudaCompileRecord:
    if not isinstance(payload, dict):
        raise ValueError("record payload must be an object")
    serialized_fields = {item.name for item in fields(CudaCompileRecord) if item.metadata.get("serialize", True)}
    _require_exact_keys(payload, serialized_fields, "record")
    if type(payload["version"]) is not int or payload["version"] != CUDA_COMPILE_RECORD_VERSION:
        raise ValueError(f"unsupported CUDA compile record version {payload['version']!r}")
    _require_sha256(payload["module_hash"], "module_hash")
    _require_non_negative_int(payload["block_dim"], "block_dim")
    if payload["block_dim"] == 0:
        raise ValueError("block_dim must be greater than zero")
    source_basename = payload["source_basename"]
    _validate_source_basename(source_basename)
    _require_sha256(payload["source_sha256"], "source_sha256")
    if not isinstance(source, bytes) or hashlib.sha256(source).hexdigest() != payload["source_sha256"]:
        raise ValueError("source contents do not match source_sha256")
    native_options = _native_options_from_json(payload["native_options"])

    if not isinstance(payload["kernels"], list):
        raise ValueError("kernels must be a list")
    kernels = _canonical_kernels(tuple(_kernel_from_json(kernel) for kernel in payload["kernels"]))
    dependencies = _canonical_dependencies(_string_pair_tuple(payload["dependencies"], "dependencies"))
    return CudaCompileRecord(
        version=payload["version"],
        module_hash=payload["module_hash"],
        block_dim=payload["block_dim"],
        source_basename=source_basename,
        source_sha256=payload["source_sha256"],
        native_options=native_options,
        kernels=kernels,
        dependencies=dependencies,
        _source=_share_cuda_source(source, payload["source_sha256"]),
    )


def _string_pair_tuple(payload: object, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(payload, list):
        raise ValueError(f"{name} must be a list")
    pairs: list[tuple[str, str]] = []
    for pair in payload:
        if not isinstance(pair, list) or len(pair) != 2 or type(pair[0]) is not str or type(pair[1]) is not str:
            raise ValueError(f"{name} entries must be string pairs")
        pairs.append((pair[0], pair[1]))
    return tuple(pairs)
