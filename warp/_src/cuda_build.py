# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile frozen CUDA programs and own their immutable cache artifacts."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from warp._src import build
from warp._src.cuda_compile import CompiledKernel, CudaCompileRecord


@dataclass(frozen=True)
class CudaArtifact:
    """One binary and its kernel descriptors, bound to the program that produced it."""

    binary_path: str
    binary_kind: Literal["ptx", "cubin"]
    target_arch: int
    arch_suffix: str
    kernels: tuple[CompiledKernel, ...]
    record: CudaCompileRecord

    @property
    def meta(self) -> dict[str, int]:
        return {
            name + "_smem_bytes": size
            for kernel in self.kernels
            for name, size in (
                (kernel.forward_name, kernel.forward_smem_bytes),
                (kernel.backward_name, kernel.backward_smem_bytes),
            )
            if name
        }


def _json_bytes(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_name(value) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Invalid CUDA cache digest")
    return value


def _write_atomic(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(contents)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def source_path(cache_dir, record: CudaCompileRecord) -> Path:
    """Return the content-addressed source location for a compile record."""
    return Path(cache_dir) / "cuda" / "sources" / f"{record.source_sha256}.cu"


def _publish_record(cache_dir, record: CudaCompileRecord) -> str:
    source = source_path(cache_dir, record)
    # Replacing a damaged entry with the identical immutable contents is safe.
    if not source.is_file() or _digest(source.read_bytes()) != record.source_sha256:
        _write_atomic(source, record.source)
    fingerprint = record.fingerprint()
    path = Path(cache_dir) / "cuda" / "records" / f"{fingerprint}.json"
    if not path.is_file() or _digest(path.read_bytes()) != fingerprint:
        _write_atomic(path, _json_bytes(record.to_json()))
    return fingerprint


def _read_artifact(
    cache_dir, fingerprint: str, module_name: str, inputs_fingerprint: str | None = None
) -> CudaArtifact:
    root = Path(cache_dir) / "cuda"
    directory = root / "artifacts" / _digest_name(fingerprint)
    manifest = json.loads((directory / "artifact.json").read_bytes())
    identity = manifest["identity"]
    if _digest(_json_bytes(manifest)) != fingerprint:
        raise ValueError("CUDA artifact manifest does not match its cache key")
    if inputs_fingerprint is not None and _digest(_json_bytes(identity)) != inputs_fingerprint:
        raise ValueError("CUDA artifact does not match the requested compilation inputs")
    record_fingerprint = _digest_name(identity["record"])
    record_bytes = (root / "records" / f"{record_fingerprint}.json").read_bytes()
    if _digest(record_bytes) != record_fingerprint:
        raise ValueError("CUDA compile record does not match its cache key")
    payload = json.loads(record_bytes)
    source = (root / "sources" / f"{_digest_name(payload['source_sha256'])}.cu").read_bytes()
    record = CudaCompileRecord.from_json(payload, source)
    record.validate(module_name)
    kind = identity["binary_kind"]
    if kind not in ("ptx", "cubin"):
        raise ValueError("Invalid CUDA artifact binary kind")
    kernels = tuple(CompiledKernel(**kernel) for kernel in identity["kernels"])
    if len(kernels) != len(record.kernels):
        raise ValueError("CUDA artifact kernel descriptors do not match its compile record")
    for kernel, original in zip(kernels, record.kernels, strict=True):
        if (kernel.forward_name, kernel.backward_name, kernel.cluster_dim) != (
            original.forward_name,
            original.backward_name,
            original.cluster_dim,
        ) or any(
            type(value) is not int or value < 0 for value in (kernel.forward_smem_bytes, kernel.backward_smem_bytes)
        ):
            raise ValueError("Invalid CUDA artifact kernel descriptor")
    binary = directory / f"module.{kind}"
    if _digest(binary.read_bytes()) != manifest["binary_sha256"]:
        raise ValueError("CUDA artifact binary does not match its manifest")
    return CudaArtifact(str(binary), kind, identity["target_arch"], identity["arch_suffix"], kernels, record)


def read_cuda_index(cache_dir, index_path, module_name: str, *, inputs_fingerprint: str | None = None) -> CudaArtifact:
    """Read and validate an ordinary module cache lookup exactly once."""
    index = json.loads(Path(index_path).read_bytes())
    return _read_artifact(cache_dir, index["artifact"], module_name, inputs_fingerprint)


def publish_cuda_index(index_path, artifact: CudaArtifact) -> None:
    """Point a mutable module lookup at a fully published immutable artifact."""
    _write_atomic(Path(index_path), _json_bytes({"artifact": Path(artifact.binary_path).parent.name}))


def publish_unsupported_index(index_path, reason: str) -> None:
    """Mark a conventional CUDA cache entry as unavailable for targeted export."""
    _write_atomic(Path(index_path), _json_bytes({"targeted_export_unsupported": reason}))


def read_unsupported_index(index_path) -> str | None:
    """Return the targeted-export exclusion attached to a conventional cache entry."""
    payload = json.loads(Path(index_path).read_bytes())
    reason = payload.get("targeted_export_unsupported")
    return reason if type(reason) is str and reason else None


def compile_cuda(
    record: CudaCompileRecord,
    module_name: str,
    cache_dir,
    target_arch: int,
    arch_suffix: str,
    binary_kind: Literal["ptx", "cubin"],
    *,
    pch_dir: str | None,
    use_cache: bool = True,
) -> tuple[CudaArtifact, bool]:
    """Compile one frozen program for JIT, AOT, or graph export."""
    record.validate(module_name)
    if type(target_arch) is not int or target_arch <= 0 or binary_kind not in ("ptx", "cubin"):
        raise ValueError("CUDA compilation requires a positive architecture and PTX or CUBIN output")
    for kernel in record.kernels:
        if kernel.cluster_dim > 1 and target_arch < 90:
            raise RuntimeError(
                f"Kernel {kernel.forward_name!r} requests cluster_dim={kernel.cluster_dim}, "
                f"but sm_{target_arch} is below sm_90 and the cluster attribute is dropped"
            )
    record_fingerprint = _publish_record(cache_dir, record)
    identity = {
        "record": record_fingerprint,
        "target_arch": target_arch,
        "arch_suffix": arch_suffix,
        "binary_kind": binary_kind,
        "kernels": [asdict(kernel) for kernel in record.kernels],
    }
    inputs_fingerprint = _digest(_json_bytes(identity))
    index_path = Path(cache_dir) / "cuda" / "targets" / f"{inputs_fingerprint}.json"
    root = Path(cache_dir) / "cuda" / "artifacts"
    if use_cache:
        try:
            return read_cuda_index(cache_dir, index_path, module_name, inputs_fingerprint=inputs_fingerprint), False
        except (OSError, ValueError, TypeError, KeyError, RuntimeError):
            # An incomplete or damaged cache entry is rebuilt from the frozen program.
            pass
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".compile-", dir=root) as staging_dir:
        staging = Path(staging_dir)
        source = staging / record.source_basename
        source.write_bytes(record.source)
        binary = staging / f"module.{binary_kind}"
        options = record.native_options
        build.build_cuda(
            str(source),
            target_arch,
            str(binary),
            config=options.mode,
            optimization_level=options.optimization_level,
            verify_fp=options.verify_fp,
            fast_math=options.fast_math,
            fuse_fp=options.fuse_fp,
            lineinfo=options.lineinfo,
            compile_time_trace=options.compile_time_trace,
            arch_suffix=arch_suffix,
            pch_dir=pch_dir,
            llvm_cuda=options.llvm_cuda,
            use_precompiled_headers=options.use_precompiled_headers,
            extra_include_dirs=options.extra_cuda_include_dirs,
        )
        manifest = {"identity": identity, "binary_sha256": _digest(binary.read_bytes())}
        # Include the output digest so forced builds can publish different bytes
        # without replacing an artifact retained by an existing executable.
        fingerprint = _digest(_json_bytes(manifest))
        directory = root / fingerprint
        (staging / "artifact.json").write_bytes(_json_bytes(manifest))
        source.unlink()
        try:
            os.rename(staging, directory)
        except OSError as publication_error:
            if publication_error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            # Concurrent publishers of this fingerprint produced identical bytes.
            try:
                _read_artifact(cache_dir, fingerprint, module_name)
            except (OSError, ValueError, TypeError, KeyError, RuntimeError):
                # Repair individual files atomically with their original contents.
                # Moving the directory could remove another repair's valid output
                # after it has already been handed to a loader.
                _write_atomic(directory / binary.name, binary.read_bytes())
                _write_atomic(directory / "artifact.json", _json_bytes(manifest))
    artifact = CudaArtifact(
        str(directory / f"module.{binary_kind}"), binary_kind, target_arch, arch_suffix, record.kernels, record
    )
    publish_cuda_index(index_path, artifact)
    return artifact, True


def export_cuda_artifact(artifact: CudaArtifact, binary_path, *, overwrite: bool = False) -> Path:
    """Write the conventional binary, metadata, and source files for an AOT caller."""
    binary_path = Path(binary_path)
    for path, data in (
        (binary_path, Path(artifact.binary_path).read_bytes()),
        (binary_path.with_suffix(".meta"), _json_bytes(artifact.meta)),
        (binary_path.parent / artifact.record.source_basename, artifact.record.source),
    ):
        if overwrite or not path.is_file() or path.read_bytes() != data:
            _write_atomic(path, data)
    return binary_path
