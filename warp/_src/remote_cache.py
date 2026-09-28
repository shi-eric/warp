# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal second-tier compilation cache primitives."""

import gzip
import hashlib
import io
import json
import math
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal, TypeAlias

import warp.config

ARCHIVE_FORMAT_VERSION = 1
_MAX_COMPRESSED_BYTES = 1 << 30
_MAX_EXTRACTED_BYTES = 4 << 30
_MAX_ARTIFACT_BYTES = 2 << 30
_MAX_MANIFEST_BYTES = 1 << 20
_CHUNK_BYTES = 1 << 20

JSONValue: TypeAlias = "str | int | float | bool | list[JSONValue] | dict[str, JSONValue] | None"


def _validate_json_value(value: object) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Remote cache identity contains a non-finite float")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Remote cache identity keys must be strings")
            _validate_json_value(item)
        return
    raise ValueError(f"Unsupported remote cache identity value: {type(value).__name__}")


def _is_basename(name: str) -> bool:
    return bool(name) and name not in (".", "..", "manifest.json") and "/" not in name and "\\" not in name


@dataclass(frozen=True)
class RemoteCacheEntry:
    kind: Literal["kernel", "lto"]
    namespace: str
    identity: Mapping[str, JSONValue]
    artifact_names: tuple[str, ...]
    _canonical: bytes = field(init=False, repr=False)
    _version: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.kind not in ("kernel", "lto"):
            raise ValueError("Invalid remote cache entry kind")
        if not isinstance(self.namespace, str) or not _is_basename(self.namespace):
            raise ValueError("Invalid remote cache namespace")
        if not isinstance(self.identity, Mapping):
            raise ValueError("Remote cache identity must be a mapping")
        _validate_json_value(self.identity)
        names = tuple(self.artifact_names)
        if (
            not names
            or len(names) != len(set(names))
            or any(not isinstance(name, str) or not _is_basename(name) for name in names)
        ):
            raise ValueError("Invalid remote cache artifact names")
        version = warp.config.version
        canonical = json.dumps(
            {
                "archive_format_version": ARCHIVE_FORMAT_VERSION,
                "warp_version": version,
                "kind": self.kind,
                "namespace": self.namespace,
                "artifact_names": list(names),
                "identity": self.identity,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        object.__setattr__(self, "artifact_names", names)
        object.__setattr__(self, "_canonical", canonical)
        object.__setattr__(self, "_version", version)

    def canonical_identity(self) -> bytes:
        return self._canonical

    def digest(self) -> str:
        return hashlib.sha256(self._canonical).hexdigest()

    def object_uri(self, root: str) -> str:
        return f"{root.rstrip('/')}/{self._version}/{self.namespace}/{self.digest()}.tar.gz"


class RemoteCacheValidationError(Exception):
    """A remote archive does not match its expected compilation identity."""


class _CountingWriter:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.count = 0

    def write(self, data: bytes) -> int:
        self.count += len(data)
        if self.count > _MAX_COMPRESSED_BYTES:
            raise RemoteCacheValidationError("Remote cache archive exceeds compressed size limit")
        return self.stream.write(data)

    def flush(self) -> None:
        self.stream.flush()


class _CountingReader:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.count = 0

    def read(self, size: int = -1) -> bytes:
        data = self.stream.read(size)
        self.count += len(data)
        if self.count > _MAX_COMPRESSED_BYTES:
            raise RemoteCacheValidationError("Remote cache archive exceeds compressed size limit")
        return data


class _DigestingReader:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.count = 0

    def read(self, size: int = -1) -> bytes:
        data = self.stream.read(size)
        self.count += len(data)
        self.digest.update(data)
        return data


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    return info


def write_archive(stream: BinaryIO, entry: RemoteCacheEntry, source_paths: Mapping[str, Path]) -> None:
    """Write a complete entry as one deterministic archive."""
    if set(source_paths) != set(entry.artifact_names):
        raise RemoteCacheValidationError("Remote cache artifact set does not match entry")

    artifacts: dict[str, dict[str, int | str]] = {}
    total = 0
    counted = _CountingWriter(stream)
    try:
        with gzip.GzipFile(filename="", mode="wb", fileobj=counted, compresslevel=1, mtime=0) as zipped:
            with tarfile.open(mode="w|", fileobj=zipped) as archive:
                for name in entry.artifact_names:
                    path = Path(source_paths[name])
                    size = path.stat().st_size
                    total += size
                    if size > _MAX_ARTIFACT_BYTES or total > _MAX_EXTRACTED_BYTES:
                        raise RemoteCacheValidationError("Remote cache artifact exceeds size limit")
                    with path.open("rb") as source:
                        digesting = _DigestingReader(source)
                        archive.addfile(_tar_info(name, size), digesting)
                    if digesting.count != size:
                        raise RemoteCacheValidationError("Remote cache artifact changed during archive creation")
                    artifacts[name] = {"size": size, "sha256": digesting.digest.hexdigest()}

                manifest = json.dumps(
                    {
                        "archive_format_version": ARCHIVE_FORMAT_VERSION,
                        "canonical_identity": json.loads(entry.canonical_identity()),
                        "artifacts": artifacts,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                if len(manifest) > _MAX_MANIFEST_BYTES or total + len(manifest) > _MAX_EXTRACTED_BYTES:
                    raise RemoteCacheValidationError("Remote cache manifest exceeds size limit")
                archive.addfile(_tar_info("manifest.json", len(manifest)), io.BytesIO(manifest))
    except RemoteCacheValidationError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise RemoteCacheValidationError("Could not create remote cache archive") from exc


def read_archive(stream: BinaryIO, entry: RemoteCacheEntry, staging_dir: Path) -> None:
    """Validate an archive and place its known members in private staging."""
    expected = set(entry.artifact_names)
    observed: dict[str, dict[str, int | str]] = {}
    created: list[Path] = []
    total = 0
    manifest: dict | None = None
    counted = _CountingReader(stream)
    try:
        with gzip.GzipFile(fileobj=counted, mode="rb") as unzipped:
            with tarfile.open(fileobj=unzipped, mode="r|") as archive:
                for info in archive:
                    if not info.isfile() or info.name not in expected | {"manifest.json"}:
                        raise RemoteCacheValidationError("Remote cache archive contains an unexpected member")
                    if info.name in observed or manifest is not None:
                        raise RemoteCacheValidationError("Remote cache archive contains duplicate or trailing members")
                    limit = _MAX_MANIFEST_BYTES if info.name == "manifest.json" else _MAX_ARTIFACT_BYTES
                    total += info.size
                    if info.size > limit or total > _MAX_EXTRACTED_BYTES:
                        raise RemoteCacheValidationError("Remote cache archive member exceeds size limit")
                    member = archive.extractfile(info)
                    if member is None:
                        raise RemoteCacheValidationError("Remote cache archive member has no data")
                    if info.name == "manifest.json":
                        manifest = json.loads(member.read(limit + 1))
                        continue
                    path = Path(staging_dir) / info.name
                    digest = hashlib.sha256()
                    written = 0
                    with path.open("xb") as output:
                        created.append(path)
                        while chunk := member.read(_CHUNK_BYTES):
                            written += len(chunk)
                            if written > info.size:
                                raise RemoteCacheValidationError("Remote cache archive member changed size")
                            output.write(chunk)
                            digest.update(chunk)
                    if written != info.size:
                        raise RemoteCacheValidationError("Remote cache archive member is truncated")
                    observed[info.name] = {"size": written, "sha256": digest.hexdigest()}
            unzipped.read(1)  # Finish the gzip stream, including checksum and trailer.

        if set(observed) != expected or not isinstance(manifest, dict):
            raise RemoteCacheValidationError("Remote cache archive is incomplete")
        if manifest != {
            "archive_format_version": ARCHIVE_FORMAT_VERSION,
            "canonical_identity": json.loads(entry.canonical_identity()),
            "artifacts": observed,
        }:
            raise RemoteCacheValidationError("Remote cache archive manifest does not match entry")
    except (OSError, EOFError, ValueError, tarfile.TarError, RemoteCacheValidationError) as exc:
        for path in created:
            path.unlink(missing_ok=True)
        if isinstance(exc, RemoteCacheValidationError):
            raise
        raise RemoteCacheValidationError("Could not validate remote cache archive") from exc
