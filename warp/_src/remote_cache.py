# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Second-tier compilation cache that mirrors the local kernel cache.

A remote root has the layout of the directory named by ``WARP_CACHE_PATH``:
``<root>/<Warp version>/<module directory>/<file>``. Each cached file is one
remote object under its local name, so a copy of either tree is a valid
instance of the other.
"""

import os
import re
import shutil
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

import warp.config
from warp._src.logger import log_debug, log_warning


class RemoteStore(Protocol):
    """Storage backend for one remote cache root.

    Keys are POSIX-style paths relative to the root. An object must become
    visible to readers only once it is complete.
    """

    def get(self, key: str, path: Path) -> bool:
        """Download ``key`` to ``path`` and return whether the object exists."""
        ...

    def put(self, path: Path, key: str) -> None:
        """Upload the file at ``path`` as ``key`` unless that object already exists."""
        ...


class DirectoryRemoteStore:
    """Store objects as files below a directory, such as a network mount."""

    def __init__(self, root: str | os.PathLike):
        root = os.path.abspath(root)
        if os.name == "nt":
            from warp._src.build import _add_long_path_prefix  # noqa: PLC0415

            root = _add_long_path_prefix(root)
        self.root = Path(root)

    def get(self, key: str, path: Path) -> bool:
        try:
            shutil.copyfile(self.root / key, path)
        except FileNotFoundError:
            return False
        return True

    def put(self, path: Path, key: str) -> None:
        target = self.root / key
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        # Readers must never observe a partial file, so copy under a private name first.
        partial = target.with_name(f".{uuid.uuid4().hex}.partial")
        try:
            shutil.copyfile(path, partial)
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)


def _create_gcs_store(root: str) -> RemoteStore:
    try:
        from warp._src.remote_cache_gcs import GCSRemoteStore  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            'gs:// remote caches need the optional Google Cloud Storage package: pip install "warp-lang[remote-cache-gcs]"'
        ) from exc
    return GCSRemoteStore(root)


# Backends for URL roots, keyed by scheme. A root without a scheme is a directory.
_URL_STORES: dict[str, Callable[[str], RemoteStore]] = {"gs": _create_gcs_store}

_root: str | None = None
_version = ""
_read_only = False
_min_compile_time = 0.0
_fetch_enabled = True
_publish_enabled = True
_store: RemoteStore | None = None
_store_pid: int | None = None


def _is_shareable_version(version: str) -> bool:
    """Whether ``version`` names one published build.

    Releases (``1.19.0``) and date-stamped nightly builds
    (``1.20.0.dev20261118``) qualify. A source checkout reports ``.dev0``
    whatever it contains, so its artifacts are not safe to share.
    """
    return re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d{8})?", version) is not None


def _resolve_root(root: str | os.PathLike) -> str:
    """Return a validated remote root: a supported URL or an absolute directory."""
    root = os.fspath(root)
    if "://" not in root:
        return os.path.abspath(root)
    parsed = urlsplit(root)
    if parsed.scheme not in _URL_STORES:
        raise ValueError(f"'{root}' is not a directory or a URL with a supported scheme")
    if not parsed.netloc:
        raise ValueError(f"'{root}' does not name a bucket")
    return root.rstrip("/")


def init_remote_cache() -> None:
    """Snapshot the remote cache settings without contacting the remote store."""
    global _root, _version, _read_only, _min_compile_time, _fetch_enabled, _publish_enabled, _store, _store_pid
    _root = None
    _store = None
    _store_pid = None
    _fetch_enabled = True
    _publish_enabled = True
    if warp.config.remote_cache_dir is None:
        return
    if not _is_shareable_version(warp.config.version):
        log_warning(
            f"Remote cache disabled: Warp {warp.config.version} is not a published build, "
            "so its compiled kernels are not safe to share."
        )
        return
    try:
        _root = _resolve_root(warp.config.remote_cache_dir)
    except (TypeError, ValueError) as exc:
        log_warning(f"Remote cache disabled: {exc}")
        return
    _version = warp.config.version
    _read_only = warp.config.remote_cache_read_only
    _min_compile_time = warp.config.remote_cache_min_compile_time


def is_enabled() -> bool:
    """Whether a usable remote cache root was configured at initialization."""
    return _root is not None


def _create_store(root: str) -> RemoteStore:
    scheme, separator, _ = root.partition("://")
    if not separator:
        return DirectoryRemoteStore(root)
    return _URL_STORES[scheme](root)


def _get_store() -> RemoteStore:
    """Return this process's store, creating it on first use and after a fork."""
    global _store, _store_pid
    pid = os.getpid()
    if _store is None or _store_pid != pid:
        _store = _create_store(_root)
        _store_pid = pid
    return _store


def _key(entry_dir: str, name: str) -> str:
    return f"{_version}/{entry_dir}/{name}"


def fetch(entry_dir: str, names: Sequence[str], staging_dir: Path) -> bool:
    """Download every file of one cache entry into a private staging directory.

    Args:
        entry_dir: Name of the entry's directory in the local cache.
        names: Names of the files that make up the entry.
        staging_dir: Existing directory that receives the files.

    Returns:
        ``True`` if the remote cache holds every file. A miss or a failed
        transfer returns ``False`` and may leave some files behind.
    """
    global _fetch_enabled, _publish_enabled
    if _root is None or not _fetch_enabled:
        return False
    try:
        store = _get_store()
        for name in names:
            if not store.get(_key(entry_dir, name), Path(staging_dir) / name):
                return False
    except Exception as exc:
        # A store that cannot be read rarely recovers within one process, and
        # each further attempt could wait for another network timeout.
        _fetch_enabled = False
        _publish_enabled = False
        log_warning(f"Remote cache disabled for this process after a failed read from '{_root}': {exc}")
        return False
    log_debug(f"Restored '{entry_dir}' from the remote cache")
    return True


def publish(entry_dir: str, paths: Sequence[Path], compile_seconds: float) -> None:
    """Upload the committed local files of one cache entry.

    Args:
        entry_dir: Name of the entry's directory in the local cache.
        paths: Local files that make up the entry.
        compile_seconds: Time the compilation took, compared against
            :attr:`warp.config.remote_cache_min_compile_time`.
    """
    global _publish_enabled
    if _root is None or _read_only or not _publish_enabled or compile_seconds < _min_compile_time:
        return
    try:
        store = _get_store()
        for path in paths:
            store.put(Path(path), _key(entry_dir, Path(path).name))
    except Exception as exc:
        _publish_enabled = False
        log_warning(
            f"Remote cache publishing disabled for this process after a failed write to '{_root}': {exc}. "
            "Set warp.config.remote_cache_read_only if this job should not publish."
        )
        return
    log_debug(f"Published '{entry_dir}' to the remote cache")
