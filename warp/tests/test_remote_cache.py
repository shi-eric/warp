# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import warp.config
from warp._src import remote_cache
from warp._src.remote_cache import RemoteCacheEntry, RemoteCacheValidationError, read_archive, write_archive


class TestRemoteCache(unittest.TestCase):
    def test_config_defaults(self):
        self.assertIsNone(warp.config.remote_cache_dir)
        self.assertFalse(warp.config.remote_cache_read_only)
        self.assertEqual(warp.config.remote_cache_min_compile_time, 1.0)

    def test_entry_key_is_canonical_and_full_length(self):
        identity = {"target": {"kind": "cpu", "features": ["avx2", "sse2"]}, "module_hash": "ab" * 32}
        reordered = {"module_hash": "ab" * 32, "target": {"features": ["avx2", "sse2"], "kind": "cpu"}}
        names = ("wp_example_1234567.cpu12345678.o", "wp_example_1234567.meta")
        entry = RemoteCacheEntry("kernel", "wp_example_1234567", identity, names)
        equivalent = RemoteCacheEntry("kernel", "wp_example_1234567", reordered, names)
        self.assertEqual(entry.digest(), equivalent.digest())
        self.assertRegex(entry.digest(), r"^[0-9a-f]{64}$")
        self.assertEqual(
            entry.object_uri("gs://cache-bucket/warp/"),
            f"gs://cache-bucket/warp/{warp.config.version}/wp_example_1234567/{entry.digest()}.tar.gz",
        )
        identity["target"]["features"].append("avx512f")
        self.assertEqual(entry.digest(), equivalent.digest())

    def test_entry_identity_dimensions(self):
        identity = {"module_hash": "ab" * 32}
        names = ("binary.o", "binary.meta")
        base = RemoteCacheEntry("kernel", "module", identity, names)
        variants = (
            RemoteCacheEntry("lto", "module", identity, names),
            RemoteCacheEntry("kernel", "other", identity, names),
            RemoteCacheEntry("kernel", "module", {"module_hash": "cd" * 32}, names),
            RemoteCacheEntry("kernel", "module", identity, ("binary.o",)),
        )
        for variant in variants:
            self.assertNotEqual(base.digest(), variant.digest())
        with patch.object(warp.config, "version", "99.0.0"):
            self.assertNotEqual(base.digest(), RemoteCacheEntry("kernel", "module", identity, names).digest())

    def test_entry_rejects_invalid_names_and_values(self):
        for name in ("../bad", "/absolute", "nested/name", "", "manifest.json"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                RemoteCacheEntry("kernel", "module", {"x": 1}, (name,))
        with self.assertRaises(ValueError):
            RemoteCacheEntry("kernel", "module", {"x": 1}, ("a.o", "a.o"))
        for invalid in (float("nan"), {1: "bad"}, object()):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                RemoteCacheEntry("kernel", "module", {"x": copy.copy(invalid)}, ("a.o",))

    def test_archive_round_trip(self):
        for kind, names in (
            ("kernel", ("binary.o", "binary.meta")),
            ("lto", ("symbol.lto", "symbol.meta", "symbol_fatbin.lto")),
        ):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                entry = RemoteCacheEntry(kind, "module", {"symbol": "abc"}, names)
                sources = {}
                for name in names:
                    path = root / name
                    path.write_bytes(name.encode() + b"\x00\xff")
                    sources[name] = path

                stream = io.BytesIO()
                write_archive(stream, entry, sources)
                archive_bytes = stream.getvalue()
                self.assertEqual(archive_bytes[:2], b"\x1f\x8b")
                with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
                    self.assertEqual(archive.getnames(), [*names, "manifest.json"])
                    self.assertTrue(all(info.mtime == 0 and info.uid == 0 and info.gid == 0 for info in archive))
                    manifest = json.load(archive.extractfile("manifest.json"))
                    self.assertEqual(manifest["canonical_identity"], json.loads(entry.canonical_identity()))
                    for name in names:
                        self.assertEqual(manifest["artifacts"][name]["size"], sources[name].stat().st_size)
                        self.assertEqual(
                            manifest["artifacts"][name]["sha256"],
                            hashlib.sha256(sources[name].read_bytes()).hexdigest(),
                        )

                restored = root / "restored"
                restored.mkdir()
                read_archive(io.BytesIO(archive_bytes), entry, restored)
                for name in names:
                    self.assertEqual((restored / name).read_bytes(), sources[name].read_bytes())

    def test_archive_rejects_unsafe_or_incomplete_members(self):
        entry = RemoteCacheEntry("kernel", "module", {"x": 1}, ("binary.o", "binary.meta"))
        payload = {"binary.o": b"binary", "binary.meta": b"metadata"}

        def make_archive(members, manifest_override=None):
            manifest = {
                "archive_format_version": 1,
                "canonical_identity": json.loads(entry.canonical_identity()),
                "artifacts": {
                    name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                    for name, data in payload.items()
                },
            }
            if manifest_override:
                manifest_override(manifest)
            output = io.BytesIO()
            with tarfile.open(fileobj=output, mode="w:gz") as archive:
                for name, data, member_type in members:
                    info = tarfile.TarInfo(name)
                    info.type = member_type
                    info.size = len(data) if member_type == tarfile.REGTYPE else 0
                    archive.addfile(info, io.BytesIO(data) if member_type == tarfile.REGTYPE else None)
                encoded = json.dumps(manifest).encode()
                info = tarfile.TarInfo("manifest.json")
                info.size = len(encoded)
                archive.addfile(info, io.BytesIO(encoded))
            return output.getvalue()

        good = [(name, data, tarfile.REGTYPE) for name, data in payload.items()]
        bad_cases = (
            good[:1],
            [*good, ("surprise", b"x", tarfile.REGTYPE)],
            [*good, good[0]],
            [("../binary.o", b"binary", tarfile.REGTYPE), good[1]],
            [("nested/binary.o", b"binary", tarfile.REGTYPE), good[1]],
            [("/binary.o", b"binary", tarfile.REGTYPE), good[1]],
            [("binary.o", b"", tarfile.SYMTYPE), good[1]],
            [("binary.o", b"", tarfile.LNKTYPE), good[1]],
            [("binary.o", b"", tarfile.DIRTYPE), good[1]],
            [("binary.o", b"", tarfile.FIFOTYPE), good[1]],
        )
        with tempfile.TemporaryDirectory() as tmp:
            for members in bad_cases:
                with self.subTest(members=members), self.assertRaises(RemoteCacheValidationError):
                    read_archive(io.BytesIO(make_archive(members)), entry, Path(tmp))
            with self.assertRaises(RemoteCacheValidationError):
                read_archive(
                    io.BytesIO(
                        make_archive(good, lambda manifest: manifest["canonical_identity"].update({"kind": "lto"}))
                    ),
                    entry,
                    Path(tmp),
                )
            with self.assertRaises(RemoteCacheValidationError):
                read_archive(
                    io.BytesIO(
                        make_archive(
                            good, lambda manifest: manifest["artifacts"]["binary.o"].update({"sha256": "0" * 64})
                        )
                    ),
                    entry,
                    Path(tmp),
                )
            with self.assertRaises(RemoteCacheValidationError):
                read_archive(io.BytesIO(make_archive(good)[:-9]), entry, Path(tmp))

    def test_archive_limits(self):
        entry = remote_cache.RemoteCacheEntry("kernel", "module", {"x": 1}, ("binary.o",))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "binary.o"
            path.write_bytes(b"a" * 40)
            with (
                patch.object(remote_cache, "_MAX_ARTIFACT_BYTES", 16),
                self.assertRaises(remote_cache.RemoteCacheValidationError),
            ):
                remote_cache.write_archive(io.BytesIO(), entry, {"binary.o": path})
            stream = io.BytesIO()
            remote_cache.write_archive(stream, entry, {"binary.o": path})
            staging = Path(tmp) / "staging"
            staging.mkdir()
            with (
                patch.object(remote_cache, "_MAX_COMPRESSED_BYTES", 8),
                self.assertRaises(remote_cache.RemoteCacheValidationError),
            ):
                remote_cache.read_archive(io.BytesIO(stream.getvalue()), entry, staging)


if __name__ == "__main__":
    unittest.main(verbosity=2)
