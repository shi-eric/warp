# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import io
import json
import tarfile
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import warp as wp
import warp.config
from warp._src import context, remote_cache
from warp._src.remote_cache import RemoteCacheEntry, RemoteCacheValidationError, read_archive, write_archive


class MemoryRemoteStore:
    def __init__(self):
        self.objects = {}
        self.lock = threading.Lock()
        self.writes = 0
        self.reads = 0

    @contextmanager
    def open_reader(self, uri):
        self.reads += 1
        with self.lock:
            data = self.objects.get(uri)
        if data is None:
            raise FileNotFoundError(uri)
        yield io.BytesIO(data)

    def put_create(self, uri, archive_path):
        with self.lock:
            if uri in self.objects:
                raise remote_cache.RemoteEntryExists(uri)
            self.objects[uri] = Path(archive_path).read_bytes()
            self.writes += 1


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

    def test_final_release_eligibility(self):
        self.assertTrue(remote_cache._is_final_release_version("1.19.0"))
        self.assertTrue(remote_cache._is_final_release_version("12.3.456"))
        for version in ("1.19", "1.19.0.dev0", "1.19.0rc1", "1.19.0.post1", "1.19.0+local", "bad"):
            with self.subTest(version=version):
                self.assertFalse(remote_cache._is_final_release_version(version))

    def test_disabled_and_invalid_policy_never_constructs_store(self):
        self.addCleanup(remote_cache.init_remote_cache)
        with patch.object(remote_cache, "_create_store") as create_store:
            remote_cache.init_remote_cache()
            self.assertFalse(
                remote_cache.download_entry(RemoteCacheEntry("kernel", "m", {"x": 1}, ("x.o",)), Path("."))
            )
            create_store.assert_not_called()
        with patch.object(warp.config, "remote_cache_dir", "file:///tmp/bad"):
            remote_cache.init_remote_cache()
            with patch.object(remote_cache, "_create_store") as create_store:
                self.assertFalse(
                    remote_cache.download_entry(RemoteCacheEntry("kernel", "m", {"x": 1}, ("x.o",)), Path("."))
                )
                create_store.assert_not_called()
        with patch.object(warp.config, "remote_cache_dir", "gs://bucket/cache"):
            remote_cache.init_remote_cache()
            with patch.object(remote_cache, "_create_store") as create_store:
                self.assertFalse(
                    remote_cache.download_entry(RemoteCacheEntry("kernel", "m", {"x": 1}, ("x.o",)), Path("."))
                )
                create_store.assert_not_called()

    def test_policy_publish_restore_threshold_and_read_only(self):
        self.addCleanup(remote_cache.init_remote_cache)
        store = MemoryRemoteStore()
        entry = RemoteCacheEntry("kernel", "module", {"x": 1}, ("binary.o", "binary.meta"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = {name: root / name for name in entry.artifact_names}
            for name, path in sources.items():
                path.write_bytes(name.encode())
            with (
                patch.object(warp.config, "remote_cache_dir", "gs://bucket/cache"),
                patch.object(remote_cache, "_is_final_release_version", return_value=True),
                patch.object(remote_cache, "_create_store", return_value=store),
            ):
                remote_cache.init_remote_cache()
                remote_cache.publish_entry(entry, sources, 0.99)
                self.assertEqual(store.writes, 0)
                remote_cache.publish_entry(entry, sources, 1.0)
                self.assertEqual(store.writes, 1)
                with tempfile.TemporaryDirectory() as staging:
                    self.assertTrue(remote_cache.download_entry(entry, Path(staging)))
                    for name, source in sources.items():
                        self.assertEqual((Path(staging) / name).read_bytes(), source.read_bytes())
                self.assertEqual(store.reads, 1)
            with (
                patch.object(warp.config, "remote_cache_dir", "gs://bucket/cache"),
                patch.object(warp.config, "remote_cache_read_only", True),
                patch.object(remote_cache, "_is_final_release_version", return_value=True),
                patch.object(remote_cache, "_create_store", return_value=store),
            ):
                remote_cache.init_remote_cache()
                remote_cache.publish_entry(RemoteCacheEntry("kernel", "other", {"x": 2}, ("binary.o",)), sources, 10.0)
                self.assertEqual(store.writes, 1)
                with tempfile.TemporaryDirectory() as staging:
                    self.assertTrue(remote_cache.download_entry(entry, Path(staging)))

    def test_policy_corrupt_or_failed_archive_keeps_local_files(self):
        self.addCleanup(remote_cache.init_remote_cache)
        store = MemoryRemoteStore()
        entry = RemoteCacheEntry("kernel", "module", {"x": 1}, ("binary.o",))
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "binary.o"
            source.write_bytes(b"local")
            uri = entry.object_uri("gs://bucket/cache")
            store.objects[uri] = b"bad archive"
            with (
                patch.object(warp.config, "remote_cache_dir", "gs://bucket/cache"),
                patch.object(remote_cache, "_is_final_release_version", return_value=True),
                patch.object(remote_cache, "_create_store", return_value=store),
            ):
                remote_cache.init_remote_cache()
                staging = Path(tmp) / "staging"
                staging.mkdir()
                self.assertFalse(remote_cache.download_entry(entry, staging))
                self.assertEqual(store.objects[uri], b"bad archive")
                remote_cache.publish_entry(entry, {"binary.o": Path(tmp) / "missing"}, 2.0)
                self.assertEqual(store.writes, 0)
                self.assertEqual(source.read_bytes(), b"local")

    def test_policy_store_is_process_local(self):
        self.addCleanup(remote_cache.init_remote_cache)
        with (
            patch.object(warp.config, "remote_cache_dir", "gs://bucket/cache"),
            patch.object(remote_cache, "_is_final_release_version", return_value=True),
            patch.object(
                remote_cache, "_create_store", side_effect=[MemoryRemoteStore(), MemoryRemoteStore()]
            ) as factory,
            patch.object(remote_cache.os, "getpid", side_effect=[101, 101, 102]),
        ):
            remote_cache.init_remote_cache()
            first = remote_cache._get_store()
            self.assertIs(first, remote_cache._get_store())
            self.assertIsNot(first, remote_cache._get_store())
            self.assertEqual(factory.call_count, 2)

    def test_policy_missing_and_transfer_failures_fall_back(self):
        self.addCleanup(remote_cache.init_remote_cache)
        store = MemoryRemoteStore()
        entry = RemoteCacheEntry("kernel", "module", {"x": 1}, ("binary.o",))
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "binary.o"
            source.write_bytes(b"local")
            with (
                patch.object(warp.config, "remote_cache_dir", "gs://bucket/cache"),
                patch.object(remote_cache, "_is_final_release_version", return_value=True),
                patch.object(remote_cache, "_create_store", return_value=store),
                patch.object(remote_cache, "log_warning") as warn,
            ):
                remote_cache.init_remote_cache()
                self.assertFalse(remote_cache.download_entry(entry, Path(tmp)))
                warn.assert_not_called()
                with patch.object(store, "open_reader", side_effect=TimeoutError("network down")):
                    self.assertFalse(remote_cache.download_entry(entry, Path(tmp)))
                    self.assertFalse(remote_cache.download_entry(entry, Path(tmp)))
                self.assertEqual(warn.call_count, 1)
                with patch.object(store, "put_create", side_effect=TimeoutError("network down")):
                    remote_cache.publish_entry(entry, {"binary.o": source}, 2.0)
                self.assertEqual(source.read_bytes(), b"local")
                self.assertEqual(warn.call_count, 2)

    def test_cpu_target_identity_dimensions(self):
        runtime = Mock()
        runtime.get_llvm_target_triple.return_value = "x86_64-pc-linux-gnu"
        with (
            patch.object(context, "runtime", runtime),
            patch.object(context, "_get_cpu_toolchain_version", return_value="22.1"),
            patch.object(context, "_get_host_cpu_name", return_value="znver5"),
            patch.object(context, "_get_cpu_feature_set", return_value=frozenset({"avx2", "sse2"})),
        ):
            baseline = context._get_cpu_remote_target_identity("-O2 -march=native")
            self.assertEqual(baseline["target_triple"], "x86_64-pc-linux-gnu")
            self.assertEqual(baseline["cpu_features"], ["avx2", "sse2"])
            self.assertNotEqual(baseline, context._get_cpu_remote_target_identity("-O3 -march=native"))
            runtime.get_llvm_target_triple.return_value = "aarch64-unknown-linux-gnu"
            self.assertNotEqual(baseline, context._get_cpu_remote_target_identity("-O2 -march=native"))
            runtime.get_llvm_target_triple.return_value = "x86_64-pc-linux-gnu"
            with patch.object(context, "_get_cpu_toolchain_version", return_value="23.0"):
                self.assertNotEqual(baseline, context._get_cpu_remote_target_identity("-O2 -march=native"))
            with patch.object(context, "_get_host_cpu_name", return_value="skylake"):
                self.assertNotEqual(baseline, context._get_cpu_remote_target_identity("-O2 -march=native"))
            with patch.object(context, "_get_cpu_feature_set", return_value=frozenset({"avx2"})):
                self.assertNotEqual(baseline, context._get_cpu_remote_target_identity("-O2 -march=native"))
            portable = context._get_cpu_remote_target_identity("-O2 -march=x86-64")
            with patch.object(context, "_get_host_cpu_name", return_value="skylake"):
                self.assertEqual(portable, context._get_cpu_remote_target_identity("-O2 -march=x86-64"))

    def test_cuda_target_identity_dimensions(self):
        runtime = Mock()
        runtime.toolkit_version = (13, 4)
        runtime.get_nvrtc_version.return_value = (13, 4)
        runtime.get_llvm_version.return_value = "22.1"
        with patch.object(context, "runtime", runtime):
            baseline = context._get_cuda_remote_target_identity(90, "a", False, False)
            self.assertEqual(baseline["compiler"], "nvrtc")
            self.assertNotIn("driver", repr(baseline))
            for args in ((90, "a", True, False), (89, "a", False, False), (90, "", False, False)):
                self.assertNotEqual(baseline, context._get_cuda_remote_target_identity(*args))
            self.assertNotEqual(baseline, context._get_cuda_remote_target_identity(90, "a", False, True))
            runtime.get_nvrtc_version.return_value = (13, 5)
            self.assertNotEqual(baseline, context._get_cuda_remote_target_identity(90, "a", False, False))
            runtime.get_nvrtc_version.return_value = (13, 4)
            runtime.toolkit_version = (13, 5)
            self.assertNotEqual(baseline, context._get_cuda_remote_target_identity(90, "a", False, False))

    def test_cpu_target_triple_comes_from_native_compiler(self):
        wp.init()
        triple = context.runtime.get_llvm_target_triple()
        self.assertIsInstance(triple, str)
        self.assertIn("-", triple)


if __name__ == "__main__":
    unittest.main(verbosity=2)
