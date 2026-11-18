# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import warp as wp
import warp.config
from warp._src import context, remote_cache
from warp._src.remote_cache import DirectoryRemoteStore


@wp.kernel
def _remote_cache_kernel(values: wp.array[int]):
    i = wp.tid()
    values[i] += 1


class FailingStore:
    """Store whose transfers always fail, counting how often Warp contacts it."""

    def __init__(self):
        self.calls = 0

    def get(self, key, path):
        self.calls += 1
        raise TimeoutError("network down")

    def put(self, path, key):
        self.calls += 1
        raise TimeoutError("network down")


class UnwritableStore(DirectoryRemoteStore):
    """Readable directory whose uploads are rejected, like a read-only bucket."""

    def __init__(self, root):
        super().__init__(root)
        self.put_calls = 0

    def put(self, path, key):
        self.put_calls += 1
        raise PermissionError("bucket is read-only")


def _files(root: Path) -> dict[str, bytes]:
    """Return every file under ``root`` keyed by its POSIX-style relative path."""
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@contextmanager
def _remote_cache(root, *, read_only=False, min_compile_time=0.0):
    """Enable the remote cache in a development checkout, then disable it again."""
    try:
        with (
            patch.object(warp.config, "remote_cache_dir", os.fspath(root)),
            patch.object(warp.config, "remote_cache_read_only", read_only),
            patch.object(warp.config, "remote_cache_min_compile_time", min_compile_time),
            patch.object(remote_cache, "_is_shareable_version", return_value=True),
        ):
            remote_cache.init_remote_cache()
            yield
    finally:
        remote_cache.init_remote_cache()


class TestRemoteCache(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.remote = self.tmp / "remote"
        self.remote.mkdir()
        self.staging = self.tmp / "staging"
        self.staging.mkdir()
        # Every remote object lives below the Warp version, as local cache files do.
        self.version_dir = self.remote / warp.config.version

    def _use_local_cache(self, name: str) -> Path:
        """Point Warp at an empty local kernel cache for the rest of the test."""
        self.addCleanup(setattr, wp.config, "kernel_cache_dir", wp.config.kernel_cache_dir)
        path = self.tmp / name
        path.mkdir()
        wp.config.kernel_cache_dir = str(path)
        return path

    def test_publish_mirrors_local_cache_layout(self):
        module_dir = self.tmp / "local" / "wp_mod_abc1234"
        binary = _write(module_dir / "wp_mod_abc1234.sm86.cubin", b"binary")
        meta = _write(module_dir / "wp_mod_abc1234.meta", b"meta")
        with _remote_cache(self.remote):
            remote_cache.publish("wp_mod_abc1234", [binary, meta], 5.0)
        self.assertEqual(
            _files(self.remote),
            {
                f"{warp.config.version}/wp_mod_abc1234/wp_mod_abc1234.sm86.cubin": b"binary",
                f"{warp.config.version}/wp_mod_abc1234/wp_mod_abc1234.meta": b"meta",
            },
        )

    def test_fetch_restores_every_requested_file(self):
        _write(self.version_dir / "wp_mod_abc1234" / "wp_mod_abc1234.o", b"binary")
        _write(self.version_dir / "wp_mod_abc1234" / "wp_mod_abc1234.meta", b"meta")
        with _remote_cache(self.remote):
            hit = remote_cache.fetch("wp_mod_abc1234", ["wp_mod_abc1234.o", "wp_mod_abc1234.meta"], self.staging)
        self.assertTrue(hit)
        self.assertEqual(_files(self.staging), {"wp_mod_abc1234.o": b"binary", "wp_mod_abc1234.meta": b"meta"})

    def test_fetch_misses_when_any_file_is_absent(self):
        _write(self.version_dir / "binary_only" / "binary_only.o", b"binary")
        _write(self.version_dir / "meta_only" / "meta_only.meta", b"meta")
        with _remote_cache(self.remote):
            for name in ("binary_only", "meta_only"):
                with self.subTest(entry=name):
                    self.assertFalse(remote_cache.fetch(name, [f"{name}.o", f"{name}.meta"], self.staging))

    def test_missing_entry_is_a_quiet_miss_that_is_retried(self):
        with _remote_cache(self.remote), patch.object(remote_cache, "log_warning") as warn:
            self.assertFalse(remote_cache.fetch("wp_mod_abc1234", ["wp_mod_abc1234.o"], self.staging))
            warn.assert_not_called()
            # A later lookup still reaches the store, so a miss does not disable the cache.
            _write(self.version_dir / "wp_mod_abc1234" / "wp_mod_abc1234.o", b"binary")
            self.assertTrue(remote_cache.fetch("wp_mod_abc1234", ["wp_mod_abc1234.o"], self.staging))

    def test_publish_skips_compilations_below_threshold(self):
        binary = _write(self.tmp / "local" / "wp_mod_abc1234.o", b"binary")
        with _remote_cache(self.remote, min_compile_time=1.0):
            remote_cache.publish("wp_mod_abc1234", [binary], 0.99)
            self.assertEqual(_files(self.remote), {})
            remote_cache.publish("wp_mod_abc1234", [binary], 1.0)
            self.assertEqual(_files(self.remote), {f"{warp.config.version}/wp_mod_abc1234/wp_mod_abc1234.o": b"binary"})

    def test_read_only_fetches_without_publishing(self):
        _write(self.version_dir / "wp_mod_abc1234" / "wp_mod_abc1234.o", b"binary")
        local = _write(self.tmp / "local" / "wp_other_def5678.o", b"other")
        before = _files(self.remote)
        with _remote_cache(self.remote, read_only=True):
            remote_cache.publish("wp_other_def5678", [local], 10.0)
            self.assertTrue(remote_cache.fetch("wp_mod_abc1234", ["wp_mod_abc1234.o"], self.staging))
        self.assertEqual(_files(self.remote), before)

    def test_publish_keeps_existing_remote_file(self):
        published = _write(self.version_dir / "wp_mod_abc1234" / "wp_mod_abc1234.o", b"first")
        local = _write(self.tmp / "local" / "wp_mod_abc1234.o", b"second")
        with _remote_cache(self.remote):
            remote_cache.publish("wp_mod_abc1234", [local], 5.0)
        self.assertEqual(_files(self.remote), {published.relative_to(self.remote).as_posix(): b"first"})

    def test_fetch_error_warns_once_and_stops_contacting_remote(self):
        store = FailingStore()
        local = _write(self.tmp / "local" / "wp_mod_abc1234.o", b"binary")
        with (
            _remote_cache(self.remote),
            patch.object(remote_cache, "_create_store", return_value=store),
            patch.object(remote_cache, "log_warning") as warn,
        ):
            self.assertFalse(remote_cache.fetch("wp_mod_abc1234", ["wp_mod_abc1234.o"], self.staging))
            self.assertFalse(remote_cache.fetch("wp_mod_abc1234", ["wp_mod_abc1234.o"], self.staging))
            remote_cache.publish("wp_mod_abc1234", [local], 5.0)
        self.assertEqual(store.calls, 1)
        self.assertEqual(warn.call_count, 1)

    def test_publish_error_stops_publishing_but_keeps_fetching(self):
        _write(self.version_dir / "wp_mod_abc1234" / "wp_mod_abc1234.o", b"binary")
        local = _write(self.tmp / "local" / "wp_other_def5678.o", b"other")
        store = UnwritableStore(self.remote)
        with (
            _remote_cache(self.remote),
            patch.object(remote_cache, "_create_store", return_value=store),
            patch.object(remote_cache, "log_warning") as warn,
        ):
            remote_cache.publish("wp_other_def5678", [local], 5.0)
            remote_cache.publish("wp_other_def5678", [local], 5.0)
            self.assertTrue(remote_cache.fetch("wp_mod_abc1234", ["wp_mod_abc1234.o"], self.staging))
        self.assertEqual(store.put_calls, 1)
        self.assertEqual(warn.call_count, 1)
        self.assertEqual(local.read_bytes(), b"other")

    def test_store_is_recreated_in_another_process(self):
        with (
            _remote_cache(self.remote),
            patch.object(remote_cache, "_create_store", side_effect=DirectoryRemoteStore) as create_store,
            patch.object(remote_cache.os, "getpid", return_value=101) as getpid,
        ):
            remote_cache.fetch("wp_mod_abc1234", ["wp_mod_abc1234.o"], self.staging)
            remote_cache.fetch("wp_mod_abc1234", ["wp_mod_abc1234.o"], self.staging)
            self.assertEqual(create_store.call_count, 1)
            getpid.return_value = 102
            remote_cache.fetch("wp_mod_abc1234", ["wp_mod_abc1234.o"], self.staging)
            self.assertEqual(create_store.call_count, 2)

    def test_only_published_builds_share_artifacts(self):
        self.addCleanup(remote_cache.init_remote_cache)
        cases = {
            "1.19.0": True,
            "12.3.456": True,
            "1.20.0.dev20261118": True,
            "1.20.0.dev0": False,
            "1.20.0.dev1": False,
            "1.19": False,
            "1.19.0rc1": False,
            "1.19.0.post1": False,
            "1.19.0+local": False,
            "bad": False,
        }
        for version, expected in cases.items():
            with (
                self.subTest(version=version),
                patch.object(warp.config, "version", version),
                patch.object(warp.config, "remote_cache_dir", str(self.remote)),
                patch.object(remote_cache, "log_warning"),
            ):
                remote_cache.init_remote_cache()
                self.assertEqual(remote_cache.is_enabled(), expected)

    def test_unsupported_root_is_disabled_with_warning(self):
        self.addCleanup(remote_cache.init_remote_cache)
        for root in ("s3://bucket/cache", "gs://", "gs:///cache"):
            with (
                self.subTest(root=root),
                patch.object(warp.config, "remote_cache_dir", root),
                patch.object(remote_cache, "_is_shareable_version", return_value=True),
                patch.object(remote_cache, "_create_store") as create_store,
                patch.object(remote_cache, "log_warning") as warn,
            ):
                remote_cache.init_remote_cache()
                self.assertFalse(remote_cache.is_enabled())
                self.assertFalse(remote_cache.fetch("wp_mod_abc1234", ["wp_mod_abc1234.o"], self.staging))
                create_store.assert_not_called()
                self.assertEqual(warn.call_count, 1)

    def test_kernel_round_trip_skips_compilation(self):
        wp.init()
        module = wp.get_module(__name__)
        device = wp.get_device("cpu")
        module_dir = module.get_module_identifier()
        names = (module._get_compile_output_name(device), module._get_meta_name())
        producer = self._use_local_cache("producer")
        with _remote_cache(self.remote):
            self.assertTrue(module._compile(device=device))
            self.assertEqual(
                _files(self.remote),
                {
                    f"{warp.config.version}/{module_dir}/{name}": (producer / module_dir / name).read_bytes()
                    for name in names
                },
            )
            with patch.object(remote_cache, "fetch", wraps=remote_cache.fetch) as fetch:
                self.assertFalse(module._compile(device=device))
                fetch.assert_not_called()

            consumer = self._use_local_cache("consumer")
            with patch.object(module, "_run_codegen", side_effect=AssertionError("remote hit compiled")):
                self.assertFalse(module._compile(device=device))
            for name in names:
                self.assertEqual(
                    (consumer / module_dir / name).read_bytes(), (producer / module_dir / name).read_bytes()
                )

    def test_copy_of_remote_root_is_a_valid_local_cache(self):
        wp.init()
        module = wp.get_module(__name__)
        device = wp.get_device("cpu")
        self._use_local_cache("producer")
        with _remote_cache(self.remote):
            self.assertTrue(module._compile(device=device))
        wp.config.kernel_cache_dir = str(self.version_dir)
        with patch.object(module, "_run_codegen", side_effect=AssertionError("mirrored cache missed")):
            self.assertFalse(module._compile(device=device))

    def test_kernel_partial_remote_entry_compiles_and_completes_it(self):
        wp.init()
        module = wp.get_module(__name__)
        device = wp.get_device("cpu")
        self._use_local_cache("producer")
        with _remote_cache(self.remote):
            self.assertTrue(module._compile(device=device))
            complete = _files(self.remote)
            (self.version_dir / module.get_module_identifier() / module._get_meta_name()).unlink()
            self._use_local_cache("consumer")
            self.assertTrue(module._compile(device=device))
        self.assertEqual(sorted(_files(self.remote)), sorted(complete))

    def test_cuda_kernel_round_trip_restores_ptx_and_cubin(self):
        wp.init()
        if not wp.is_cuda_available():
            self.skipTest("CUDA device required")
        device = wp.get_device("cuda:0")
        module = wp.get_module(__name__)
        module_dir = module.get_module_identifier()
        producer = self._use_local_cache("producer")
        consumer = self.tmp / "consumer"
        consumer.mkdir()
        with _remote_cache(self.remote):
            for use_ptx in (True, False):
                with self.subTest(use_ptx=use_ptx):
                    if not use_ptx and device.arch not in context.runtime.nvrtc_supported_archs:
                        self.skipTest("NVRTC cannot emit a CUBIN for this device")
                    binary_name = module._get_compile_output_name(device, use_ptx=use_ptx)
                    wp.config.kernel_cache_dir = str(producer)
                    self.assertTrue(module._compile(device=device, use_ptx=use_ptx))
                    wp.config.kernel_cache_dir = str(consumer)
                    with patch.object(module, "_run_codegen", side_effect=AssertionError("remote hit compiled")):
                        self.assertFalse(module._compile(device=device, use_ptx=use_ptx))
                    self.assertEqual(
                        (consumer / module_dir / binary_name).read_bytes(),
                        (producer / module_dir / binary_name).read_bytes(),
                    )

    def test_kernel_remote_is_skipped_for_nonstandard_builds(self):
        wp.init()
        module = wp.get_module(__name__)
        device = wp.get_device("cpu")
        self._use_local_cache("producer")
        with _remote_cache(self.remote):
            self.assertTrue(module._compile(device=device))
            seeded = _files(self.remote)
            self._use_local_cache("consumer")
            custom_output = self.tmp / "custom-output"
            custom_output.mkdir()
            verify_options = module.resolve_options(warp.config) | {"verify_autograd_array_access": True}
            cases = {
                "custom output directory": (
                    patch.object(warp.config, "cache_kernels", True),
                    {"output_dir": custom_output},
                ),
                "autograd verification": (
                    patch.object(warp.config, "cache_kernels", True),
                    {"options": verify_options},
                ),
                "caching disabled": (patch.object(warp.config, "cache_kernels", False), {}),
                "stripped hash": (patch.dict(module.options, {"strip_hash": True}), {}),
            }
            for name, (setting, kwargs) in cases.items():
                with (
                    self.subTest(case=name),
                    setting,
                    patch.object(remote_cache, "fetch", wraps=remote_cache.fetch) as fetch,
                    patch.object(remote_cache, "publish", wraps=remote_cache.publish) as publish,
                ):
                    self.assertTrue(module._compile(device=device, **kwargs))
                    fetch.assert_not_called()
                    publish.assert_not_called()
            self.assertEqual(_files(self.remote), seeded)

    def _run_helper(self, mode: str, local_cache: Path) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env["WARP_CACHE_PATH"] = str(local_cache)
        helper = Path(__file__).with_name("aux_test_remote_cache.py")
        return subprocess.run(
            [sys.executable, str(helper), mode, str(self.remote)],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=120,
        )

    def test_kernel_cache_crosses_processes(self):
        producer = self._run_helper("producer", self.tmp / "producer")
        self.assertEqual(producer.returncode, 0, producer.stdout)
        consumer = self._run_helper("consumer", self.tmp / "consumer")
        self.assertEqual(consumer.returncode, 0, consumer.stdout)

    def test_kernel_remote_commit_into_empty_destination(self):
        staging = self.staging
        (staging / "binary.o").write_bytes(b"binary")
        (staging / "binary.meta").write_bytes(b"meta")
        destination = self.tmp / "module"
        context._commit_remote_kernel_entry(staging, destination, "binary.o", "binary.meta")
        self.assertEqual((destination / "binary.o").read_bytes(), b"binary")
        self.assertEqual((destination / "binary.meta").read_bytes(), b"meta")

    def test_kernel_remote_commit_preserves_complete_race_winner(self):
        staging = self.staging
        (staging / "binary.o").write_bytes(b"remote binary")
        (staging / "binary.meta").write_bytes(b"remote meta")
        destination = self.tmp / "module"
        destination.mkdir()
        (destination / "binary.o").write_bytes(b"local binary")
        (destination / "binary.meta").write_bytes(b"local meta")
        context._commit_remote_kernel_entry(staging, destination, "binary.o", "binary.meta")
        self.assertEqual((destination / "binary.o").read_bytes(), b"local binary")
        self.assertEqual((destination / "binary.meta").read_bytes(), b"local meta")

    def test_two_downloaders_commit_one_complete_pair(self):
        destination = self.tmp / "module"
        gate = threading.Barrier(2)

        def commit(index):
            staging = self.tmp / f"staging-{index}"
            staging.mkdir()
            (staging / "binary.o").write_bytes(f"binary-{index}".encode())
            (staging / "binary.meta").write_bytes(f"meta-{index}".encode())
            gate.wait(timeout=5)
            context._commit_remote_kernel_entry(staging, destination, "binary.o", "binary.meta")

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(commit, index) for index in range(2)]
            for future in futures:
                future.result(timeout=10)
        winner = (destination / "binary.o").read_text().removeprefix("binary-")
        self.assertIn(winner, {"0", "1"})
        self.assertEqual((destination / "binary.meta").read_text(), f"meta-{winner}")

    def test_kernel_remote_commit_does_not_publish_meta_after_binary_failure(self):
        staging = self.staging
        (staging / "binary.o").write_bytes(b"remote binary")
        (staging / "binary.meta").write_bytes(b"remote meta")
        destination = self.tmp / "module"
        destination.mkdir()
        (destination / "placeholder").write_bytes(b"keep")
        with patch.object(context.os, "replace", side_effect=OSError("move failed")):
            context._commit_remote_kernel_entry(staging, destination, "binary.o", "binary.meta")
        self.assertFalse((destination / "binary.meta").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
