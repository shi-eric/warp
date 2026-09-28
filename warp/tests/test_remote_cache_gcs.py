# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Google Cloud Storage adapter tests against fake-gcs-server."""

import io
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

try:
    from google.api_core.exceptions import PreconditionFailed
    from google.auth.credentials import AnonymousCredentials
    from google.cloud import storage
except ImportError:
    PreconditionFailed = None
    AnonymousCredentials = None
    storage = None

from warp._src import remote_cache


class TestRemoteCacheGCS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("STORAGE_EMULATOR_HOST"):
            raise unittest.SkipTest("STORAGE_EMULATOR_HOST is required")
        if storage is None:
            raise unittest.SkipTest("Install the remote-cache-gcs extra")
        from warp._src.remote_cache_gcs import GCSRemoteStore  # noqa: PLC0415

        cls.client = storage.Client(credentials=AnonymousCredentials(), project="test")
        cls.bucket = cls.client.create_bucket(f"warp-cache-test-{uuid.uuid4().hex}")
        cls.store = GCSRemoteStore(client=cls.client)

    def _uri(self, name):
        return f"gs://{self.bucket.name}/{uuid.uuid4().hex}/{name}"

    def test_missing_object_is_a_miss(self):
        with self.assertRaises(FileNotFoundError), self.store.open_reader(self._uri("missing")):
            pass

    def test_create_only_upload_for_multipart_and_resumable(self):
        for size in (1 << 20, 9 << 20):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as tmp:
                first = Path(tmp) / "first.tar.gz"
                second = Path(tmp) / "second.tar.gz"
                first.write_bytes(b"a" * size)
                second.write_bytes(b"b" * size)
                uri = self._uri(f"{size}.tar.gz")
                self.store.put_create(uri, first)
                blob = storage.Blob.from_uri(uri, client=self.client)
                blob.reload()
                generation = blob.generation
                with self.assertRaises(remote_cache.RemoteEntryExists):
                    self.store.put_create(uri, second)
                blob.reload()
                self.assertEqual(blob.generation, generation)
                self.assertEqual(blob.download_as_bytes(), first.read_bytes())

    def test_reader_passes_generation_guard_to_every_chunk(self):
        uri = self._uri("generation.tar.gz")
        blob = storage.Blob.from_uri(uri, client=self.client)
        original = b"a" * (2 << 20)
        replacement = b"b" * (2 << 20)
        blob.upload_from_string(original)
        blob.reload()
        first_generation = blob.generation
        original_open = storage.Blob.open
        original_download = storage.Blob.download_as_bytes
        chunk_guards = []

        def replace_before_open(reader_blob, *args, **kwargs):
            self.assertEqual(reader_blob.generation, first_generation)
            self.assertEqual(kwargs["if_generation_match"], first_generation)
            storage.Blob.from_uri(uri, client=self.client).upload_from_string(replacement)
            return original_open(reader_blob, *args, **kwargs)

        def inspect_chunk(reader_blob, *args, **kwargs):
            chunk_guards.append(kwargs["if_generation_match"])
            return original_download(reader_blob, *args, **kwargs)

        with (
            patch.object(storage.Blob, "open", replace_before_open),
            patch.object(storage.Blob, "download_as_bytes", inspect_chunk),
        ):
            try:
                with self.store.open_reader(uri) as stream:
                    data = stream.read(1 << 20) + stream.read(1 << 20)
            except (FileNotFoundError, PreconditionFailed):
                data = None
        self.assertGreaterEqual(len(chunk_guards), 2)
        self.assertEqual(set(chunk_guards), {first_generation})
        # The emulator currently ignores read preconditions after replacement.
        self.assertIn(data, (None, original, replacement))

    def test_archive_with_sidecars_and_malformed_object(self):
        entry = remote_cache.RemoteCacheEntry(
            "lto", "lto", {"symbol": "fft"}, ("symbol.lto", "symbol.meta", "symbol_fatbin.lto")
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = {name: root / name for name in entry.artifact_names}
            for name, path in sources.items():
                path.write_bytes(name.encode())
            archive_path = root / "archive.tar.gz"
            with archive_path.open("wb") as stream:
                remote_cache.write_archive(stream, entry, sources)
            uri = self._uri("valid.tar.gz")
            self.store.put_create(uri, archive_path)
            with self.store.open_reader(uri) as stream:
                staged = root / "staged"
                staged.mkdir()
                remote_cache.read_archive(stream, entry, staged)
            for name, source in sources.items():
                self.assertEqual((staged / name).read_bytes(), source.read_bytes())

            bad_uri = self._uri("malformed.tar.gz")
            storage.Blob.from_uri(bad_uri, client=self.client).upload_from_file(io.BytesIO(b"bad archive"))
            with self.store.open_reader(bad_uri) as stream:
                with self.assertRaises(remote_cache.RemoteCacheValidationError):
                    remote_cache.read_archive(stream, entry, staged)

    def _run_helper(self, mode, root, *extra):
        env = os.environ.copy()
        env["WARP_CACHE_PATH"] = str(root)
        helper = Path(__file__).with_name("aux_test_remote_cache.py")
        return subprocess.run(
            [sys.executable, str(helper), mode, self._uri_root, *map(str, extra)],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=90,
        )

    def test_kernel_cache_crosses_processes_and_read_only_consumer(self):
        self._uri_root = self._uri("kernel")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            producer = self._run_helper("producer", root / "producer")
            self.assertEqual(producer.returncode, 0, producer.stdout)
            consumer = self._run_helper("consumer", root / "consumer")
            self.assertEqual(consumer.returncode, 0, consumer.stdout)

    def test_two_producers_publish_one_entry_for_third_process(self):
        self._uri_root = self._uri("race")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            helper = Path(__file__).with_name("aux_test_remote_cache.py")
            barrier = root / "barrier"
            processes = []
            for index in range(2):
                env = os.environ.copy()
                env["WARP_CACHE_PATH"] = str(root / f"producer-{index}")
                processes.append(
                    subprocess.Popen(
                        [sys.executable, str(helper), "race", self._uri_root, str(barrier)],
                        env=env,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                )
            for process in processes:
                output, _ = process.communicate(timeout=90)
                self.assertEqual(process.returncode, 0, output)
            consumer = self._run_helper("consumer", root / "consumer")
            self.assertEqual(consumer.returncode, 0, consumer.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
