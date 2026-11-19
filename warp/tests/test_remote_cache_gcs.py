# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Google Cloud Storage backend tests.

The backend runs against an in-memory stand-in for the storage client, so the
tests need the ``remote-cache-gcs`` extra but no network, credentials, or
storage service.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import warp.config
from warp._src import remote_cache

try:
    from google.api_core.exceptions import Forbidden, NotFound, PreconditionFailed
except ImportError:
    Forbidden = NotFound = PreconditionFailed = None

KEY = "1.19.0/wp_mod_abc1234/wp_mod_abc1234.o"


class FakeBlob:
    """One object of a :class:`FakeClient`, with the two transfers the backend uses."""

    def __init__(self, client: "FakeClient", bucket: str, name: str):
        self._client = client
        self._key = (bucket, name)

    def download_to_filename(self, filename, **kwargs):
        if self._client.error is not None:
            raise self._client.error
        if self._key not in self._client.objects:
            raise NotFound(f"No such object: {self._key}")
        Path(filename).write_bytes(self._client.objects[self._key])

    def upload_from_filename(self, filename, if_generation_match=None, **kwargs):
        if self._client.error is not None:
            raise self._client.error
        # Generation 0 is the service's way to say "only if the object does not exist".
        if if_generation_match == 0 and self._key in self._client.objects:
            raise PreconditionFailed(f"Object exists: {self._key}")
        self._client.objects[self._key] = Path(filename).read_bytes()


class FakeBucket:
    def __init__(self, client: "FakeClient", name: str):
        self._client = client
        self._name = name

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self._client, self._name, name)


class FakeClient:
    """In-memory stand-in for ``google.cloud.storage.Client``.

    Args:
        error: Exception that every transfer raises, as a service failure would.
    """

    def __init__(self, error: Exception | None = None):
        self.objects: dict[tuple[str, str], bytes] = {}  # (bucket name, object name) -> contents
        self.error = error

    def bucket(self, name: str) -> FakeBucket:
        return FakeBucket(self, name)


class TestRemoteCacheGCS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if NotFound is None:
            raise unittest.SkipTest("Install the remote-cache-gcs extra")

    def setUp(self):
        from warp._src.remote_cache_gcs import GCSRemoteStore  # noqa: PLC0415

        self.GCSRemoteStore = GCSRemoteStore
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.source = self.tmp / "source.o"
        self.source.write_bytes(b"binary")
        self.client = FakeClient()
        self.store = GCSRemoteStore("gs://bucket/warp-cache", client=self.client)

    def test_missing_object_is_a_miss(self):
        self.assertFalse(self.store.get(KEY, self.tmp / "missing.o"))

    def test_get_restores_object_contents(self):
        self.client.objects["bucket", "warp-cache/1.19.0/wp_mod_abc1234/wp_mod_abc1234.o"] = b"published"
        restored = self.tmp / "restored.o"
        self.assertTrue(self.store.get(KEY, restored))
        self.assertEqual(restored.read_bytes(), b"published")

    def test_put_names_object_by_root_prefix_and_key(self):
        cases = {
            "gs://bucket/warp-cache": "warp-cache/1.19.0/wp_mod_abc1234/wp_mod_abc1234.o",
            "gs://bucket/team/warp-cache/": "team/warp-cache/1.19.0/wp_mod_abc1234/wp_mod_abc1234.o",
            "gs://bucket": "1.19.0/wp_mod_abc1234/wp_mod_abc1234.o",
        }
        for root, name in cases.items():
            with self.subTest(root=root):
                client = FakeClient()
                self.GCSRemoteStore(root, client=client).put(self.source, KEY)
                self.assertEqual(client.objects, {("bucket", name): b"binary"})

    def test_put_keeps_existing_object(self):
        name = "warp-cache/1.19.0/wp_mod_abc1234/wp_mod_abc1234.o"
        self.client.objects["bucket", name] = b"first"
        self.store.put(self.source, KEY)
        self.assertEqual(self.client.objects, {("bucket", name): b"first"})

    def test_transfer_errors_reach_the_caller(self):
        store = self.GCSRemoteStore("gs://bucket/warp-cache", client=FakeClient(error=Forbidden("access denied")))
        with self.assertRaises(Forbidden):
            store.get(KEY, self.tmp / "denied.o")
        with self.assertRaises(Forbidden):
            store.put(self.source, KEY)

    def test_rejects_roots_that_are_not_gcs_urls(self):
        for root in ("gs://", "s3://bucket/warp-cache", "bucket/warp-cache"):
            with self.subTest(root=root), self.assertRaises(ValueError):
                self.GCSRemoteStore(root, client=self.client)

    def test_gs_root_reaches_backend_through_remote_cache(self):
        self.addCleanup(remote_cache.init_remote_cache)
        staging = self.tmp / "staging"
        staging.mkdir()
        local = self.tmp / "wp_mod_abc1234.o"
        local.write_bytes(b"binary")
        with (
            patch.object(warp.config, "remote_cache_dir", "gs://bucket/warp-cache/"),
            patch.object(warp.config, "remote_cache_min_compile_time", 0.0),
            patch.object(remote_cache, "_is_shareable_version", return_value=True),
            patch("google.cloud.storage.Client", return_value=self.client),
        ):
            remote_cache.init_remote_cache()
            remote_cache.publish("wp_mod_abc1234", [local], 5.0)
            self.assertEqual(
                self.client.objects,
                {("bucket", f"warp-cache/{warp.config.version}/wp_mod_abc1234/wp_mod_abc1234.o"): b"binary"},
            )
            self.assertTrue(remote_cache.fetch("wp_mod_abc1234", ["wp_mod_abc1234.o"], staging))
        self.assertEqual((staging / "wp_mod_abc1234.o").read_bytes(), b"binary")


if __name__ == "__main__":
    unittest.main(verbosity=2)
