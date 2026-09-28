# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Google Cloud Storage adapter for the internal compilation cache."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from google.api_core.exceptions import NotFound, PreconditionFailed
from google.api_core.retry import Retry
from google.cloud import storage

from warp._src.remote_cache import RemoteEntryExists

_REQUEST_TIMEOUT = (10, 30)
_RETRY = Retry(timeout=60)
_READ_CHUNK_BYTES = 1 << 20


class GCSRemoteStore:
    """Read exact object keys and publish only absent objects."""

    def __init__(self, client: storage.Client | None = None):
        self.client = client if client is not None else storage.Client()

    @contextmanager
    def open_reader(self, uri: str) -> Iterator[BinaryIO]:
        blob = storage.Blob.from_uri(uri, client=self.client)
        try:
            blob.reload(timeout=_REQUEST_TIMEOUT, retry=_RETRY)
            with blob.open("rb", chunk_size=_READ_CHUNK_BYTES, timeout=_REQUEST_TIMEOUT, retry=_RETRY) as stream:
                yield stream
        except NotFound as exc:
            raise FileNotFoundError(uri) from exc

    def put_create(self, uri: str, archive_path: Path) -> None:
        blob = storage.Blob.from_uri(uri, client=self.client)
        try:
            blob.upload_from_filename(
                str(archive_path),
                content_type="application/gzip",
                if_generation_match=0,
                timeout=_REQUEST_TIMEOUT,
                retry=_RETRY,
            )
        except PreconditionFailed as exc:
            raise RemoteEntryExists(uri) from exc
