# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Google Cloud Storage backend for the remote compilation cache."""

from pathlib import Path
from urllib.parse import urlsplit

from google.api_core.exceptions import NotFound, PreconditionFailed
from google.api_core.retry import Retry
from google.cloud import storage

_REQUEST_TIMEOUT = (10, 30)
_RETRY = Retry(timeout=60)


class GCSRemoteStore:
    """Store each cache file as one object below a ``gs://bucket/prefix`` root.

    Args:
        root: Remote cache root as a ``gs://bucket/prefix`` URL.
        client: Client to use instead of one built from Application Default
            Credentials.
    """

    def __init__(self, root: str, client: storage.Client | None = None):
        parsed = urlsplit(root)
        if parsed.scheme != "gs" or not parsed.netloc:
            raise ValueError(f"'{root}' is not a gs://bucket/prefix URL")
        prefix = parsed.path.strip("/")
        self._prefix = f"{prefix}/" if prefix else ""
        self._bucket = (client if client is not None else storage.Client()).bucket(parsed.netloc)

    def get(self, key: str, path: Path) -> bool:
        blob = self._bucket.blob(self._prefix + key)
        try:
            blob.download_to_filename(str(path), timeout=_REQUEST_TIMEOUT, retry=_RETRY)
        except NotFound:
            return False
        return True

    def put(self, path: Path, key: str) -> None:
        blob = self._bucket.blob(self._prefix + key)
        try:
            # Creating without replacing needs no delete permission on the bucket.
            blob.upload_from_filename(str(path), if_generation_match=0, timeout=_REQUEST_TIMEOUT, retry=_RETRY)
        except PreconditionFailed:
            # Another job published this file first, and its copy is equivalent.
            pass
