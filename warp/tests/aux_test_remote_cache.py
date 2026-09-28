# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subprocess entry point for remote kernel cache integration tests."""

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

from google.auth.credentials import AnonymousCredentials
from google.cloud import storage

import warp as wp
from warp._src import context, remote_cache
from warp._src.remote_cache_gcs import GCSRemoteStore


@wp.kernel
def _remote_cache_subprocess_kernel(values: wp.array[int]):
    i = wp.tid()
    values[i] += 7


def _wait_for_producers(barrier_dir: Path) -> None:
    barrier_dir.mkdir(parents=True, exist_ok=True)
    (barrier_dir / str(os.getpid())).touch()
    deadline = time.monotonic() + 30
    while len(list(barrier_dir.iterdir())) < 2:
        if time.monotonic() > deadline:
            raise TimeoutError("Remote cache producer barrier timed out")
        time.sleep(0.05)


def main() -> None:
    mode = sys.argv[1]
    wp.config.remote_cache_dir = sys.argv[2]
    wp.config.remote_cache_min_compile_time = 0.0
    wp.config.remote_cache_read_only = mode == "consumer"
    client = storage.Client(credentials=AnonymousCredentials(), project="test")
    store = GCSRemoteStore(client=client)
    if mode == "race":
        original_put = store.put_create
        barrier_dir = Path(sys.argv[3])

        def put_after_barrier(uri, archive_path):
            _wait_for_producers(barrier_dir)
            original_put(uri, archive_path)

        store.put_create = put_after_barrier

    with (
        patch.object(remote_cache, "_is_final_release_version", return_value=True),
        patch.object(remote_cache, "_create_store", return_value=store),
    ):
        wp.init()
        values = wp.array([1, 2, 3], dtype=int, device="cpu")
        if mode == "consumer":
            with patch.object(context.Module, "_run_codegen", side_effect=AssertionError("consumer compiled")):
                wp.launch(_remote_cache_subprocess_kernel, dim=3, inputs=[values], device="cpu")
        else:
            wp.launch(_remote_cache_subprocess_kernel, dim=3, inputs=[values], device="cpu")
        assert values.numpy().tolist() == [8, 9, 10]


if __name__ == "__main__":
    main()
