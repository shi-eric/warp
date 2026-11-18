# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subprocess entry point for remote kernel cache integration tests.

Usage: ``aux_test_remote_cache.py MODE ROOT``, where ``MODE`` is ``producer``
or ``consumer`` (read-only and not allowed to compile).
"""

import sys
from contextlib import ExitStack
from unittest.mock import patch

import warp as wp
from warp._src import context, remote_cache


@wp.kernel
def _remote_cache_subprocess_kernel(values: wp.array[int]):
    i = wp.tid()
    values[i] += 7


def main() -> None:
    mode, root = sys.argv[1], sys.argv[2]
    wp.config.remote_cache_dir = root
    wp.config.remote_cache_min_compile_time = 0.0
    wp.config.remote_cache_read_only = mode == "consumer"

    with ExitStack() as stack:
        # A development checkout is not a published build, so it is opted in here.
        stack.enter_context(patch.object(remote_cache, "_is_shareable_version", return_value=True))
        if mode == "consumer":
            stack.enter_context(
                patch.object(context.Module, "_run_codegen", side_effect=AssertionError("consumer compiled"))
            )
        wp.init()
        values = wp.array([1, 2, 3], dtype=int, device="cpu")
        wp.launch(_remote_cache_subprocess_kernel, dim=3, inputs=[values], device="cpu")
        assert values.numpy().tolist() == [8, 9, 10]


if __name__ == "__main__":
    main()
