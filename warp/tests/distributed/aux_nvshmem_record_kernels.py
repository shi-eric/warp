# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernels whose adjoints exercise the per-kernel NVSHMEM records written to module metadata.

The kernels share this one Warp module, so a single compile produces one metadata file holding every record.
``TestNVSHMEMAutodiff.test_codegen_records`` compiles the module ahead of time and checks the records.
"""

import warp as wp


@wp.struct
class FloatPair:
    x: float
    y: float


@wp.func
def push_helper(dest: wp.array[wp.float32], v: float, pe: int):
    wp.nvshmem_p(dest, 0, v, pe)


@wp.kernel
def push_kernel(dest: wp.array[wp.float32], values: wp.array[wp.float32]):
    wp.nvshmem_p(dest, 0, values[0], 1)


@wp.kernel
def via_func_kernel(dest: wp.array[wp.float32], values: wp.array[wp.float32]):
    push_helper(dest, values[0], 1)


@wp.kernel
def pull_kernel(source: wp.array[wp.float32], out: wp.array[wp.float32]):
    out[0] = wp.nvshmem_g(source, 0, 1)


@wp.kernel
def get_only_kernel(dest: wp.array[wp.float32], source: wp.array[wp.float32]):
    wp.nvshmem_get(dest, source, 1)


@wp.kernel
def fused_kernel(dest: wp.array[wp.float32], source: wp.array[wp.float32], sig: wp.array[wp.uint64]):
    wp.nvshmem_put_signal(dest, source, sig, 0, 1, wp.NVSHMEM_SIGNAL_SET, 1)
    wp.nvshmem_signal_wait_until(sig, 0, wp.NVSHMEM_CMP_EQ, 1)


@wp.kernel
def pipelined_kernel(dest: wp.array[wp.float32], values: wp.array[wp.float32], sig: wp.array[wp.uint64]):
    wp.nvshmem_signal_wait_until(sig, 0, wp.NVSHMEM_CMP_EQ, 1)
    wp.nvshmem_p(dest, 0, values[0], 1)


@wp.kernel
def local_kernel(values: wp.array[wp.float32]):
    values[0] = values[0] * 2.0


@wp.kernel
def other_dtypes_kernel(
    dest_int8: wp.array[wp.int8],
    values_int8: wp.array[wp.int8],
    dest_uint16: wp.array[wp.uint16],
    values_uint16: wp.array[wp.uint16],
    dest_bool: wp.array[wp.bool],
    values_bool: wp.array[wp.bool],
    dest_pair: wp.array[FloatPair],
    values_pair: wp.array[FloatPair],
):
    # Compiling the backward pass of this kernel covers the push adjoints for non-float element types.
    wp.nvshmem_p(dest_int8, 0, values_int8[0], 1)
    wp.nvshmem_p(dest_uint16, 0, values_uint16[0], 1)
    wp.nvshmem_p(dest_bool, 0, values_bool[0], 1)
    wp.nvshmem_p(dest_pair, 0, values_pair[0], 1)
