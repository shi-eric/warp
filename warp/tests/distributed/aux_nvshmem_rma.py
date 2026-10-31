# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-PE test of the NVSHMEM RMA and signal built-ins.

This program is launched by ``TestNVSHMEMMultiPE.test_rma`` through ``mpirun -np 2``. PE 0 issues the
operations against PE 1, and each check runs on the PE that owns the result.
"""

import numpy as np
import nvshmem.core as nvshmem
from mpi4py import MPI
from nvshmem.bindings import nvshmem as nvshmem_bindings

import warp as wp
from warp.types import type_size_in_bytes

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

wp.config.quiet = True

local_rank = comm.Split_type(MPI.COMM_TYPE_SHARED).Get_rank()
wp.set_device(f"cuda:{local_rank}" if wp.get_cuda_device_count() > 1 else "cuda:0")
device = wp.get_device()
nvshmem.init(mpi_comm=comm, initializer_method="mpi")
assert nvshmem.n_pes() == 2, "Need exactly 2 PEs"

symmetric_buffers = []


def symmetric(shape, dtype):
    """Allocate a zeroed symmetric array. Every PE must call this in the same order."""
    buf = nvshmem.buffer(int(np.prod(shape)) * type_size_in_bytes(dtype))
    symmetric_buffers.append(buf)
    arr = wp.array(ptr=int(buf.handle), shape=shape, dtype=dtype, device=device)
    arr.zero_()
    return arr


def sync():
    wp.synchronize_device(device)
    nvshmem_bindings.barrier_all()


def check(name, actual, expected):
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64))
    print(f"PE {rank}: {name}: PASSED", flush=True)


@wp.kernel
def put_scalars(
    strided: wp.array[wp.float32],
    grid: wp.array2d[wp.float64],
    vectors: wp.array[wp.vec3],
    halves: wp.array[wp.float16],
):
    if wp.nvshmem_my_pe() == 0:
        wp.nvshmem_p(strided, 1, 42.0, 1)
        wp.nvshmem_p(grid, 1, 2, wp.float64(2.5), 1)
        wp.nvshmem_p(vectors, 3, wp.vec3(1.0, 2.0, 3.0), 1)
        wp.nvshmem_p(halves, 0, wp.float16(1.5), 1)
        wp.nvshmem_quiet()


@wp.kernel
def get_scalars(remote: wp.array[wp.int64], remote_vectors: wp.array[wp.vec3], out: wp.array[wp.float64]):
    if wp.nvshmem_my_pe() == 0:
        for i in range(remote.shape[0]):
            out[i] = wp.float64(wp.nvshmem_g(remote, i, 1))
        out[remote.shape[0]] = wp.float64(wp.nvshmem_g(remote_vectors, 1, 1)[2])


@wp.kernel
def transfer(
    dest: wp.array[wp.float32],
    dest_strided: wp.array[wp.float32],
    dest_columns: wp.array2d[wp.float32],
    source: wp.array[wp.float32],
    source_columns: wp.array2d[wp.float32],
    remote_strided: wp.array[wp.float32],
    fetched: wp.array[wp.float32],
):
    if wp.nvshmem_my_pe() == 0:
        wp.nvshmem_put(dest, source, 1)
        wp.nvshmem_put(dest_strided, source[0:4], 1)
        wp.nvshmem_put(dest_columns, source_columns, 1)
        wp.nvshmem_get(fetched, remote_strided, 1)
        wp.nvshmem_quiet()


@wp.kernel
def put_with_signal(
    dest: wp.array[wp.float32],
    dest_strided: wp.array[wp.float32],
    source: wp.array[wp.float32],
    signals: wp.array[wp.uint64],
    received: wp.array[wp.float32],
):
    if wp.nvshmem_my_pe() == 0:
        wp.nvshmem_put_signal(dest, source, signals, 0, 1, wp.NVSHMEM_SIGNAL_SET, 1)
        wp.nvshmem_put_signal(dest_strided, source[0:4], signals, 1, 1, wp.NVSHMEM_SIGNAL_SET, 1)
        wp.nvshmem_signal_op(signals, 2, 5, wp.NVSHMEM_SIGNAL_ADD, 1)
        wp.nvshmem_signal_op(signals, 2, 5, wp.NVSHMEM_SIGNAL_ADD, 1)
    else:
        wp.nvshmem_signal_wait_until(signals, 0, wp.NVSHMEM_CMP_EQ, 1)
        wp.nvshmem_signal_wait_until(signals, 1, wp.NVSHMEM_CMP_EQ, 1)
        total = wp.nvshmem_signal_wait_until(signals, 2, wp.NVSHMEM_CMP_GE, 10)
        for i in range(dest.shape[0]):
            received[i] = dest[i]
        for i in range(dest_strided.shape[0]):
            received[dest.shape[0] + i] = dest_strided[i]
        received[dest.shape[0] + dest_strided.shape[0]] = wp.float32(total)


# Element puts: a strided view, a 2D index, a compound dtype, and a 16-bit dtype.
strided_base = symmetric(8, wp.float32)
grid = symmetric((3, 4), wp.float64)
vectors = symmetric(4, wp.vec3)
halves = symmetric(2, wp.float16)
sync()
wp.launch(put_scalars, dim=1, inputs=[strided_base[::2], grid, vectors, halves], device=device)
sync()
if rank == 1:
    check("p into strided view", strided_base.numpy(), [0, 0, 42, 0, 0, 0, 0, 0])
    expected_grid = np.zeros((3, 4))
    expected_grid[1, 2] = 2.5
    check("p with 2D index", grid.numpy(), expected_grid)
    check("p of vec3", vectors.numpy()[3], [1, 2, 3])
    check("p of float16", halves.numpy()[0], 1.5)

# Element gets from PE 1's values.
remote = symmetric(5, wp.int64)
remote_vectors = symmetric(2, wp.vec3)
if rank == 1:
    remote.assign(np.array([10, -20, 30, -40, 2**40], dtype=np.int64))
    remote_vectors.assign(np.array([[0, 0, 0], [4, 5, 6]], dtype=np.float32))
gotten = wp.zeros(6, dtype=wp.float64, device=device)
sync()
wp.launch(get_scalars, dim=1, inputs=[remote, remote_vectors, gotten], device=device)
sync()
if rank == 0:
    check("g of int64 and vec3", gotten.numpy(), [10, -20, 30, -40, 2**40, 6])

# Bulk transfers: contiguous, strided 1D, non-contiguous 2D, and a strided get.
dest = symmetric(6, wp.float32)
dest_strided_base = symmetric(8, wp.float32)
dest_grid = symmetric((4, 4), wp.float32)
remote_values = symmetric(8, wp.float32)
source = wp.array(np.arange(1, 7, dtype=np.float32), device=device)
source_columns = wp.array(np.arange(100, 108, dtype=np.float32).reshape(4, 2), device=device)
fetched = wp.zeros(4, dtype=wp.float32, device=device)
if rank == 1:
    remote_values.assign(np.arange(200, 208, dtype=np.float32))
sync()
wp.launch(
    transfer,
    dim=1,
    inputs=[dest, dest_strided_base[1::2], dest_grid[:, 1:3], source, source_columns, remote_values[::2], fetched],
    device=device,
)
sync()
if rank == 1:
    check("contiguous put", dest.numpy(), np.arange(1, 7))
    check("strided put", dest_strided_base.numpy(), [0, 1, 0, 2, 0, 3, 0, 4])
    expected_columns = np.zeros((4, 4))
    expected_columns[:, 1:3] = np.arange(100, 108).reshape(4, 2)
    check("2D non-contiguous put", dest_grid.numpy(), expected_columns)
if rank == 0:
    check("strided get", fetched.numpy(), [200, 202, 204, 206])

# Signaled puts and signal operations; PE 1 waits on the signals inside the kernel.
signaled = symmetric(6, wp.float32)
signaled_strided_base = symmetric(8, wp.float32)
signals = symmetric(3, wp.uint64)
received = wp.zeros(11, dtype=wp.float32, device=device)
sync()
wp.launch(
    put_with_signal,
    dim=1,
    inputs=[signaled, signaled_strided_base[::2], source, signals, received],
    device=device,
)
sync()
if rank == 1:
    check("put_signal, signal_op, and signal_wait_until", received.numpy(), [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 10])

sync()
for buf in symmetric_buffers:
    nvshmem.free(buf)
nvshmem.finalize()
MPI.Finalize()
