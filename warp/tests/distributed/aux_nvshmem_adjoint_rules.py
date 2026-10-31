# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-PE test of the adjoints of the NVSHMEM push built-ins.

This program is launched by ``TestNVSHMEMAutodiff.test_adjoint_rules`` through ``mpirun -np 2``. PE 0 pushes
values to PE 1 in a forward launch, PE 1 seeds the gradients of the destination arrays, and the adjoint launch
on PE 0 must pull those gradients back into the gradients of the pushed values.
"""

import numpy as np
import nvshmem.core as nvshmem
from mpi4py import MPI
from nvshmem.bindings import nvshmem as nvshmem_bindings

import warp as wp
from warp.types import type_size_in_bytes

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

wp.config.log_level = wp.LOG_WARNING

local_rank = comm.Split_type(MPI.COMM_TYPE_SHARED).Get_rank()
wp.set_device(f"cuda:{local_rank}" if wp.get_cuda_device_count() > 1 else "cuda:0")
device = wp.get_device()
nvshmem.init(mpi_comm=comm, initializer_method="mpi")
assert nvshmem.n_pes() == 2, "Need exactly 2 PEs"

symmetric_buffers = []


def symmetric(shape, dtype, requires_grad=False):
    """Allocate a zeroed symmetric array. Every PE must call this in the same order.

    With ``requires_grad``, a second zeroed symmetric allocation becomes the array's gradient, so other PEs can
    address the gradient too.
    """

    def allocate():
        buf = nvshmem.buffer(int(np.prod(shape)) * type_size_in_bytes(dtype))
        symmetric_buffers.append(buf)
        arr = wp.array(ptr=int(buf.handle), shape=shape, dtype=dtype, device=device)
        arr.zero_()
        return arr

    arr = allocate()
    if requires_grad:
        arr.grad = allocate()
    return arr


def sync():
    wp.synchronize_device(device)
    nvshmem_bindings.barrier_all()


def check(name, actual, expected):
    """Compare on this PE; on a mismatch, abort every PE instead of leaving the other one in a collective."""
    try:
        np.testing.assert_allclose(np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64))
    except AssertionError as error:
        print(f"PE {rank}: {name}: FAILED\n{error}", flush=True)
        comm.Abort(1)
    print(f"PE {rank}: {name}: PASSED", flush=True)


@wp.kernel
def push_elements(
    dest: wp.array[wp.float32],
    grid: wp.array2d[wp.float64],
    vectors: wp.array[wp.vec3],
    halves: wp.array[wp.float16],
    values: wp.array[wp.float32],
):
    if wp.nvshmem_my_pe() == 0:
        v = values[0] * 2.0
        wp.nvshmem_p(dest, 3, v, 1)
        wp.nvshmem_p(grid, 1, 2, wp.float64(values[1]), 1)
        wp.nvshmem_p(vectors, 1, wp.vec3(values[2], values[2], values[2]), 1)
        wp.nvshmem_p(halves, 0, wp.float16(values[3]), 1)


@wp.kernel
def push_bulk(
    dest: wp.array[wp.float32],
    dest_strided: wp.array[wp.float32],
    dest_grid: wp.array2d[wp.float32],
    source: wp.array[wp.float32],
    source_grid: wp.array2d[wp.float32],
):
    if wp.nvshmem_my_pe() == 0:
        wp.nvshmem_put(dest, source, 1)
        wp.nvshmem_put(dest_strided, source[0:4], 1)
        wp.nvshmem_put(dest_grid, source_grid, 1)


@wp.kernel
def push_signaled(dest: wp.array[wp.float32], source: wp.array[wp.float32], signals: wp.array[wp.uint64]):
    if wp.nvshmem_my_pe() == 0:
        wp.nvshmem_put_signal(dest, source, signals, 0, 1, wp.NVSHMEM_SIGNAL_SET, 1)


def grads(arrays):
    return [a.grad for a in arrays]


# Element puts: a 1D index, a 2D index, a compound dtype, and a 16-bit dtype.
dest = symmetric(4, wp.float32, requires_grad=True)
grid = symmetric((3, 4), wp.float64, requires_grad=True)
vectors = symmetric(2, wp.vec3, requires_grad=True)
halves = symmetric(2, wp.float16, requires_grad=True)
values = wp.array([1.5, 2.5, 3.5, 4.5], dtype=wp.float32, device=device, requires_grad=True)
element_args = [dest, grid, vectors, halves, values]
dest_seed = np.array([10, 20, 30, 40], dtype=np.float32)
grid_seed = np.arange(100, 112, dtype=np.float64).reshape(3, 4)
vectors_seed = np.array([[1, 2, 3], [4, 5, 6.5]], dtype=np.float32)
halves_seed = np.array([0.75, -1.25], dtype=np.float16)
sync()
wp.launch(push_elements, dim=1, inputs=element_args, device=device)
sync()
if rank == 1:
    dest.grad.assign(dest_seed)
    grid.grad.assign(grid_seed)
    vectors.grad.assign(vectors_seed)
    halves.grad.assign(halves_seed)
sync()
wp.launch(push_elements, dim=1, inputs=element_args, adj_inputs=grads(element_args), adjoint=True, device=device)
sync()
if rank == 0:
    check(
        "p element gradients",
        values.grad.numpy(),
        [2 * dest_seed[3], grid_seed[1, 2], vectors_seed[1].sum(), halves_seed[0]],
    )

# Bulk puts: contiguous, a strided 1D view fed by an in-kernel slice, and a non-contiguous 2D view.
dest = symmetric(6, wp.float32, requires_grad=True)
dest_strided_base = symmetric(8, wp.float32, requires_grad=True)
dest_grid_base = symmetric((4, 4), wp.float32, requires_grad=True)
source = wp.array(np.arange(1, 7, dtype=np.float32), device=device, requires_grad=True)
source_grid = wp.array(np.arange(100, 108, dtype=np.float32).reshape(4, 2), device=device, requires_grad=True)
bulk_args = [dest, dest_strided_base[1::2], dest_grid_base[:, 1:3], source, source_grid]
dest_seed = np.arange(1, 7, dtype=np.float32)
dest_strided_seed = np.arange(10, 18, dtype=np.float32)
dest_grid_seed = np.arange(200, 216, dtype=np.float32).reshape(4, 4)
sync()
wp.launch(push_bulk, dim=1, inputs=bulk_args, device=device)
sync()
if rank == 1:
    dest.grad.assign(dest_seed)
    dest_strided_base.grad.assign(dest_strided_seed)
    dest_grid_base.grad.assign(dest_grid_seed)
sync()
wp.launch(push_bulk, dim=1, inputs=bulk_args, adj_inputs=grads(bulk_args), adjoint=True, device=device)
sync()
if rank == 0:
    expected_source = dest_seed.copy()
    expected_source[0:4] += dest_strided_seed[1::2]
    check(
        "put bulk gradients",
        np.concatenate((source.grad.numpy(), source_grid.grad.numpy().ravel())),
        np.concatenate((expected_source, dest_grid_seed[:, 1:3].ravel())),
    )

# Signaled put; the signal word carries no gradient.
signaled = symmetric(6, wp.float32, requires_grad=True)
signals = symmetric(1, wp.uint64)
signaled_source = wp.array(np.arange(1, 7, dtype=np.float32), device=device, requires_grad=True)
signaled_args = [signaled, signaled_source, signals]
signaled_seed = np.arange(20, 26, dtype=np.float32)
sync()
wp.launch(push_signaled, dim=1, inputs=signaled_args, device=device)
sync()
if rank == 1:
    signaled.grad.assign(signaled_seed)
sync()
wp.launch(
    push_signaled,
    dim=1,
    inputs=signaled_args,
    adj_inputs=[signaled.grad, signaled_source.grad, None],
    adjoint=True,
    device=device,
)
sync()
if rank == 0:
    check("put_signal gradients", signaled_source.grad.numpy(), signaled_seed)

# Bulk puts into destinations without gradients leave the source gradients untouched.
unattached_args = [
    symmetric(6, wp.float32),
    symmetric(8, wp.float32)[1::2],
    symmetric((4, 4), wp.float32)[:, 1:3],
    source,
    source_grid,
]
source.grad.zero_()
source_grid.grad.zero_()
sync()
wp.launch(push_bulk, dim=1, inputs=unattached_args, device=device)
sync()
wp.launch(
    push_bulk,
    dim=1,
    inputs=unattached_args,
    adj_inputs=[None, None, None, source.grad, source_grid.grad],
    adjoint=True,
    device=device,
)
sync()
if rank == 0:
    check(
        "no grad attached",
        np.concatenate((source.grad.numpy(), source_grid.grad.numpy().ravel())),
        np.zeros(14),
    )

sync()
for buf in symmetric_buffers:
    nvshmem.free(buf)
nvshmem.finalize()
MPI.Finalize()
