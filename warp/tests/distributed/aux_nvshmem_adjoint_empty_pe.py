# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-PE test that a PE with no threads in a flagged adjoint launch still takes part in its barrier.

This program is launched by ``TestNVSHMEMAutodiff.test_empty_pe`` through ``mpirun -np 2``. PE 0 pushes a value
into PE 1 with one thread, while PE 1 launches the same kernel with no threads, forward and adjoint. If PE 1 skipped
the barrier before its empty adjoint launch, the barrier counts of the two PEs would drift apart and a later
collective would hang, until the parent test's timeout ends the run.
"""

import numpy as np
import nvshmem.core as nvshmem
from mpi4py import MPI
from nvshmem.bindings import nvshmem as nvshmem_bindings

import warp as wp
from warp.tests.distributed.aux_nvshmem_record_kernels import push_kernel
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


# push_kernel pushes values[0] into dest[0] on PE 1.
dest = symmetric(1, wp.float32, requires_grad=True)
values = wp.array([1.5], dtype=wp.float32, device=device, requires_grad=True)
dest_seed = 7.25
dim = 1 if rank == 0 else 0

sync()
wp.launch(push_kernel, dim=dim, inputs=[dest, values], device=device)
sync()
if rank == 1:
    dest.grad.fill_(dest_seed)
sync()
wp.launch(push_kernel, dim=dim, inputs=[dest, values], adj_inputs=[dest.grad, values.grad], adjoint=True, device=device)
sync()
if rank == 0:
    check("empty PE participates", values.grad.numpy(), [dest_seed])

sync()
for buf in symmetric_buffers:
    nvshmem.free(buf)
nvshmem.finalize()
MPI.Finalize()
