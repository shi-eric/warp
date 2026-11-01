# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distributed Jacobi solver with NVSHMEM puts from the stencil kernel.

This example solves the Laplace equation with Jacobi iteration on a square mesh. The interior rows are split into one
slab per PE, and the rows wrap around, so the PE above PE 0 is the last PE. Each PE keeps a ghost row above and below
its slab. The stencil kernel writes each boundary value into the neighbor's ghost row with ``wp.nvshmem_p()`` as it
computes it, and after each step the host waits for the device and then for every PE at an NVSHMEM barrier. At the
end, PE 0 prints the L2 norm of the last update.

This is the first stage of the halo exchange guide, which ends with ``example_jacobi_nvshmem_overlap.py``:
https://nvidia.github.io/warp/stable/user_guide/distributed/nvshmem_halo_exchange.html

Launch:
    mpirun -np 2 uv run --with mpi4py --with nvshmem4py-cu13 --with "nvidia-nvshmem-cu13>=3.9" \
        python -m warp.examples.distributed.example_jacobi_nvshmem_device [--iterations N] [--size N]
"""

# docs: setup begin
import argparse

import numpy as np
import nvshmem.core as nvshmem
from mpi4py import MPI
from nvshmem.bindings import nvshmem as nvshmem_bindings

import warp as wp

parser = argparse.ArgumentParser(description="Distributed Jacobi solver with NVSHMEM halo exchange.")
parser.add_argument("--iterations", type=int, default=1000, help="Number of Jacobi steps.")
parser.add_argument("--size", type=int, default=512, help="Number of rows and columns of the square mesh.")
args = parser.parse_args()

wp.config.cuda_output = "cubin"
wp.config.log_level = wp.LOG_WARNING

# Importing mpi4py.MPI has initialized MPI.
mpi_comm = MPI.COMM_WORLD

# One GPU per PE on each node, or a shared GPU when only one is visible. The CUDA context must exist before
# nvshmem.init().
local_rank = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED).Get_rank()
wp.set_device(f"cuda:{local_rank}" if wp.get_cuda_device_count() > 1 else "cuda:0")
device = wp.get_device()
nvshmem.init(mpi_comm=mpi_comm, initializer_method="mpi")
my_pe = nvshmem.my_pe()
n_pes = nvshmem.n_pes()

NX = NY = args.size
N_STEPS = args.iterations
# Steps between convergence checks in the later stages of the guide.
CHECK_EVERY = 10

# Split the NY - 2 interior rows as evenly as possible. Every PE allocates the largest slab plus two ghost rows, since a
# symmetric allocation must have the same size on every PE.
n_interior = NY - 2
chunk_low = n_interior // n_pes
chunk_high = chunk_low + 1
n_pes_low = n_pes * chunk_low + n_pes - n_interior
alloc_rows = chunk_high + 2


def slab_rows(pe):
    """Return the number of rows that ``pe`` owns."""
    return chunk_low if pe < n_pes_low else chunk_high


def slab_start(pe):
    """Return the global row, from 1 to ``NY - 2``, of the first row that ``pe`` owns."""
    return sum(slab_rows(p) for p in range(pe)) + 1


# The values that the step loops pass to the kernels, in the order of the guide's names for them: num_local_rows,
# top_pe, top_halo_row, bottom_pe, bottom_halo_row. The first owned row goes into the bottom ghost row of the PE above,
# which follows that PE's own rows, and the last owned row into the top ghost row 0 of the PE below. The step loops
# unpack these values inside a function: module globals of those names would enter the hash of every kernel with a
# parameter of the same name, which would then differ between PEs.
top_neighbor = (my_pe - 1) % n_pes
bottom_neighbor = (my_pe + 1) % n_pes
decomposition = (slab_rows(my_pe), top_neighbor, slab_rows(top_neighbor) + 1, bottom_neighbor, 0)

# Symmetric allocation is collective: every PE allocates the same arrays in the same order and frees them at the end.
symmetric_buffers = []


def symmetric(shape, dtype):
    """Allocate a zeroed array on the NVSHMEM symmetric heap. Every PE must call this in the same order."""
    buf = nvshmem.buffer(int(np.prod(shape)) * wp.types.type_size_in_bytes(dtype))
    symmetric_buffers.append(buf)
    arr = wp.array(ptr=int(buf.handle), shape=shape, dtype=dtype, device=device)
    arr.zero_()
    return arr


# Each field holds this PE's rows from the top ghost row down; the rows after the bottom ghost row of a PE with fewer
# rows stay unused.
a = symmetric(alloc_rows * NX, wp.float32)
a_new = symmetric(alloc_rows * NX, wp.float32)
# The later stages of the guide use these. flags[0] and flags[1] hold the step that reads the rows that last arrived in
# the top and bottom ghost rows, and partials holds the partial sums of the residual of every PE.
flags = symmetric(2, wp.uint64)
partials = symmetric(n_pes, wp.float32)

# The host writes the initial field into both arrays: zero, with the Dirichlet boundary values sin(2 pi y / (NY - 1))
# in the left and right columns of global row y.
initial = np.zeros((alloc_rows, NX), dtype=np.float32)
global_rows = slab_start(my_pe) - 1 + np.arange(slab_rows(my_pe) + 2)
initial[: len(global_rows), 0] = initial[: len(global_rows), -1] = np.sin(2.0 * np.pi * global_rows / (NY - 1))
a.assign(initial.ravel())
a_new.assign(initial.ravel())

# Every PE has initialized its fields before any PE puts into them.
wp.synchronize_device(device)
nvshmem_bindings.barrier_all()
# docs: setup end

if my_pe == 0:
    print(f"Jacobi: {n_pes} PEs, {NY}x{NX} mesh, {N_STEPS} iterations", flush=True)


# docs: stage-3-kernel begin
@wp.kernel
def jacobi_step(
    a: wp.array[wp.float32],
    a_new: wp.array[wp.float32],
    nx: int,
    num_local_rows: int,
    top_pe: int,
    top_halo_row: int,
    bottom_pe: int,
    bottom_halo_row: int,
):
    """5-point stencil update + NVSHMEM boundary exchange."""
    tid = wp.tid()
    num_interior = num_local_rows * (nx - 2)
    if tid >= num_interior:
        return

    iy = tid // (nx - 2) + 1  # owned rows: 1..num_local_rows
    ix = tid % (nx - 2) + 1  # interior columns: 1..nx-2

    new_val = 0.25 * (a[(iy - 1) * nx + ix] + a[(iy + 1) * nx + ix] + a[iy * nx + (ix - 1)] + a[iy * nx + (ix + 1)])
    a_new[iy * nx + ix] = new_val

    # Send top boundary row to top neighbor's bottom halo
    if iy == 1:
        wp.nvshmem_p(a_new, top_halo_row * nx + ix, new_val, top_pe)

    # Send bottom boundary row to bottom neighbor's top halo
    if iy == num_local_rows:
        wp.nvshmem_p(a_new, bottom_halo_row * nx + ix, new_val, bottom_pe)

    if iy == 1 or iy == num_local_rows:
        # A put is complete only after the issuing thread's quiet (or a device barrier); a host barrier orders PEs,
        # not device-issued puts.
        wp.nvshmem_quiet()


# docs: stage-3-kernel end


@wp.kernel
def residual(
    a: wp.array[wp.float32],
    a_new: wp.array[wp.float32],
    nx: int,
    n_rows: int,
    partial: wp.array[wp.float32],
):
    # Sum the squared update over the owned interior cells of this PE.
    tid = wp.tid()
    iy = tid // (nx - 2) + 1
    ix = tid % (nx - 2) + 1

    i = iy * nx + ix
    d = a_new[i] - a[i]
    wp.atomic_add(partial, 0, d * d)


def solve(a, a_new):
    """Run ``N_STEPS`` Jacobi steps from the field in ``a``.

    Returns:
        The sum of the squared update of the last step over the owned interior cells of this PE.
    """
    num_local_rows, top_pe, top_halo_row, bottom_pe, bottom_halo_row = decomposition
    # docs: stage-3-loop begin
    for _step in range(N_STEPS):
        wp.launch(
            jacobi_step,
            dim=num_local_rows * (NX - 2),
            inputs=[a, a_new, NX, num_local_rows, top_pe, top_halo_row, bottom_pe, bottom_halo_row],
            device=device,
        )
        # The host waits until this PE has finished the step, then until every PE has: the next step reads the rows
        # that the neighbors put in this one.
        wp.synchronize_device(device)
        nvshmem_bindings.barrier_all()
        a, a_new = a_new, a
    # docs: stage-3-loop end

    # After the last swap, a holds the output of the last step and a_new its input.
    partial = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(residual, dim=num_local_rows * (NX - 2), inputs=[a_new, a, NX, num_local_rows, partial], device=device)
    return float(partial.numpy()[0])


norm = np.sqrt(mpi_comm.allreduce(solve(a, a_new), op=MPI.SUM))
if my_pe == 0:
    print(f"l2 norm after {N_STEPS} iterations: {norm:.6f}", flush=True)

# Freeing symmetric memory is collective, and no PE may free it while another PE still uses it.
wp.synchronize_device(device)
nvshmem_bindings.barrier_all()
for buf in symmetric_buffers:
    nvshmem.free(buf)
nvshmem.finalize()
MPI.Finalize()
