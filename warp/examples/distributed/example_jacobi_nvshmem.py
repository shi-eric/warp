# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple distributed Jacobi solver with NVSHMEM in-kernel halo exchange.

Launch:
    mpirun -np 2 uv run --with mpi4py --with nvshmem4py-cu13 \
        python -m warp.examples.distributed.example_jacobi_nvshmem
"""

import os
import tempfile

import nvshmem.core as nvshmem
from mpi4py import MPI
from nvshmem.bindings import nvshmem as _nvshmem_bindings

# Must set WARP_CACHE_ROOT before importing Warp (read at import time)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()
os.environ["WARP_CACHE_ROOT"] = os.path.join(tempfile.gettempdir(), f"warp_nvshmem_rank{rank}")

import warp as wp  # noqa: E402

wp.config.cuda_output = "cubin"
wp.config.quiet = True

# Select GPU based on local rank (not global rank) to handle multi-node correctly
local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
local_rank = local_comm.Get_rank()
num_cuda_devices = wp.get_cuda_device_count()
if num_cuda_devices > 1:
    wp.set_device(f"cuda:{local_rank}")
else:
    wp.set_device("cuda:0")
device = wp.get_device()

# CUDA context must exist before nvshmem.init()
nvshmem.init(mpi_comm=comm, initializer_method="mpi")
my_pe = nvshmem.my_pe()
n_pes = nvshmem.n_pes()

# ----- Problem setup -----
NX = 512
NY = 512
N_ITER = 1000

# Domain decomposition: distribute (NY-2) interior rows across PEs
total_interior = NY - 2
chunk_low = total_interior // n_pes
chunk_high = chunk_low + 1
n_ranks_low = n_pes * chunk_low + n_pes - total_interior

if my_pe < n_ranks_low:
    num_local_rows = chunk_low
    iy_start_global = my_pe * chunk_low + 1
else:
    num_local_rows = chunk_high
    iy_start_global = n_ranks_low * chunk_low + (my_pe - n_ranks_low) * chunk_high + 1

# All PEs allocate the same size (NVSHMEM symmetric requirement)
alloc_rows = chunk_high + 2  # max chunk + 2 halo rows
alloc_size = alloc_rows * NX

# Neighbor PEs (periodic)
top_pe = (my_pe - 1) if my_pe > 0 else (n_pes - 1)
bottom_pe = (my_pe + 1) % n_pes

# Where to write on neighbor:
# Top neighbor's bottom halo = row (chunk_high + 1)
# But top neighbor may have fewer rows. Compute its actual row count.
if top_pe < n_ranks_low:
    top_pe_rows = chunk_low
else:
    top_pe_rows = chunk_high
top_halo_row = top_pe_rows + 1  # bottom halo of top neighbor

# Bottom neighbor's top halo = row 0
bottom_halo_row = 0

if my_pe == 0:
    print(f"Jacobi: {n_pes} PEs, {NY}x{NX} mesh, {N_ITER} iterations", flush=True)


# ----- Kernels -----
@wp.kernel
def init_boundaries(
    a: wp.array(dtype=wp.float32),
    a_new: wp.array(dtype=wp.float32),
    nx: int,
    ny_global: int,
    offset: int,
    num_rows: int,
):
    """Set sin() Dirichlet BCs on left and right edges."""
    iy = wp.tid()
    if iy >= num_rows + 2:
        return
    y_global = iy + offset - 1  # -1 because row 0 is top halo
    val = wp.sin(2.0 * wp.PI * wp.float32(y_global) / wp.float32(ny_global - 1))
    a[iy * nx] = val
    a_new[iy * nx] = val
    a[iy * nx + nx - 1] = val
    a_new[iy * nx + nx - 1] = val


@wp.kernel
def jacobi_step(
    a: wp.array(dtype=wp.float32),
    a_new: wp.array(dtype=wp.float32),
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
        wp.nvshmem_float_p(a_new, top_halo_row * nx + ix, new_val, top_pe)

    # Send bottom boundary row to bottom neighbor's top halo
    if iy == num_local_rows:
        wp.nvshmem_float_p(a_new, bottom_halo_row * nx + ix, new_val, bottom_pe)


# ----- Allocate and initialize -----
a = wp.zeros(alloc_size, dtype=wp.float32, device=device, symmetric=True)
a_new = wp.zeros(alloc_size, dtype=wp.float32, device=device, symmetric=True)

wp.launch(
    init_boundaries,
    dim=alloc_rows,
    inputs=[a, a_new, NX, NY, iy_start_global, num_local_rows],
    device=device,
)
wp.synchronize_device(device)
_nvshmem_bindings.barrier_all()

# ----- Jacobi iteration -----
import time  # noqa: E402

update_size = num_local_rows * (NX - 2)

_nvshmem_bindings.barrier_all()
t_start = time.perf_counter()

for _it in range(N_ITER):
    wp.launch(
        jacobi_step,
        dim=update_size,
        inputs=[a, a_new, NX, num_local_rows, top_pe, top_halo_row, bottom_pe, bottom_halo_row],
        device=device,
    )
    wp.synchronize_device(device)
    _nvshmem_bindings.barrier_all()

    # Swap
    a, a_new = a_new, a

wp.synchronize_device(device)
_nvshmem_bindings.barrier_all()
t_end = time.perf_counter()

if my_pe == 0:
    print(f"Completed {N_ITER} iterations in {t_end - t_start:.3f} s", flush=True)

nvshmem.finalize()
MPI.Finalize()
