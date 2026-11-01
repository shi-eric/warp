# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Differentiable distributed Jacobi solver with NVSHMEM in-kernel halo exchange.

Each PE runs a fixed number of Jacobi steps on its block of rows and pushes its first and last rows into the halo
rows of its neighbors with ``wp.nvshmem_p()``. The steps are recorded on a tape, and the backward pass computes the
gradient of the sum of squares of the final field's interior, over all PEs, with respect to the initial field.

See the automatic differentiation guide:
https://nvidia.github.io/warp/stable/user_guide/distributed/nvshmem_autodiff.html

Launch:
    mpirun -np 2 uv run --with mpi4py --with nvshmem4py-cu13 --with "nvidia-nvshmem-cu13>=3.9" \
        python -m warp.examples.distributed.example_jacobi_nvshmem_grad
"""

import numpy as np
import nvshmem.core as nvshmem
from mpi4py import MPI
from nvshmem.bindings import nvshmem as _nvshmem_bindings

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()

import warp as wp  # noqa: E402

wp.config.cuda_output = "cubin"
wp.config.log_level = wp.LOG_WARNING

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
NX = 64
NY = 64
N_ITER = 20

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
top_neighbor = (my_pe - 1) if my_pe > 0 else (n_pes - 1)
bottom_neighbor = (my_pe + 1) % n_pes

# Where to write on neighbor:
# Top neighbor's bottom halo = row (its row count + 1); the top neighbor may have fewer rows than this PE.
# Bottom neighbor's top halo = row 0
top_neighbor_rows = chunk_low if top_neighbor < n_ranks_low else chunk_high

# The top_pe, top_halo_row, bottom_pe and bottom_halo_row arguments of jacobi_step. They are not module globals of
# those names, because such globals enter the kernel's hash, which would then differ between PEs.
neighbor_args = [top_neighbor, top_neighbor_rows + 1, bottom_neighbor, 0]

if my_pe == 0:
    print(f"Jacobi gradient: {n_pes} PEs, {NY}x{NX} mesh, {N_ITER} iterations", flush=True)


# ----- Kernels -----
@wp.kernel
def init_boundaries(u: wp.array[wp.float32], nx: int, ny_global: int, offset: int, num_rows: int):
    """Set sin() Dirichlet BCs on left and right edges."""
    iy = wp.tid()
    if iy >= num_rows + 2:
        return
    y_global = iy + offset - 1  # -1 because row 0 is top halo
    val = wp.sin(2.0 * wp.PI * wp.float32(y_global) / wp.float32(ny_global - 1))
    u[iy * nx] = val
    u[iy * nx + nx - 1] = val


@wp.kernel
def jacobi_step(
    u: wp.array[wp.float32],  # field at step k, including ghost rows
    u_new: wp.array[wp.float32],  # field at step k + 1
    nx: int,
    n_rows: int,
    top_pe: int,
    top_halo_row: int,
    bottom_pe: int,
    bottom_halo_row: int,
):
    """5-point stencil update + NVSHMEM boundary exchange."""
    tid = wp.tid()
    iy = tid // (nx - 2) + 1  # owned rows: 1..n_rows
    ix = tid % (nx - 2) + 1  # interior columns: 1..nx-2

    v = 0.25 * (u[(iy - 1) * nx + ix] + u[(iy + 1) * nx + ix] + u[iy * nx + ix - 1] + u[iy * nx + ix + 1])
    u_new[iy * nx + ix] = v

    # Send top boundary row to top neighbor's bottom halo. The adjoint pulls that halo's gradient back from the
    # neighbor, after the neighbor has accumulated it.
    if iy == 1:
        wp.nvshmem_p(u_new, top_halo_row * nx + ix, v, top_pe)

    # Send bottom boundary row to bottom neighbor's top halo
    if iy == n_rows:
        wp.nvshmem_p(u_new, bottom_halo_row * nx + ix, v, bottom_pe)


@wp.kernel
def exchange_halo(
    u: wp.array[wp.float32],
    nx: int,
    n_rows: int,
    top_pe: int,
    top_halo_row: int,
    bottom_pe: int,
    bottom_halo_row: int,
):
    """Push the first and last owned rows of ``u`` into the neighbors' halo rows of ``u``."""
    tid = wp.tid()
    ix = tid % (nx - 2) + 1  # interior columns: 1..nx-2

    # The first nx-2 threads send the top boundary row, the others the bottom boundary row, as in jacobi_step.
    if tid < nx - 2:
        wp.nvshmem_p(u, top_halo_row * nx + ix, u[1 * nx + ix], top_pe)
    else:
        wp.nvshmem_p(u, bottom_halo_row * nx + ix, u[n_rows * nx + ix], bottom_pe)


@wp.kernel
def interior_loss(u: wp.array[wp.float32], nx: int, n_rows: int, loss: wp.array[wp.float32]):
    """Add the sum of squares of the owned interior of ``u`` to ``loss``."""
    tid = wp.tid()
    iy = tid // (nx - 2) + 1
    ix = tid % (nx - 2) + 1
    wp.atomic_add(loss, 0, u[iy * nx + ix] * u[iy * nx + ix])


@wp.kernel
def barrier_all():
    """Synchronize all PEs. Launch with dim=1: the device-side barrier must be called by one thread per PE."""
    wp.nvshmem_barrier_all()


# ----- Allocate and initialize -----
# Symmetric allocation is collective: every PE allocates the same arrays in the same order and frees them explicitly
# at the end.
symmetric_buffers = []


def symmetric_array(shape, dtype, requires_grad=False):
    """Allocate a zeroed symmetric array. Every PE must call this in the same order.

    With ``requires_grad``, the gradient is a second symmetric allocation, so that other PEs can read it in the
    backward pass. ``wp.array(..., requires_grad=True)`` would allocate a gradient that other PEs cannot address.
    """
    nbytes = int(np.prod(shape)) * wp.types.type_size_in_bytes(dtype)
    buf = nvshmem.buffer(nbytes)
    symmetric_buffers.append(buf)
    arr = wp.array(ptr=int(buf.handle), shape=shape, dtype=dtype, device=device)
    arr.zero_()
    if requires_grad:
        grad_buf = nvshmem.buffer(nbytes)
        symmetric_buffers.append(grad_buf)
        arr.grad = wp.array(ptr=int(grad_buf.handle), shape=shape, dtype=dtype, device=device)
        arr.grad.zero_()
    return arr


# One field per step, so that no step overwrites values that the backward pass still needs.
fields = [symmetric_array(alloc_size, wp.float32, requires_grad=True) for _ in range(N_ITER + 1)]
loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

for field in fields:
    wp.launch(
        init_boundaries,
        dim=alloc_rows,
        inputs=[field, NX, NY, iy_start_global, num_local_rows],
        device=device,
    )

# Every PE has initialized its fields before any PE pushes into them.
wp.synchronize_device(device)
_nvshmem_bindings.barrier_all()

# ----- Forward and backward pass -----
update_size = num_local_rows * (NX - 2)

with wp.Tape() as tape:
    # Fill the halo rows of the initial field on the tape, so that the gradient through these copies of a PE's rows
    # flows back to that PE.
    wp.launch(
        exchange_halo,
        dim=2 * (NX - 2),
        inputs=[fields[0], NX, num_local_rows, *neighbor_args],
        device=device,
    )
    wp.launch(barrier_all, dim=1, device=device, record_tape=False)
    for k in range(N_ITER):
        wp.launch(
            jacobi_step,
            dim=update_size,
            inputs=[fields[k], fields[k + 1], NX, num_local_rows, *neighbor_args],
            device=device,
        )
        # The next step reads the rows the neighbors pushed in this one. This synchronization belongs to the forward
        # pass only, so the tape does not record it.
        wp.launch(barrier_all, dim=1, device=device, record_tape=False)
    wp.launch(interior_loss, dim=update_size, inputs=[fields[N_ITER], NX, num_local_rows, loss], device=device)

# Each PE seeds the gradient of its own share of the loss, so together they differentiate the total loss. Warp
# synchronizes all PEs before each adjoint launch of jacobi_step and exchange_halo and after the last one, so no PE
# pulls a halo gradient before the neighbor has finished accumulating it.
tape.backward(loss)

# Rows 1..num_local_rows of the initial field belong to this PE, and their gradient includes the paths through the
# neighbors' halo copies of them.
grad = fields[0].grad.numpy().reshape(alloc_rows, NX)[1 : num_local_rows + 1]
print(
    f"PE {my_pe}: local share of the loss {loss.numpy()[0]:.6f}, "
    f"gradient norm w.r.t. this PE's rows of the initial field {np.linalg.norm(grad):.6f}",
    flush=True,
)

# Freeing symmetric memory is collective and must wait until the backward pass has completed on every PE, which the
# barrier that ends tape.backward() guarantees once the device is synchronized.
wp.synchronize_device(device)
del tape, fields, field
for buf in symmetric_buffers:
    nvshmem.free(buf)
nvshmem.finalize()
MPI.Finalize()
