# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distributed Jacobi solver with NVSHMEM signals, overlapped communication, and a CUDA graph.

This example solves the same problem as ``example_jacobi_nvshmem_device.py`` with the halo exchange that ends the guide:
https://nvidia.github.io/warp/stable/user_guide/distributed/nvshmem_halo_exchange.html

It combines the later stages of the guide:

- Each step pushes this PE's first and last rows as whole rows with ``wp.nvshmem_put_signal()``, and the next step
  waits for the signal with ``wp.nvshmem_signal_wait_until()`` in the threads that read a ghost row. No step waits at
  a barrier.
- The push runs on a second stream while the compute stream updates the interior rows.
- Every ``CHECK_EVERY`` steps, each PE sums its squared update on the device and puts the sum into the ``partials``
  array of every PE, so each PE reads the L2 norm of the update from its own memory.
- A block of ``CHECK_EVERY`` steps and its residual is captured into a CUDA graph and replayed. The step number that
  the waits and signals compare against is kept in a device array that each step increments, so every replay of the
  same graph continues where the last one stopped.

``--iterations`` must be a multiple of ``CHECK_EVERY`` (10). PE 0 prints the L2 norm of the update after every 100
steps and after the last one.

Launch:
    mpirun -np 2 uv run --with mpi4py --with nvshmem4py-cu13 --with "nvidia-nvshmem-cu13>=3.9" \
        python -m warp.examples.distributed.example_jacobi_nvshmem_overlap [--iterations N] [--size N]
"""

import argparse

import numpy as np
import nvshmem.core as nvshmem
from mpi4py import MPI
from nvshmem.bindings import nvshmem as nvshmem_bindings

import warp as wp

# Steps between convergence checks, and the number of steps in the captured block. It is even, so the two fields have
# the same roles after a block as before it, and every replay of the block can use the same graph.
CHECK_EVERY = 10

parser = argparse.ArgumentParser(description="Distributed Jacobi solver with overlapped NVSHMEM halo exchange.")
parser.add_argument(
    "--iterations", type=int, default=1000, help=f"Number of Jacobi steps, a multiple of {CHECK_EVERY}."
)
parser.add_argument("--size", type=int, default=512, help="Number of rows and columns of the square mesh.")
args = parser.parse_args()
if args.iterations <= 0 or args.iterations % CHECK_EVERY != 0:
    parser.error(f"--iterations must be a positive multiple of {CHECK_EVERY}, got {args.iterations}")

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

# Split the NY - 2 interior rows as evenly as possible. Every PE allocates the largest slab plus two ghost rows, since a
# symmetric allocation must have the same size on every PE.
n_interior = NY - 2
chunk_low = n_interior // n_pes
chunk_high = chunk_low + 1
n_pes_low = n_pes * chunk_low + n_pes - n_interior
alloc_rows = chunk_high + 2
# The interior kernel updates rows 2 to n_rows - 1, so every PE needs at least two rows.
if chunk_low < 2:
    parser.error(f"--size {args.size} gives some of the {n_pes} PEs fewer than 2 rows; use at least {2 * n_pes + 2}")


def slab_rows(pe):
    """Return the number of rows that ``pe`` owns."""
    return chunk_low if pe < n_pes_low else chunk_high


def slab_start(pe):
    """Return the global row, from 1 to ``NY - 2``, of the first row that ``pe`` owns."""
    return sum(slab_rows(p) for p in range(pe)) + 1


# The values that the step loop passes to the kernels: num_local_rows, top_pe, top_halo_row, bottom_pe,
# bottom_halo_row. The first owned row goes into the bottom ghost row of the PE above, which follows that PE's own
# rows, and the last owned row into the top ghost row 0 of the PE below. The step loop unpacks these values inside a
# function: module globals of those names would enter the hash of every kernel with a parameter of the same name, which
# would then differ between PEs.
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
# rows stay unused. flags[0] and flags[1] hold the step that reads the rows that last arrived in the top and bottom
# ghost rows, and partials holds the partial sums of the residual of every PE.
a = symmetric(alloc_rows * NX, wp.float32)
a_new = symmetric(alloc_rows * NX, wp.float32)
flags = symmetric(2, wp.uint64)
partials = symmetric(n_pes, wp.float32)

# The host writes the initial field into both arrays: zero, with the Dirichlet boundary values sin(2 pi y / (NY - 1))
# in the left and right columns of global row y.
initial = np.zeros((alloc_rows, NX), dtype=np.float32)
global_rows = slab_start(my_pe) - 1 + np.arange(slab_rows(my_pe) + 2)
initial[: len(global_rows), 0] = initial[: len(global_rows), -1] = np.sin(2.0 * np.pi * global_rows / (NY - 1))
a.assign(initial.ravel())
a_new.assign(initial.ravel())

# Every PE has initialized its fields before any PE pushes into them.
wp.synchronize_device(device)
nvshmem_bindings.barrier_all()

if my_pe == 0:
    print(f"Jacobi: {n_pes} PEs, {NY}x{NX} mesh, {N_STEPS} iterations", flush=True)


# docs: kernels begin
@wp.kernel
def boundary_rows(
    a: wp.array[wp.float32],
    a_new: wp.array[wp.float32],
    flags: wp.array[wp.uint64],
    nx: int,
    n_rows: int,
    step_counter: wp.array[wp.int32],
):
    tid = wp.tid()
    side = tid // (nx - 2)  # 0: row 1, which reads the top ghost row; 1: row n_rows, which reads the bottom one
    iy = 1 + side * (n_rows - 1)
    ix = tid % (nx - 2) + 1

    # flags[side] guards the ghost row that this thread reads. The host wrote the ghost rows of step 0, so the waits
    # of step 0 pass on flags that start at zero.
    wp.nvshmem_signal_wait_until(flags, side, wp.NVSHMEM_CMP_GE, wp.uint64(step_counter[0]))

    i = iy * nx + ix
    a_new[i] = 0.25 * (a[i - nx] + a[i + nx] + a[i - 1] + a[i + 1])


@wp.kernel
def interior_rows(a: wp.array[wp.float32], a_new: wp.array[wp.float32], nx: int, n_rows: int):
    # Rows 2 to n_rows - 1 read only owned rows, so they wait for nothing.
    tid = wp.tid()
    iy = tid // (nx - 2) + 2
    ix = tid % (nx - 2) + 1

    i = iy * nx + ix
    a_new[i] = 0.25 * (a[i - nx] + a[i + nx] + a[i - 1] + a[i + 1])


@wp.kernel
def push_rows(
    a_new: wp.array[wp.float32],
    flags: wp.array[wp.uint64],
    nx: int,
    n_rows: int,
    top_pe: int,
    top_halo_row: int,
    bottom_pe: int,
    bottom_halo_row: int,
    step_counter: wp.array[wp.int32],
):
    # The step that reads these rows.
    next_step = wp.uint64(step_counter[0] + 1)
    # Each thread copies one whole row, then sets the receiver's flag once, after the last element has arrived. A
    # flag set per element would let the receiver read the row as soon as the first element arrived.
    if wp.tid() == 0:
        # First owned row -> bottom ghost row of the PE above, which its flags[1] guards.
        wp.nvshmem_put_signal(
            a_new[top_halo_row * nx : (top_halo_row + 1) * nx],
            a_new[nx : 2 * nx],
            flags,
            1,
            next_step,
            wp.NVSHMEM_SIGNAL_SET,
            top_pe,
        )
    else:
        # Last owned row -> top ghost row of the PE below, which its flags[0] guards.
        wp.nvshmem_put_signal(
            a_new[bottom_halo_row * nx : (bottom_halo_row + 1) * nx],
            a_new[n_rows * nx : (n_rows + 1) * nx],
            flags,
            0,
            next_step,
            wp.NVSHMEM_SIGNAL_SET,
            bottom_pe,
        )
    # Complete the push before the kernel ends, so the push of the next step cannot overtake it and set the flag back.
    wp.nvshmem_quiet()


@wp.kernel
def advance_step(step_counter: wp.array[wp.int32]):
    # The last launch of a step, after the push has read the counter.
    step_counter[0] = step_counter[0] + 1


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


@wp.kernel
def exchange_partials(partial: wp.array[wp.float32], partials: wp.array[wp.float32]):
    # One thread writes this PE's partial sum into slot my_pe of partials on every PE, this one included.
    my_pe = wp.nvshmem_my_pe()
    for pe in range(wp.nvshmem_n_pes()):
        wp.nvshmem_p(partials, my_pe, partial[0], pe)
    wp.nvshmem_quiet()


@wp.kernel
def barrier_all():
    # NVSHMEM's device-side barrier is a collective of one thread per PE: launch with dim=1.
    wp.nvshmem_barrier_all()


# docs: kernels end


# docs: step begin
# The device's current stream, which also runs partial.zero_() and the copy of partials.numpy() in order with the
# steps.
compute = wp.get_stream(device)
comm_stream = wp.Stream(device)
boundary_done = wp.Event(device)
push_done = wp.Event(device)
# The step that the waits and signals compare against. It lives on the device, so each replay of a captured block
# continues from where the previous block stopped.
step_counter = wp.zeros(1, dtype=wp.int32, device=device)
partial = wp.zeros(1, dtype=wp.float32, device=device)


def run_block(a, a_new):
    """Issue ``CHECK_EVERY`` steps, then the residual of the last step and the exchange of the partial sums."""
    num_local_rows, top_pe, top_halo_row, bottom_pe, bottom_halo_row = decomposition
    for _step in range(CHECK_EVERY):
        wp.launch(
            boundary_rows,
            dim=2 * (NX - 2),
            inputs=[a, a_new, flags, NX, num_local_rows, step_counter],
            stream=compute,
        )
        compute.record_event(boundary_done)

        # The push waits for the boundary rows only.
        comm_stream.wait_event(boundary_done)
        wp.launch(
            push_rows,
            dim=2,
            inputs=[a_new, flags, NX, num_local_rows, top_pe, top_halo_row, bottom_pe, bottom_halo_row, step_counter],
            stream=comm_stream,
        )
        comm_stream.record_event(push_done)

        # The interior rows run while the boundary rows are in flight.
        wp.launch(
            interior_rows,
            dim=(num_local_rows - 2) * (NX - 2),
            inputs=[a, a_new, NX, num_local_rows],
            stream=compute,
        )
        # Join: the push reads rows 1 and n_rows of a_new, which the step after next overwrites, and it reads the step
        # counter, which advance_step increments. The join orders both on this PE's own streams, and a graph capture
        # must join comm_stream back into the capture stream before it ends.
        compute.wait_event(push_done)
        wp.launch(advance_step, dim=1, inputs=[step_counter], stream=compute)
        a, a_new = a_new, a

    # a now holds the output of the last step and a_new its input.
    partial.zero_()
    wp.launch(
        residual,
        dim=num_local_rows * (NX - 2),
        inputs=[a_new, a, NX, num_local_rows, partial],
        stream=compute,
    )
    # Every PE reaches this barrier only after its host has read the previous sums, so a fast PE cannot write its next
    # sums into partials while a slow PE still reads the previous ones.
    wp.launch(barrier_all, dim=1, stream=compute)
    wp.launch(exchange_partials, dim=1, inputs=[partial, partials], stream=compute)
    # After this barrier, every PE holds the partial sums of all PEs.
    wp.launch(barrier_all, dim=1, stream=compute)


# docs: step end


# docs: capture begin
def report_norm(steps_done):
    """Read the L2 norm of the last update from the partial sums of all PEs.

    PE 0 prints it every 100 steps and after the last step.
    """
    # partials.numpy() waits for the block, which ran on the compute stream. Every PE reads the sums after every block,
    # whether or not it prints, because the barrier before the next exchange relies on that.
    norm = np.sqrt(partials.numpy().sum())
    if my_pe == 0 and (steps_done % 100 == 0 or steps_done == N_STEPS):
        print(f"l2 norm after {steps_done} iterations: {norm:.6f}", flush=True)


# The first block runs eagerly. Its first launches load the module and register it with NVSHMEM, which cannot happen
# during a capture.
run_block(a, a_new)
report_norm(CHECK_EVERY)

# The capture records the launches of one block without running them.
with wp.ScopedCapture(stream=compute) as capture:
    run_block(a, a_new)

# Each replay runs the next block.
for block in range(1, N_STEPS // CHECK_EVERY):
    wp.capture_launch(capture.graph, stream=compute)
    report_norm((block + 1) * CHECK_EVERY)
# docs: capture end

# docs: teardown begin
# Freeing symmetric memory is collective, and no PE may free it while another PE still uses it.
wp.synchronize_device(device)
nvshmem_bindings.barrier_all()
for buf in symmetric_buffers:
    nvshmem.free(buf)
nvshmem.finalize()
MPI.Finalize()
# docs: teardown end
