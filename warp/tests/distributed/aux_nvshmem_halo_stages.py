# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one stage of the NVSHMEM halo-exchange guide on two PEs and check it against one GPU.

This program is launched by ``TestNVSHMEMHaloExchange`` through ``mpirun -np 2``, with the stage, from 3 to 7, in
``WP_TEST_HALO_STAGE``. The stage numbers are the numbers of the guide's sections, which the rendered guide does not
show. The documentation includes the kernels and step loops of stages 4 to 7, and the on-stream
barrier that ends every stage, from the blocks between the ``# docs: <name> begin`` and ``# docs: <name> end`` lines;
stage 3 runs a copy of the kernel of ``example_jacobi_nvshmem_device.py``. Each stage runs the Jacobi solver of that
example on a small periodic domain whose 13 rows split 6 and 7 between the PEs, so the PEs write into ghost rows at
different offsets. The owned rows of the final field must match the same steps run over the full domain on one GPU.

With ``WP_TEST_HALO_DELAY=1`` in the environment, PE 1 spins on the stream that pushes its rows before each push in
stages 5 to 7, so PE 0 reaches its waits before the rows it waits for have arrived.
"""

import os

import numpy as np
import nvshmem.core as nvshmem
from mpi4py import MPI
from nvshmem.bindings import nvshmem as nvshmem_bindings

import warp as wp
from warp.types import type_size_in_bytes

mpi_comm = MPI.COMM_WORLD

wp.config.cuda_output = "cubin"
wp.config.log_level = wp.LOG_WARNING

STAGE = int(os.environ["WP_TEST_HALO_STAGE"])
DELAY = os.environ.get("WP_TEST_HALO_DELAY") == "1"

local_rank = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED).Get_rank()
wp.set_device(f"cuda:{local_rank}" if wp.get_cuda_device_count() > 1 else "cuda:0")
device = wp.get_device()
nvshmem.init(mpi_comm=mpi_comm, initializer_method="mpi")
my_pe = nvshmem.my_pe()
n_pes = nvshmem.n_pes()

NX = 34
NY = 15
N_STEPS = 20
CHECK_EVERY = 5
SEED = 42
# Long enough for PE 0 to reach its waits well before a delayed push from PE 1 arrives.
SPIN_ITERATIONS = 2_000_000

# The decomposition of example_jacobi_nvshmem_device.py: the NY - 2 interior rows of the periodic domain, split as
# evenly as possible. Every PE allocates the largest slab plus two ghost rows.
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


# The values that the stage loops pass to the kernels, in the order of the guide's names for them: num_local_rows,
# top_pe, top_halo_row, bottom_pe, bottom_halo_row. The first owned row goes into the bottom ghost row of the PE above,
# which follows that PE's own rows, and the last owned row into the top ghost row 0 of the PE below. They are not
# globals of those names, because such globals enter the hash of every kernel with a parameter of the same name, which
# would then differ between PEs.
top_neighbor = (my_pe - 1) % n_pes
bottom_neighbor = (my_pe + 1) % n_pes
decomposition = (slab_rows(my_pe), top_neighbor, slab_rows(top_neighbor) + 1, bottom_neighbor, 0)

symmetric_buffers = []


def symmetric(shape, dtype):
    """Allocate a zeroed symmetric array. Every PE must call this in the same order."""
    buf = nvshmem.buffer(int(np.prod(shape)) * type_size_in_bytes(dtype))
    symmetric_buffers.append(buf)
    arr = wp.array(ptr=int(buf.handle), shape=shape, dtype=dtype, device=device)
    arr.zero_()
    return arr


def sync():
    """Wait for this PE's device work, then for every PE on the host."""
    wp.synchronize_device(device)
    nvshmem_bindings.barrier_all()


# Each field holds this PE's rows from the top ghost row down; the rows after the bottom ghost row of a PE with fewer
# rows stay unused. flags[0] and flags[1] hold the step that reads the rows that last arrived in the top and bottom
# ghost rows.
fields = [symmetric(alloc_rows * NX, wp.float32) for _ in range(2)]
flags = symmetric(2, wp.uint64)
partials = symmetric(n_pes, wp.float32)

# The full periodic field: the interior rows of the domain, with the sin() boundary values of
# example_jacobi_nvshmem_device.py in the edge columns. The interior starts random rather than zero, so that a step
# that reads a stale ghost row changes every column it reaches.
initial = np.zeros((n_interior, NX), dtype=np.float32)
initial[:, 0] = initial[:, -1] = np.sin(2.0 * np.pi * np.arange(1, n_interior + 1) / (NY - 1))
initial[:, 1:-1] = np.random.default_rng(SEED).standard_normal((n_interior, NX - 2))


def initialize_fields():
    """Write this PE's rows of the initial field, ghost rows included, into both fields."""
    n_rows = decomposition[0]
    local = np.zeros((alloc_rows, NX), dtype=np.float32)
    local[: n_rows + 2] = initial[(slab_start(my_pe) - 2 + np.arange(n_rows + 2)) % n_interior]
    for field in fields:
        field.assign(local.ravel())


initialize_fields()
# Every PE has initialized its fields before any PE pushes into them.
sync()


@wp.func
def busy_work(iterations: int) -> float:
    """Keep the calling thread busy for ``iterations`` dependent steps, and return a value that is never negative."""
    x = float(0.0)
    for _i in range(iterations):
        x = x * 0.5 + 1.0
    return x


@wp.kernel
def spin_kernel(iterations: int, sink: wp.array[wp.float32], runs: wp.array[wp.int32]):
    """Keep one thread busy and count the run; the store keeps the busy work from being optimized out."""
    sink[0] = busy_work(iterations)
    wp.atomic_add(runs, 0, 1)


spin_sink = wp.zeros(1, dtype=wp.float32, device=device)
spin_runs = wp.zeros(1, dtype=wp.int32, device=device)


def delay_pushes():
    """Make every launch of ``push_rows`` on this PE start with ``spin_kernel`` on the stream of the push.

    The stage loops launch ``push_rows`` through ``wp.launch``, so wrapping it delays each push without adding test
    code to the loops that the documentation includes.
    """
    launch = wp.launch

    def launch_after_spin(kernel, *args, **kwargs):
        if kernel is push_rows:
            launch(
                spin_kernel,
                dim=1,
                inputs=[SPIN_ITERATIONS, spin_sink, spin_runs],
                device=device,
                stream=kwargs.get("stream"),
            )
        return launch(kernel, *args, **kwargs)

    wp.launch = launch_after_spin


@wp.kernel
def jacobi_step_reference(u: wp.array[wp.float32], u_new: wp.array[wp.float32], nx: int, ny: int):
    """Apply the Jacobi stencil to every row of a periodic field that has no ghost rows, keeping the edge columns."""
    tid = wp.tid()
    iy = tid // (nx - 2)
    ix = tid % (nx - 2) + 1
    iy_up = (iy + ny - 1) % ny
    iy_down = (iy + 1) % ny

    u_new[iy * nx + ix] = 0.25 * (u[iy_up * nx + ix] + u[iy_down * nx + ix] + u[iy * nx + ix - 1] + u[iy * nx + ix + 1])


def reference():
    """Run ``N_STEPS`` steps over the full field on one GPU.

    Returns:
        The final full field, and the L2 norm of the update at every step that stage 7 checks.
    """
    u = wp.array(initial.ravel(), dtype=wp.float32, device=device)
    u_new = wp.clone(u)
    norms = []
    for k in range(N_STEPS):
        wp.launch(jacobi_step_reference, dim=n_interior * (NX - 2), inputs=[u, u_new, NX, n_interior], device=device)
        if (k + 1) % CHECK_EVERY == 0:
            delta = u_new.numpy().astype(np.float64) - u.numpy()
            norms.append(np.sqrt(np.sum(delta**2)))
        u, u_new = u_new, u
    return u.numpy().reshape(n_interior, NX), norms


def check(actual, expected, what, rtol=1e-5, atol=1e-6):
    """Compare on this PE; on a mismatch, abort every PE instead of leaving the other one in a collective."""
    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as error:
        print(f"PE {my_pe}: stage {STAGE}: {what}: FAILED\n{error}", flush=True)
        mpi_comm.Abort(1)


# Stage 3: the kernel of example_jacobi_nvshmem_device.py, which the guide includes from there, with a host barrier
# after each step.
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


def stage_3():
    a, a_new = fields
    num_local_rows, top_pe, top_halo_row, bottom_pe, bottom_halo_row = decomposition
    update_size = num_local_rows * (NX - 2)
    for _it in range(N_STEPS):
        wp.launch(
            jacobi_step,
            dim=update_size,
            inputs=[a, a_new, NX, num_local_rows, top_pe, top_halo_row, bottom_pe, bottom_halo_row],
            device=device,
        )
        wp.synchronize_device(device)
        nvshmem_bindings.barrier_all()
        a, a_new = a_new, a
    return a


# docs: stage-4-kernels begin
@wp.kernel
def barrier_all():
    # NVSHMEM's device-side barrier is a collective of one thread per PE: launch with dim=1.
    wp.nvshmem_barrier_all()


# docs: stage-4-kernels end


def stage_4():
    a, a_new = fields
    num_local_rows, top_pe, top_halo_row, bottom_pe, bottom_halo_row = decomposition
    # docs: stage-4-loop begin
    for _step in range(N_STEPS):
        wp.launch(
            jacobi_step,
            dim=num_local_rows * (NX - 2),
            inputs=[a, a_new, NX, num_local_rows, top_pe, top_halo_row, bottom_pe, bottom_halo_row],
            device=device,
        )
        # The stream waits until every PE has finished the step and its puts have landed. The host does not wait.
        wp.launch(barrier_all, dim=1, device=device)
        a, a_new = a_new, a
    # docs: stage-4-loop end
    return a


# docs: stage-5-kernels begin
@wp.kernel
def jacobi_rows(
    a: wp.array[wp.float32],
    a_new: wp.array[wp.float32],
    flags: wp.array[wp.uint64],  # flags[0]: top ghost row, flags[1]: bottom ghost row
    nx: int,
    n_rows: int,
    step: int,
):
    tid = wp.tid()
    iy = tid // (nx - 2) + 1  # owned rows: 1..n_rows
    ix = tid % (nx - 2) + 1  # interior columns: 1..nx-2

    # Only the threads that read a ghost row wait for the neighbor's row of this step to arrive. The host wrote the
    # ghost rows of step 0, so the waits of step 0 pass on flags that start at zero.
    if iy == 1:
        wp.nvshmem_signal_wait_until(flags, 0, wp.NVSHMEM_CMP_GE, step)
    if iy == n_rows:
        wp.nvshmem_signal_wait_until(flags, 1, wp.NVSHMEM_CMP_GE, step)

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
    next_step: int,  # the step that reads these rows
):
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


# docs: stage-5-kernels end


def stage_5():
    a, a_new = fields
    num_local_rows, top_pe, top_halo_row, bottom_pe, bottom_halo_row = decomposition
    # docs: stage-5-loop begin
    for step in range(N_STEPS):
        wp.launch(
            jacobi_rows,
            dim=num_local_rows * (NX - 2),
            inputs=[a, a_new, flags, NX, num_local_rows, step],
            device=device,
        )
        # One thread per row: thread 0 pushes the first owned row, thread 1 the last.
        wp.launch(
            push_rows,
            dim=2,
            inputs=[a_new, flags, NX, num_local_rows, top_pe, top_halo_row, bottom_pe, bottom_halo_row, step + 1],
            device=device,
        )
        a, a_new = a_new, a
    # docs: stage-5-loop end
    return a


# docs: stage-6-kernels begin
@wp.kernel
def boundary_rows(
    a: wp.array[wp.float32],
    a_new: wp.array[wp.float32],
    flags: wp.array[wp.uint64],
    nx: int,
    n_rows: int,
    step: int,
):
    tid = wp.tid()
    side = tid // (nx - 2)  # 0: row 1, which reads the top ghost row; 1: row n_rows, which reads the bottom one
    iy = 1 + side * (n_rows - 1)
    ix = tid % (nx - 2) + 1

    # flags[side] guards the ghost row that this thread reads.
    wp.nvshmem_signal_wait_until(flags, side, wp.NVSHMEM_CMP_GE, step)

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


# docs: stage-6-kernels end


def stage_6():
    a, a_new = fields
    num_local_rows, top_pe, top_halo_row, bottom_pe, bottom_halo_row = decomposition
    # docs: stage-6-loop begin
    # Compute on the device's current stream, which the host's copies such as a.numpy() also use.
    compute = wp.get_stream(device)
    comm_stream = wp.Stream(device)
    boundary_done = wp.Event(device)
    push_done = wp.Event(device)

    for step in range(N_STEPS):
        wp.launch(
            boundary_rows,
            dim=2 * (NX - 2),
            inputs=[a, a_new, flags, NX, num_local_rows, step],
            stream=compute,
        )
        compute.record_event(boundary_done)

        # The push waits for the boundary rows only.
        comm_stream.wait_event(boundary_done)
        wp.launch(
            push_rows,
            dim=2,
            inputs=[a_new, flags, NX, num_local_rows, top_pe, top_halo_row, bottom_pe, bottom_halo_row, step + 1],
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
        # Join: the push reads rows 1 and n_rows of a_new, which the step after next overwrites. The join orders that
        # on this PE's own streams without relying on the neighbors' signals, and a graph capture must join
        # comm_stream back into the capture stream before it ends.
        compute.wait_event(push_done)
        a, a_new = a_new, a
    # docs: stage-6-loop end
    return a


# docs: stage-7-kernels begin
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


# docs: stage-7-kernels end


def stage_7():
    a, a_new = fields
    num_local_rows, top_pe, top_halo_row, bottom_pe, bottom_halo_row = decomposition
    # docs: stage-7-loop begin
    # The current stream, which also runs partial.zero_() and the copy of partials.numpy() in order with the steps.
    compute = wp.get_stream(device)
    comm_stream = wp.Stream(device)
    boundary_done = wp.Event(device)
    push_done = wp.Event(device)
    partial = wp.zeros(1, dtype=wp.float32, device=device)
    norms = []  # L2 norm of the update at every check

    for step in range(N_STEPS):
        wp.launch(
            boundary_rows,
            dim=2 * (NX - 2),
            inputs=[a, a_new, flags, NX, num_local_rows, step],
            stream=compute,
        )
        compute.record_event(boundary_done)

        comm_stream.wait_event(boundary_done)
        wp.launch(
            push_rows,
            dim=2,
            inputs=[a_new, flags, NX, num_local_rows, top_pe, top_halo_row, bottom_pe, bottom_halo_row, step + 1],
            stream=comm_stream,
        )
        comm_stream.record_event(push_done)

        wp.launch(
            interior_rows,
            dim=(num_local_rows - 2) * (NX - 2),
            inputs=[a, a_new, NX, num_local_rows],
            stream=compute,
        )
        compute.wait_event(push_done)

        if (step + 1) % CHECK_EVERY == 0:
            partial.zero_()
            wp.launch(
                residual,
                dim=num_local_rows * (NX - 2),
                inputs=[a, a_new, NX, num_local_rows, partial],
                stream=compute,
            )
            # Every PE reaches this barrier only after its host has read the previous sums, so a fast PE cannot
            # write its next sums into partials while a slow PE still reads the previous ones.
            wp.launch(barrier_all, dim=1, stream=compute)
            wp.launch(exchange_partials, dim=1, inputs=[partial, partials], stream=compute)
            # After this barrier, every PE holds the partial sums of all PEs.
            wp.launch(barrier_all, dim=1, stream=compute)
            norms.append(np.sqrt(partials.numpy().sum()))
        a, a_new = a_new, a
    # docs: stage-7-loop end

    # The exchanged sums match the one-GPU residual at every check. partials and partial still hold the sums of the
    # last check, and the exchanged ones add up to the local ones summed through MPI.
    check(norms, expected_norms, "residual")
    check(partials.numpy().sum(), mpi_comm.allreduce(float(partial.numpy()[0]), MPI.SUM), "allreduce", atol=0.0)
    return a


def main():
    delayed = DELAY and STAGE >= 5 and my_pe == 1
    if delayed:
        delay_pushes()

    final = {3: stage_3, 4: stage_4, 5: stage_5, 6: stage_6, 7: stage_7}[STAGE]()

    # docs: on-stream-barrier begin
    # A barrier across all PEs, enqueued on Warp's current stream. The host does not wait for it.
    stream = wp.get_stream(device)
    nvshmem.barrier_all(stream=stream)
    # docs: on-stream-barrier end

    num_local_rows = decomposition[0]
    row_offset = slab_start(my_pe) - 1
    owned = final.numpy().reshape(alloc_rows, NX)[1 : num_local_rows + 1]
    check(owned, expected_field[row_offset : row_offset + num_local_rows], "field")
    if delayed:
        # Every push of the stage ran after a spin, so the delay cannot have been bypassed.
        wp.synchronize_device(device)
        check(spin_runs.numpy()[0], N_STEPS, "spin before every push", rtol=0, atol=0)
    print(f"PE {my_pe}: stage {STAGE}: PASSED", flush=True)


# Every kernel is defined by now, so the module loads once, with the same hash on every PE.
expected_field, expected_norms = reference()
main()

sync()
for buf in symmetric_buffers:
    nvshmem.free(buf)
nvshmem.finalize()
MPI.Finalize()
