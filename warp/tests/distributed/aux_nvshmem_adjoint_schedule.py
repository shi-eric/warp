# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-PE test that the NVSHMEM adjoint barriers give the gradients of a one-GPU run.

This program is launched by ``TestNVSHMEMAutodiff.test_schedule`` through ``mpirun -np 2``. It runs the Jacobi
example of the NVSHMEM autodiff design: each PE owns ``n_rows`` rows of a periodic field, pushes its first and last
rows into the ghost rows of its neighbors, and differentiates the sum of squares of its interior after ``N_STEPS``
steps. PE 1 spins before each of its adjoint steps, so an adjoint launch on PE 0 that does not wait for PE 1 pulls
gradients that PE 1 has not accumulated yet. PE 1 also spins inside the kernel that pushes its rows in the first step,
so its last adjoint launch is still pulling gradients from PE 0 after PE 0 has finished its backward pass. The gradient
of the initial field must match the same stencil run over the full domain on one GPU on each of four passes over the
same arrays:

1. The PEs synchronize the forward steps with a barrier kernel.
2. The same again, with no host synchronization across PEs after the first pass.
3. The PEs synchronize the forward steps with signals instead: each step waits for the rows of the previous step.
4. As in pass 3, with the backward pass captured into a CUDA graph and replayed twice.

With ``WP_TEST_NVSHMEM_NO_BARRIER=1`` in the environment, the program turns the barriers off. That is a manual
negative control: the gradients are then expected to mismatch in some runs.
"""

import os

import numpy as np
import nvshmem.core as nvshmem
from mpi4py import MPI
from nvshmem.bindings import nvshmem as nvshmem_bindings

import warp as wp
from warp.types import type_size_in_bytes

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

wp.config.log_level = wp.LOG_WARNING
if os.environ.get("WP_TEST_NVSHMEM_NO_BARRIER") == "1":
    wp.config.nvshmem_adjoint_barrier = False

local_rank = comm.Split_type(MPI.COMM_TYPE_SHARED).Get_rank()
wp.set_device(f"cuda:{local_rank}" if wp.get_cuda_device_count() > 1 else "cuda:0")
device = wp.get_device()
nvshmem.init(mpi_comm=comm, initializer_method="mpi")
n_pes = nvshmem.n_pes()
assert n_pes == 2, "Need exactly 2 PEs"

NX = 16
N_STEPS = 6
SEED = 42
# Long enough for PE 0 to finish a whole adjoint step while PE 1 spins, if no barrier holds PE 0 back.
SPIN_ITERATIONS = 2_000_000

n_rows = 4
alloc_rows = n_rows + 2
NY = n_pes * n_rows
row_offset = rank * n_rows

# The arguments top_pe, top_halo_row, bottom_pe and bottom_halo_row of jacobi_step on a periodic domain: the first
# owned row goes into the bottom ghost row of the PE above, and the last owned row into the top ghost row of the PE
# below. They are not globals of those names, because such globals enter the kernel's hash, which would then differ
# between PEs.
neighbors = [(rank - 1) % n_pes, n_rows + 1, (rank + 1) % n_pes, 0]

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
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    except AssertionError as error:
        print(f"PE {rank}: {name}: FAILED\n{error}", flush=True)
        comm.Abort(1)
    print(f"PE {rank}: {name}: PASSED", flush=True)


@wp.func
def busy_work(iterations: int) -> float:
    """Keep the calling thread busy for ``iterations`` dependent steps, and return a value that is never negative."""
    x = float(0.0)
    for _i in range(iterations):
        x = x * 0.5 + 1.0
    return x


@wp.kernel
def jacobi_step(
    u: wp.array[wp.float32],  # field at step k, including ghost rows
    u_new: wp.array[wp.float32],  # field at step k + 1, symmetric
    nx: int,
    n_rows: int,
    top_pe: int,
    top_halo_row: int,
    bottom_pe: int,
    bottom_halo_row: int,
    spin: int,  # busy-work iterations, which change no value
    sink: wp.array[wp.float32],  # never written
):
    tid = wp.tid()
    iy = tid // (nx - 2) + 1
    ix = tid % (nx - 2) + 1

    # The adjoint kernel replays this before it pulls gradients, so the busy work delays the pulls too. The store,
    # which never happens, keeps the compiler from removing the busy work.
    if busy_work(spin) < 0.0:
        sink[0] = 0.0

    v = 0.25 * (u[(iy - 1) * nx + ix] + u[(iy + 1) * nx + ix] + u[iy * nx + ix - 1] + u[iy * nx + ix + 1])
    u_new[iy * nx + ix] = v

    if iy == 1:
        wp.nvshmem_p(u_new, top_halo_row * nx + ix, v, top_pe)
    if iy == n_rows:
        wp.nvshmem_p(u_new, bottom_halo_row * nx + ix, v, bottom_pe)


@wp.kernel
def jacobi_step_signaled(
    u: wp.array[wp.float32],  # field at step k, including ghost rows
    u_new: wp.array[wp.float32],  # field at step k + 1, symmetric
    flags: wp.array[wp.uint64],  # the last step whose rows arrived in the top and bottom ghost rows, symmetric
    nx: int,
    n_rows: int,
    step: int,  # k
):
    """Apply the stencil of ``jacobi_step`` once the ghost rows of ``u`` hold step ``step``, without pushing rows."""
    tid = wp.tid()
    iy = tid // (nx - 2) + 1
    ix = tid % (nx - 2) + 1

    # Each thread that reads a ghost row first waits on this PE's flag for it. The host wrote the ghost rows of the
    # initial field, so the flags of step 0, which start at zero, are already satisfied.
    if iy == 1:
        wp.nvshmem_signal_wait_until(flags, 0, wp.NVSHMEM_CMP_GE, step)
    if iy == n_rows:
        wp.nvshmem_signal_wait_until(flags, 1, wp.NVSHMEM_CMP_GE, step)

    v = 0.25 * (u[(iy - 1) * nx + ix] + u[(iy + 1) * nx + ix] + u[iy * nx + ix - 1] + u[iy * nx + ix + 1])
    u_new[iy * nx + ix] = v


@wp.kernel
def push_rows(
    u_new: wp.array[wp.float32],  # field at step k + 1, symmetric
    flags: wp.array[wp.uint64],
    nx: int,
    n_rows: int,
    top_pe: int,
    top_halo_row: int,
    bottom_pe: int,
    bottom_halo_row: int,
    step: int,  # k + 1
    spin: int,  # busy-work iterations, which change no value
    sink: wp.array[wp.float32],  # never written
):
    """Push the first and last owned rows of ``u_new`` into the ghost rows of the neighbors, one row per thread.

    Each push sets the receiving PE's flag for the ghost row to ``step`` after the row has arrived. One thread pushes
    a whole row, so the pushes run in a launch of their own, after the stencil has written every column.
    """
    # As in jacobi_step, the adjoint kernel replays this before it pulls gradients.
    if busy_work(spin) < 0.0:
        sink[0] = 0.0

    # The first owned row goes into the bottom ghost row of the PE above, whose flags[1] guards it, and the last owned
    # row into the top ghost row of the PE below, whose flags[0] guards it.
    if wp.tid() == 0:
        wp.nvshmem_put_signal(
            u_new[top_halo_row * nx + 1 : top_halo_row * nx + nx - 1],
            u_new[nx + 1 : 2 * nx - 1],
            flags,
            1,
            step,
            wp.NVSHMEM_SIGNAL_SET,
            top_pe,
        )
    else:
        wp.nvshmem_put_signal(
            u_new[bottom_halo_row * nx + 1 : bottom_halo_row * nx + nx - 1],
            u_new[n_rows * nx + 1 : n_rows * nx + nx - 1],
            flags,
            0,
            step,
            wp.NVSHMEM_SIGNAL_SET,
            bottom_pe,
        )
    # Complete the push in this launch, so that the push of the next step cannot overtake it and set the flag back.
    wp.nvshmem_quiet()


@wp.kernel
def interior_loss(u: wp.array[wp.float32], nx: int, n_rows: int, loss: wp.array[wp.float32]):
    tid = wp.tid()
    iy = tid // (nx - 2) + 1
    ix = tid % (nx - 2) + 1
    wp.atomic_add(loss, 0, u[iy * nx + ix] * u[iy * nx + ix])


@wp.kernel
def barrier_all():
    """One thread per PE, which is what the device-side collective requires."""
    wp.nvshmem_barrier_all()


@wp.kernel
def jacobi_step_reference(u: wp.array[wp.float32], u_new: wp.array[wp.float32], nx: int, ny: int):
    """Apply the stencil of ``jacobi_step`` to every row of a periodic field that has no ghost rows."""
    tid = wp.tid()
    iy = tid // (nx - 2)
    ix = tid % (nx - 2) + 1
    iy_up = (iy + ny - 1) % ny
    iy_down = (iy + 1) % ny

    v = 0.25 * (u[iy_up * nx + ix] + u[iy_down * nx + ix] + u[iy * nx + ix - 1] + u[iy * nx + ix + 1])
    u_new[iy * nx + ix] = v


@wp.kernel
def reference_loss(u: wp.array[wp.float32], nx: int, loss: wp.array[wp.float32]):
    """Sum the squares of every row of a field that has no ghost rows, excluding the edge columns."""
    tid = wp.tid()
    iy = tid // (nx - 2)
    ix = tid % (nx - 2) + 1
    wp.atomic_add(loss, 0, u[iy * nx + ix] * u[iy * nx + ix])


@wp.kernel(enable_backward=False)
def spin_kernel(iterations: int, sink: wp.array[wp.float32]):
    """Keep one thread busy; the store keeps the busy work from being optimized out."""
    sink[0] = busy_work(iterations)


# The full field, with zero edge columns that no step updates. Each PE takes its own rows and, as ghost rows, the
# neighboring rows of the periodic domain.
rng = np.random.default_rng(SEED)
initial = np.zeros((NY, NX), dtype=np.float32)
initial[:, 1:-1] = rng.standard_normal((NY, NX - 2))


def local_rows(pe):
    """Return the rows of the full field that ``fields[0]`` holds on ``pe``, from the top ghost row down."""
    return (pe * n_rows - 1 + np.arange(alloc_rows)) % NY


# Reference: the same steps and loss over the full domain on one GPU.
reference_fields = [wp.zeros(NY * NX, dtype=wp.float32, device=device, requires_grad=True) for _ in range(N_STEPS + 1)]
reference_fields[0].assign(initial.ravel())
reference_loss_value = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
with wp.Tape() as reference_tape:
    for k in range(N_STEPS):
        wp.launch(
            jacobi_step_reference,
            dim=NY * (NX - 2),
            inputs=[reference_fields[k], reference_fields[k + 1], NX, NY],
            device=device,
        )
    wp.launch(
        reference_loss,
        dim=NY * (NX - 2),
        inputs=[reference_fields[N_STEPS], NX, reference_loss_value],
        device=device,
    )
reference_tape.backward(reference_loss_value)
expected = reference_fields[0].grad.numpy().reshape(NY, NX)[row_offset : row_offset + n_rows]

fields = [symmetric(alloc_rows * NX, wp.float32, requires_grad=True) for _ in range(N_STEPS + 1)]
fields[0].assign(initial[local_rows(rank)].ravel())
flags = symmetric(2, wp.uint64)
loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
spin_sink = wp.zeros(1, dtype=wp.float32, device=device)


def delay():
    wp.launch(spin_kernel, dim=1, inputs=[SPIN_ITERATIONS, spin_sink], device=device)


def forward(pipelined=False):
    """Run the forward pass on a new tape and return the tape.

    The PEs synchronize the steps with a barrier kernel, or with ``pipelined``, with the signals of ``push_rows``. The
    arithmetic is the same either way.
    """
    if pipelined:
        # The flags count the steps of one pass from zero. The fields are zeroed too, so that a step that read its
        # ghost rows before they arrived would read zeros rather than the same values from an earlier pass. Every
        # push of an earlier pass has arrived on this PE by the time the stream gets here, and the host
        # synchronization keeps the other PE from pushing again before the flags and fields are zero.
        for field in fields[1:]:
            field.zero_()
        flags.zero_()
        sync()
    loss.zero_()
    with wp.Tape() as tape:
        for k in range(N_STEPS):
            spin = SPIN_ITERATIONS if rank == 1 and k == 0 else 0
            if pipelined:
                wp.launch(
                    jacobi_step_signaled,
                    dim=n_rows * (NX - 2),
                    inputs=[fields[k], fields[k + 1], flags, NX, n_rows, k],
                    device=device,
                )
                wp.launch(
                    push_rows,
                    dim=2,
                    inputs=[fields[k + 1], flags, NX, n_rows, *neighbors, k + 1, spin, spin_sink],
                    device=device,
                )
            else:
                wp.launch(
                    jacobi_step,
                    dim=n_rows * (NX - 2),
                    inputs=[fields[k], fields[k + 1], NX, n_rows, *neighbors, spin, spin_sink],
                    device=device,
                )
                # The forward synchronization between steps is the program's own, so the tape does not record it.
                wp.launch(barrier_all, dim=1, device=device, record_tape=False)
            if rank == 1:
                # Runs before the adjoint of this step.
                tape.record_func(backward=delay, arrays=[])
        wp.launch(interior_loss, dim=n_rows * (NX - 2), inputs=[fields[N_STEPS], NX, n_rows, loss], device=device)
    return tape


def forward_backward(pipelined=False):
    """Run the forward pass on a tape and the backward pass, and return a device copy of ``fields[0].grad``."""
    tape = forward(pipelined)
    tape.backward(loss)
    # Copied on the stream, so the host goes on to the next pass without waiting for the other PE.
    return wp.clone(fields[0].grad)


def zero_gradients():
    for field in fields:
        field.grad.zero_()


def owned_gradient(local_gradient):
    """Return the gradient of the global loss with respect to this PE's rows of the initial field.

    The ghost rows of ``fields[0]`` are copies that the host wrote, not values pushed on the tape, so the gradient
    through each copy stays with the PE that read it. Add every PE's ghost-row gradients into the rows they copy.
    """
    full = np.zeros((NY, NX), dtype=np.float64)
    for pe, gradient in enumerate(comm.allgather(local_gradient.reshape(alloc_rows, NX))):
        np.add.at(full, local_rows(pe), gradient)
    return full[row_offset : row_offset + n_rows]


# Every PE has initialized its fields before any PE pushes into them.
sync()
first_gradient = forward_backward()

# The first pass pins the barriers before adjoint launches: without them, PE 0 pulls gradients that PE 1 has not
# accumulated yet. The second pass pins the trailing barrier: with no host synchronization across PEs in between, PE 0
# zeroes its gradients for it while the last adjoint launch of the first pass on PE 1 is still pulling them, so
# without the trailing barrier PE 1's first gradient mismatches. The single-PE script checks how many trailing barriers
# Tape.backward() enqueues.
zero_gradients()
second_gradient = forward_backward()

first = owned_gradient(first_gradient.numpy())
second = owned_gradient(second_gradient.numpy())
check("gradient matches reference", first, expected)
check("second iteration matches reference", second, expected)

# The third pass synchronizes the forward steps with signals instead of barriers. The backward pass needs nothing in
# their place: the signals and waits have no adjoint, and the adjoint barriers alone order the pulls.
zero_gradients()
pipelined_gradient = forward_backward(pipelined=True)
check("pipelined signals match reference", owned_gradient(pipelined_gradient.numpy()), expected)

# The eager backward passes above loaded the barrier kernel, so the fourth pass can capture its backward pass. As in
# the second pass, no host synchronization across PEs separates the replays: only the trailing barrier in the graph
# keeps PE 0 from zeroing its gradients for the second replay while PE 1's last adjoint launch still pulls them.
tape = forward(pipelined=True)
with wp.ScopedCapture(device=device) as capture:
    tape.backward(loss)
replay_gradients = []
for _replay in range(2):
    zero_gradients()
    wp.capture_launch(capture.graph)
    replay_gradients.append(wp.clone(fields[0].grad))
check(
    "captured backward matches reference",
    [owned_gradient(gradient.numpy()) for gradient in replay_gradients],
    [expected, expected],
)

sync()
for buf in symmetric_buffers:
    nvshmem.free(buf)
nvshmem.finalize()
MPI.Finalize()
