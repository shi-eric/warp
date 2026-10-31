# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-PE test of the barriers and diagnostics of NVSHMEM adjoint launches.

This program is launched by ``TestNVSHMEMAutodiff.test_single_pe_behaviors`` through ``mpirun -np 1``. With one
PE every barrier completes at once, so the checks count the barriers each launch path enqueues on the device
instead of observing the ordering they enforce. The push and pull kernels below take the target PE as an
argument so that they can address PE 0, the only PE of the job.
"""

import contextlib
import io
import warnings

import numpy as np
import nvshmem.core as nvshmem
from mpi4py import MPI

import warp as wp
import warp._src.context as warp_context
from warp.tests.distributed import aux_nvshmem_record_kernels as record_kernels
from warp.types import type_size_in_bytes

wp.config.log_level = wp.LOG_WARNING
wp.set_device("cuda:0")
device = wp.get_device()
nvshmem.init(mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")

symmetric_buffers = []


def symmetric(shape, dtype, requires_grad=False):
    """Allocate a zeroed symmetric array, with a zeroed symmetric gradient if ``requires_grad``."""

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


def report(name, passed, detail):
    """Print the outcome of a check; a failure is reported and the remaining checks still run."""
    if passed:
        print(f"PE 0: {name}: PASSED", flush=True)
    else:
        print(f"PE 0: {name}: FAILED ({detail})", flush=True)


def count_barriers(action):
    """Return the number of adjoint barriers that ``action()`` enqueues on the device."""
    before = device.nvshmem_adjoint_barriers
    action()
    return device.nvshmem_adjoint_barriers - before


def warp_warnings(action):
    """Return the Warp warnings that ``action()`` emits, one line each.

    Warp's default logger writes its warnings to standard error itself, so ``warnings.catch_warnings(record=True)``
    does not collect them.
    """
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()) as stderr:
        warnings.simplefilter("always")
        action()
    return [line for line in stderr.getvalue().splitlines() if line.startswith("Warp ")]


def launch_twice(kernel, **kwargs):
    """Launch ``kernel`` twice with the same arguments."""
    for _launch in range(2):
        wp.launch(kernel, **kwargs)


@wp.kernel
def push_kernel(dest: wp.array[wp.float32], values: wp.array[wp.float32], pe: int):
    wp.nvshmem_p(dest, 0, values[0], pe)


@wp.kernel
def pull_kernel(source: wp.array[wp.float32], out: wp.array[wp.float32], pe: int):
    out[0] = wp.nvshmem_g(source, 0, pe)


@wp.kernel
def loss_kernel(source: wp.array[wp.float32], loss: wp.array[wp.float32]):
    loss[0] = 2.0 * source[0]


# In a module of its own, so that switching that module to deterministic mode leaves the other kernels alone.
# The atomic makes Warp route the kernel's launches through the deterministic launcher.
@wp.kernel(module="unique")
def push_sum_kernel(dest: wp.array[wp.float32], values: wp.array[wp.float32], total: wp.array[wp.float32], pe: int):
    i = wp.tid()
    wp.atomic_add(total, 0, values[i])
    if i == 0:
        wp.nvshmem_p(dest, 0, values[0], pe)


dest = symmetric(1, wp.float32, requires_grad=True)
source = symmetric(1, wp.float32, requires_grad=True)
values = wp.array([1.5], dtype=wp.float32, device=device, requires_grad=True)
out = wp.zeros(1, dtype=wp.float32, device=device)
push_inputs = [dest, values, 0]
push_adj_inputs = [dest.grad, values.grad, 0]


def push_adjoint(dim=1, **kwargs):
    return wp.launch(push_kernel, dim=dim, inputs=push_inputs, adj_inputs=push_adj_inputs, adjoint=True, **kwargs)


# The records reach the launch path through the module metadata, also when the module comes from the kernel cache.
wp.launch(push_kernel, dim=1, inputs=push_inputs)
wp.launch(pull_kernel, dim=1, inputs=[source, out, 0])
expected_records = {
    push_kernel: (True, (), False),
    pull_kernel: (True, ("nvshmem_g",), False),
}


def load_records():
    return {kernel: kernel.module.load(device).get_kernel_hooks(kernel).nvshmem_adjoint for kernel in expected_records}


first_records = load_records()
for kernel in expected_records:
    kernel.module.unload()
cached_records = load_records()
report(
    "records survive a cache hit",
    first_records == expected_records and cached_records == expected_records,
    f"first load {first_records}, cache hit {cached_records}",
)

# The barrier kernel has not run yet, so capturing the first adjoint launch must fail before anything is enqueued.
wp.launch(push_kernel, dim=1, inputs=push_inputs)
capture_error = None
try:
    with wp.ScopedCapture():
        push_adjoint()
except RuntimeError as error:
    capture_error = error
report(
    "capture before warm-up raises",
    capture_error is not None and "run one backward pass before capturing" in str(capture_error),
    f"raised {capture_error!r}",
)

added = count_barriers(push_adjoint)
report("flagged adjoint launch adds one barrier", added == 1, f"added {added}")

# These record kernels address PE 1, but their adjoints access no remote memory, so they run on one PE.
added_local = count_barriers(
    lambda: wp.launch(record_kernels.local_kernel, dim=1, inputs=[values], adj_inputs=[values.grad], adjoint=True)
)
# Capturing the pull-family warning that the adjoint launch of get_only_kernel draws keeps standard error clean.
added_get = count_barriers(
    lambda: warp_warnings(
        lambda: wp.launch(
            record_kernels.get_only_kernel,
            dim=1,
            inputs=[dest, source],
            adj_inputs=[dest.grad, source.grad],
            adjoint=True,
        )
    )
)
report(
    "unflagged adjoint launch adds none",
    added_local == 0 and added_get == 0,
    f"local_kernel added {added_local}, get_only_kernel added {added_get}",
)

added = count_barriers(lambda: wp.launch(push_kernel, dim=1, inputs=push_inputs))
report("forward launch adds none", added == 0, f"added {added}")

added = count_barriers(lambda: push_adjoint(dim=0))
report("zero-size adjoint launch adds one", added == 1, f"added {added}")

before = device.nvshmem_adjoint_barriers
command = push_adjoint(record_cmd=True)
added_at_record = device.nvshmem_adjoint_barriers - before
added_at_launch = count_barriers(lambda: (command.launch(), command.launch()))
report(
    "record_cmd",
    added_at_record == 0 and added_at_launch == 2,
    f"recording added {added_at_record}, two launches added {added_at_launch}",
)

wp.config.nvshmem_adjoint_barrier = False
try:
    added = count_barriers(push_adjoint)
finally:
    wp.config.nvshmem_adjoint_barrier = True
report("config off adds none", added == 0, f"added {added}")

sum_module = push_sum_kernel.module
previous_mode = wp.get_module_options(sum_module)["deterministic"]
wp.set_module_options({"deterministic": wp.DeterministicMode.RUN_TO_RUN}, module=sum_module)
try:
    sum_values = wp.ones(4, dtype=wp.float32, device=device, requires_grad=True)
    total = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    sum_inputs = [dest, sum_values, total, 0]
    sum_adj_inputs = [dest.grad, sum_values.grad, total.grad, 0]

    def sum_adjoint(**kwargs):
        return wp.launch(push_sum_kernel, dim=4, inputs=sum_inputs, adj_inputs=sum_adj_inputs, adjoint=True, **kwargs)

    before = device.nvshmem_adjoint_barriers
    command = sum_adjoint(record_cmd=True)
    added_at_record = device.nvshmem_adjoint_barriers - before
    added_at_launch = count_barriers(command.launch)
    added_immediate = count_barriers(sum_adjoint)
finally:
    wp.set_module_options({"deterministic": previous_mode}, module=sum_module)
report(
    "deterministic path adds one barrier",
    type(command).__name__ == "DeterministicLaunch"
    and added_at_record == 0
    and added_at_launch == 1
    and added_immediate == 1,
    f"recorded a {type(command).__name__}, recording added {added_at_record}, its launch added "
    f"{added_at_launch}, an immediate launch added {added_immediate}",
)

# values -> first -> second -> loss: two flagged launches and one local launch on the tape.
first = symmetric(1, wp.float32, requires_grad=True)
second = symmetric(1, wp.float32, requires_grad=True)
loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
tape_grads = [values.grad, first.grad, second.grad]
for grad in tape_grads:
    grad.zero_()
with wp.Tape() as tape:
    wp.launch(push_kernel, dim=1, inputs=[first, values, 0])
    wp.launch(push_kernel, dim=1, inputs=[second, first, 0])
    wp.launch(loss_kernel, dim=1, inputs=[second, loss])
added = count_barriers(lambda: tape.backward(loss))
report("tape adds trailing barrier", added == 3, f"added {added}")
eager_grads = [grad.numpy() for grad in tape_grads]

with wp.ScopedCapture() as capture:
    tape.backward(loss)
replay_grads = []
for _replay in range(2):
    for grad in tape_grads:
        grad.zero_()
    wp.capture_launch(capture.graph)
    replay_grads.append([grad.numpy() for grad in tape_grads])
try:
    # d(loss)/d(second) = 2, and each push passes the gradient of its destination back to its value.
    np.testing.assert_allclose(eager_grads, np.full((3, 1), 2.0))
    for grads in replay_grads:
        np.testing.assert_allclose(grads, eager_grads)
except AssertionError as error:
    report("graph capture after warm-up", False, error)
else:
    report("graph capture after warm-up", True, "")

pull_out = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
pull_inputs = [source, pull_out, 0]
forward_warnings = warp_warnings(lambda: launch_twice(pull_kernel, dim=1, inputs=pull_inputs))
adjoint_warnings = warp_warnings(
    lambda: launch_twice(
        pull_kernel, dim=1, inputs=pull_inputs, adj_inputs=[source.grad, pull_out.grad, 0], adjoint=True
    )
)
report(
    "pull warning once",
    not forward_warnings
    and len(adjoint_warnings) == 1
    and "wp.nvshmem_g() or wp.nvshmem_get()" in adjoint_warnings[0]
    and '"pull_kernel"' in adjoint_warnings[0],
    f"forward launches warned {forward_warnings}, adjoint launches warned {adjoint_warnings}",
)

# The record kernels address PE 1, which this job does not have. Their adjoint launches have no threads, so
# nothing addresses it, but the launch still reaches the diagnostics, which run ahead of the zero-size return.
sig = wp.zeros(1, dtype=wp.uint64, device=device)
fused_warnings = warp_warnings(
    lambda: launch_twice(
        record_kernels.fused_kernel,
        dim=0,
        inputs=[dest, source, sig],
        adj_inputs=[dest.grad, source.grad, None],
        adjoint=True,
    )
)
pipelined_warnings = warp_warnings(
    lambda: launch_twice(
        record_kernels.pipelined_kernel,
        dim=0,
        inputs=[dest, values, sig],
        adj_inputs=[dest.grad, values.grad, None],
        adjoint=True,
    )
)
report(
    "fused warning once",
    len(fused_warnings) == 1
    and "before waiting on a signal" in fused_warnings[0]
    and '"fused_kernel"' in fused_warnings[0]
    and not pipelined_warnings,
    f"fused_kernel warned {fused_warnings}, pipelined_kernel warned {pipelined_warnings}",
)

is_symmetric = warp_context.runtime.core.wp_nvshmem_is_symmetric
local_probe = is_symmetric(values.ptr)
symmetric_probe = is_symmetric(dest.ptr)
print(f"PE 0: wp_nvshmem_is_symmetric: local array {local_probe}, symmetric array {symmetric_probe}", flush=True)
report(
    "symmetric check probe",
    local_probe == 0 and symmetric_probe == 1,
    f"local array {local_probe}, symmetric array {symmetric_probe}",
)


def push_adjoint_error(inputs, adj_inputs):
    """Return the ``RuntimeError`` that an adjoint launch of ``push_kernel`` raises, or ``None``."""
    try:
        wp.launch(push_kernel, dim=1, inputs=inputs, adj_inputs=adj_inputs, adjoint=True)
    except RuntimeError as error:
        return error
    return None


def is_dest_gradient_error(error):
    """Return whether ``error`` reports the non-symmetric gradient of ``push_kernel``'s ``dest`` argument."""
    message = str(error)
    return (
        error is not None
        and "gradient array is not" in message
        and 'argument "dest"' in message
        and 'kernel "push_kernel"' in message
    )


# Symmetric data whose gradient Warp allocates, which is not symmetric. The launch passes no adjoint for the
# destination, so the gradient it would use is the array's own.
bad_buf = nvshmem.buffer(type_size_in_bytes(wp.float32))
symmetric_buffers.append(bad_buf)
bad = wp.array(ptr=int(bad_buf.handle), shape=1, dtype=wp.float32, device=device, requires_grad=True)
error = push_adjoint_error([bad, values, 0], [None, values.grad, 0])
report("non-symmetric grad raises", is_dest_gradient_error(error), f"raised {error!r}")

error = push_adjoint_error([dest, values, 0], [wp.zeros_like(dest), values.grad, 0])
report("non-symmetric adj_inputs raises", is_dest_gradient_error(error), f"raised {error!r}")

local_values = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
error = push_adjoint_error([dest, local_values, 0], [dest.grad, local_values.grad, 0])
report("local arrays pass", error is None, f"raised {error!r}")

# The module of get_only_kernel uses NVSHMEM, but its adjoint accesses no remote memory, so its gradients need not be
# symmetric and bad's Warp-allocated gradient is accepted. The launch has no threads, so the kernel's hard-coded PE 1
# is never addressed, but it still reaches the adjoint launch checks.
error = None
try:
    warp_warnings(
        lambda: wp.launch(
            record_kernels.get_only_kernel, dim=0, inputs=[dest, bad], adj_inputs=[dest.grad, None], adjoint=True
        )
    )
except RuntimeError as launch_error:
    error = launch_error
report("validation skipped without remote access", error is None, f"raised {error!r}")

for buf in symmetric_buffers:
    nvshmem.free(buf)
nvshmem.finalize()
MPI.Finalize()
