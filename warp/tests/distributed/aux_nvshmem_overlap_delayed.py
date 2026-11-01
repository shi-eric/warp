# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run ``example_jacobi_nvshmem_overlap.py`` on two PEs with every push of PE 1 delayed.

This program is launched by ``TestNVSHMEMHaloExchange`` through ``mpirun -np 2``. On PE 1, it wraps ``wp.launch`` so
that every launch of the example's ``push_rows`` kernel starts with ``spin_kernel`` on the stream of the push, and it
then runs the unchanged example with 40 iterations on a 64 by 64 mesh. PE 0 reaches its waits before the rows it waits
for have arrived, so the example prints the norm of ``example_jacobi_nvshmem_device.py`` only if the waits hold.

The example runs as ``__main__``, so ``spin_kernel`` is in the module of the example's kernels. It loads with them
during the eager first block, and the captured graph contains it, so every replay delays the pushes too. At the end,
PE 1 prints how many pushes were delayed.
"""

import runpy
import sys

from mpi4py import MPI

import warp as wp

# Long enough for PE 0 to reach its waits well before a delayed push from PE 1 arrives.
SPIN_ITERATIONS = 2_000_000

rank = MPI.COMM_WORLD.Get_rank()


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


launch = wp.launch
spin_sink = None
spin_runs = None


def launch_after_spin(kernel, *args, **kwargs):
    """Launch ``kernel``, after ``spin_kernel`` on the same stream if ``kernel`` is the example's ``push_rows``."""
    global spin_sink, spin_runs
    if kernel.key.endswith("push_rows"):
        stream = kwargs.get("stream")
        device = stream.device if stream is not None else wp.get_device(kwargs.get("device"))
        if spin_runs is None:
            # The first push runs in the example's eager block, so these allocations are not captured.
            spin_sink = wp.zeros(1, dtype=wp.float32, device=device)
            spin_runs = wp.zeros(1, dtype=wp.int32, device=device)
        launch(spin_kernel, dim=1, inputs=[SPIN_ITERATIONS, spin_sink, spin_runs], device=device, stream=stream)
    return launch(kernel, *args, **kwargs)


if rank == 1:
    wp.launch = launch_after_spin

sys.argv = ["example", "--iterations", "40", "--size", "64"]
runpy.run_module("warp.examples.distributed.example_jacobi_nvshmem_overlap", run_name="__main__")

if rank == 1:
    # Every push ran after a spin, in the eager block and in every replay, so the delay cannot have been bypassed.
    delayed = 0 if spin_runs is None else int(spin_runs.numpy()[0])
    print(f"PE 1: {delayed} pushes delayed", flush=True)
