# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVSHMEM put bandwidth benchmark.

Measures unidirectional put bandwidth between two GPUs across a range of
message sizes. PE 0 puts data to PE 1 using parallel scalar puts (one
thread per element). A separate single-thread kernel handles the
quiet + barrier between iterations.

Launch:
    mpirun -np 2 uv run --with mpi4py --with nvshmem4py-cu13 \
        python -m warp.examples.distributed.example_nvshmem_put_bw
"""

import os
import tempfile
import time

import nvshmem.core as nvshmem
from mpi4py import MPI
from nvshmem.bindings import nvshmem as _nvshmem_bindings

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
os.environ["WARP_CACHE_ROOT"] = os.path.join(tempfile.gettempdir(), f"warp_nvshmem_rank{rank}")

import warp as wp  # noqa: E402

wp.config.cuda_output = "cubin"
wp.config.quiet = True

local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
local_rank = local_comm.Get_rank()
if wp.get_cuda_device_count() > 1:
    wp.set_device(f"cuda:{local_rank}")
else:
    wp.set_device("cuda:0")
device = wp.get_device()
nvshmem.init(mpi_comm=comm, initializer_method="mpi")

my_pe = nvshmem.my_pe()
n_pes = nvshmem.n_pes()
assert n_pes == 2, f"This benchmark requires exactly 2 PEs, got {n_pes}"


@wp.kernel
def put_kernel(src: wp.array[float], dest: wp.array[float], nelems: int, pe: int):
    """Each thread puts one element to the remote PE."""
    tid = wp.tid()
    if tid < nelems:
        wp.nvshmem_float_p(dest, tid, src[tid], pe)


@wp.kernel
def sync_kernel():
    """Single-thread kernel: quiet (complete all puts) + barrier (sync PEs)."""
    wp.nvshmem_quiet()
    wp.nvshmem_barrier_all()


# Benchmark parameters
WARMUP = 5
ITERATIONS = 20
MAX_SIZE = 64 * 1024 * 1024  # 64 MB
max_nelems = MAX_SIZE // 4  # float32 = 4 bytes

# Allocate max-size buffers once
src = wp.zeros(max_nelems, dtype=wp.float32, device=device)
dest = wp.zeros(max_nelems, dtype=wp.float32, device=device, symmetric=True)

if my_pe == 0:
    print("NVSHMEM Put Bandwidth (PE 0 -> PE 1)")
    print(f"{'Size (B)':>12}  {'Bandwidth (GB/s)':>16}")
    print(f"{'-' * 12}  {'-' * 16}")

_nvshmem_bindings.barrier_all()

# Sweep message sizes: 4B to 64MB (powers of 2, in floats)
nelems = 1
while nelems <= max_nelems:
    size_bytes = nelems * 4
    peer = 1 - my_pe

    # Warm up
    for _ in range(WARMUP):
        wp.launch(put_kernel, dim=max(nelems, 1), inputs=[src, dest, nelems, peer], device=device)
        wp.launch(sync_kernel, dim=1, inputs=[], device=device)
    wp.synchronize_device(device)

    # Timed run
    _nvshmem_bindings.barrier_all()
    t_start = time.perf_counter()

    for _ in range(ITERATIONS):
        wp.launch(put_kernel, dim=max(nelems, 1), inputs=[src, dest, nelems, peer], device=device)
        wp.launch(sync_kernel, dim=1, inputs=[], device=device)
    wp.synchronize_device(device)

    _nvshmem_bindings.barrier_all()
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    bw_gb = (size_bytes * ITERATIONS) / elapsed / 1e9

    if my_pe == 0:
        print(f"{size_bytes:>12}  {bw_gb:>16.2f}")

    nelems *= 2

_nvshmem_bindings.barrier_all()
nvshmem.finalize()
MPI.Finalize()
