# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-PE test: verify nvshmem_float_p writes to remote PE.

Launch:
    mpirun -np 2 uv run --with mpi4py --with nvshmem4py-cu13 python -m warp.tests.distributed.test_nvshmem_float_p
"""

import os
import tempfile

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

assert nvshmem.n_pes() >= 2, "Need at least 2 PEs"


@wp.kernel
def put_kernel(buf: wp.array[float], my_pe: int):
    if my_pe == 0:
        wp.nvshmem_float_p(buf, 0, 42.0, 1)
    wp.nvshmem_quiet()
    wp.nvshmem_barrier_all()


buf = wp.zeros(4, dtype=wp.float32, device=device, symmetric=True)
wp.launch(put_kernel, dim=1, inputs=[buf, nvshmem.my_pe()], device=device)
wp.synchronize_device(device)
_nvshmem_bindings.barrier_all()

result = buf.numpy()
if rank == 1:
    assert abs(result[0] - 42.0) < 1e-6, f"Expected 42.0, got {result[0]}"
    print("PE 1: PASSED", flush=True)
else:
    print(f"PE {rank}: OK", flush=True)

nvshmem.finalize()
MPI.Finalize()
