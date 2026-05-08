# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-PE test: verify nvshmem_my_pe and nvshmem_n_pes return correct values.

Launch:
    mpirun -np 2 uv run --with mpi4py --with nvshmem4py-cu13 python -m warp.tests.distributed.test_nvshmem_pe_query
"""

import os
import tempfile

import nvshmem.core as nvshmem
from mpi4py import MPI

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


@wp.kernel
def query_kernel(out_pe: wp.array[wp.int32], out_npes: wp.array[wp.int32]):
    out_pe[0] = wp.nvshmem_my_pe()
    out_npes[0] = wp.nvshmem_n_pes()


out_pe = wp.zeros(1, dtype=wp.int32, device=device)
out_npes = wp.zeros(1, dtype=wp.int32, device=device)
wp.launch(query_kernel, dim=1, inputs=[out_pe, out_npes], device=device)
wp.synchronize_device(device)

pe_val = out_pe.numpy()[0]
npes_val = out_npes.numpy()[0]
my_pe = nvshmem.my_pe()
n_pes = nvshmem.n_pes()

assert pe_val == my_pe, f"Expected pe={my_pe}, got {pe_val}"
assert npes_val == n_pes, f"Expected n_pes={n_pes}, got {npes_val}"
print(f"PE {my_pe}/{n_pes}: PASSED", flush=True)

nvshmem.finalize()
MPI.Finalize()
