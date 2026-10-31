# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-PE program that ``TestNVSHMEMMultiPE.test_cached_module_in_new_process`` runs twice.

Both runs share a kernel cache through ``WARP_CACHE_PATH``, so the second run loads the module from the
cache instead of compiling it. Its kernel must still see initialized NVSHMEM state.
"""

import nvshmem.core as nvshmem
from mpi4py import MPI

import warp as wp

wp.set_device("cuda:0")
nvshmem.init(mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")


@wp.kernel
def query(out: wp.array[wp.int32]):
    out[0] = wp.nvshmem_my_pe()
    out[1] = wp.nvshmem_n_pes()


out = wp.full(2, -1, dtype=wp.int32)
wp.launch(query, dim=1, inputs=[out])
print(f"PE 0 query: {out.numpy().tolist()}", flush=True)

nvshmem.finalize()
MPI.Finalize()
