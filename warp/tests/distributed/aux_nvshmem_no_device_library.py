# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-PE program that ``TestNVSHMEMMultiPE.test_missing_device_library_fails_at_compile`` runs.

With no device library override and an NVSHMEM installation that does not ship the device library, as with wheels
older than NVSHMEM 3.9, compiling a module that uses NVSHMEM built-ins must fail with an error that names the version
requirement. An installation that ships the device library compiles the module, so the program reports that the
check does not apply.
"""

import nvshmem.core as nvshmem
from mpi4py import MPI

import warp as wp

wp.config.quiet = True
wp.set_device("cuda:0")
nvshmem.init(mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")


@wp.kernel
def query(out: wp.array[wp.int32]):
    out[0] = wp.nvshmem_my_pe()


try:
    wp.load_module(query.module, device="cuda:0")
except RuntimeError as e:
    verdict = "PASSED" if "NVSHMEM 3.9 or newer" in str(e) else "FAILED"
    print(f"{verdict}: {e}", flush=True)
else:
    print("SKIPPED: the NVSHMEM installation ships the device library", flush=True)

nvshmem.finalize()
MPI.Finalize()
