# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-PE program that ``TestNVSHMEMMultiPE.test_host_too_old_without_override`` runs.

With a host library older than NVSHMEM 3.9 and no device library override, compiling a module that uses NVSHMEM
built-ins must fail with an error that names the version requirement. With a newer host library the device library is
found next to it, so the program reports that the check does not apply.
"""

import nvshmem.core as nvshmem
from mpi4py import MPI

import warp as wp
import warp._src.context as warp_context
import warp._src.nvshmem as nvshmem_support

wp.config.quiet = True
wp.set_device("cuda:0")
nvshmem.init(mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")


@wp.kernel
def query(out: wp.array[wp.int32]):
    out[0] = wp.nvshmem_my_pe()


host_version = nvshmem_support.decode_version(warp_context.runtime.core.wp_nvshmem_get_loaded_version())
if host_version >= nvshmem_support.MIN_HOST_VERSION:
    print(f"SKIPPED: host library {'.'.join(map(str, host_version))} ships the device library", flush=True)
else:
    try:
        wp.load_module(query.module, device="cuda:0")
    except RuntimeError as e:
        verdict = "PASSED" if "NVSHMEM 3.9 or newer" in str(e) else "FAILED"
        print(f"{verdict}: {e}", flush=True)
    else:
        print("FAILED: module loaded without a device library", flush=True)

nvshmem.finalize()
MPI.Finalize()
