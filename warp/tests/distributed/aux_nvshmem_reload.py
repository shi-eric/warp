# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-PE test that unloads an NVSHMEM module and then loads another one.

This program is launched by ``TestNVSHMEMMultiPE.test_reload_after_unload`` through ``mpirun -np 1``. NVSHMEM
tracks registered modules by the address of their device state, and the CUDA driver can place the second
module's state at the address freed by the first. Warp must unregister the first module when it unloads it,
or NVSHMEM skips initializing the second module.
"""

import gc

import nvshmem.core as nvshmem
from mpi4py import MPI

import warp as wp

wp.config.quiet = True
wp.set_device("cuda:0")
nvshmem.init(mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")


@wp.kernel(module="unique")
def first_kernel(out: wp.array[wp.int32]):
    out[0] = wp.nvshmem_n_pes()


@wp.kernel(module="unique")
def second_kernel(out: wp.array[wp.int32]):
    out[0] = wp.nvshmem_n_pes() + 100


out = wp.zeros(1, dtype=wp.int32)
wp.launch(first_kernel, dim=1, inputs=[out])
assert out.numpy()[0] == 1, out.numpy()

first_kernel.module.unload()
gc.collect()
wp.synchronize()

wp.launch(second_kernel, dim=1, inputs=[out])
assert out.numpy()[0] == 101, f"second module saw {out.numpy()[0] - 100} PEs instead of 1"
print("PE 0: reload after unload: PASSED", flush=True)

nvshmem.finalize()
MPI.Finalize()
