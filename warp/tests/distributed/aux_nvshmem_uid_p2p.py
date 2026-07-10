# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-PE UID-bootstrap test using only NVSHMEM's local P2P transport.

This program is launched by ``TestNvshmemMultiPE.test_uid_local_p2p`` through
the ``nvshmrun.hydra`` executable packaged with NVSHMEM. Rank 0 shares the UID
through a temporary file, avoiding an MPI runtime and communicator.
"""

import gc
import os
import pathlib
import time

import numpy as np
import nvshmem.core as nvshmem
from nvshmem.bindings import nvshmem as nvshmem_bindings

rank = int(os.environ["PMI_RANK"])
nranks = int(os.environ["PMI_SIZE"])
local_rank = int(os.environ["MPI_LOCALRANKID"])
assert nranks == 2
assert os.environ["NVSHMEM_REMOTE_TRANSPORT"] == "none"

cache_path = os.path.join(os.environ["NVSHMEM_TEST_CACHE_ROOT"], f"rank-{rank}")
os.environ["WARP_CACHE_PATH"] = cache_path
os.environ["WARP_CACHE_ROOT"] = cache_path

import warp as wp  # noqa: E402


@wp.kernel
def query_and_put(
    symmetric_buffer: wp.array[wp.float32],
    out_pe: wp.array[wp.int32],
    out_npes: wp.array[wp.int32],
):
    pe = wp.nvshmem_my_pe()
    out_pe[0] = pe
    out_npes[0] = wp.nvshmem_n_pes()
    if pe == 0:
        wp.nvshmem_float_p(symmetric_buffer, 0, 42.0, 1)
    wp.nvshmem_quiet()
    wp.nvshmem_barrier_all()


wp.config.cuda_output = "cubin"
wp.config.log_level = wp.LOG_WARNING
assert wp.get_cuda_device_count() >= nranks
device = wp.get_device(f"cuda:{local_rank}")
wp.set_device(device)

uid_path = pathlib.Path(os.environ["NVSHMEM_TEST_UID_FILE"])
if rank == 0:
    uid = nvshmem.get_unique_id()
    temporary_path = uid_path.with_suffix(f".{os.getpid()}.tmp")
    temporary_path.write_bytes(uid._data.view(np.uint8).tobytes())
    temporary_path.replace(uid_path)
else:
    deadline = time.monotonic() + 30.0
    while not uid_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for NVSHMEM UID at {uid_path}")
        time.sleep(0.05)
    uid_bytes = uid_path.read_bytes()
    uid = nvshmem.get_unique_id(empty=True)
    uid._data.view(np.uint8)[:] = np.frombuffer(uid_bytes, dtype=np.uint8)

nvshmem.init(uid=uid, rank=rank, nranks=nranks, initializer_method="uid")
version = nvshmem.get_version()
assert version.libnvshmem_version == os.environ["NVSHMEM_TEST_VERSION"], version

symmetric_buffer = wp.zeros(1, dtype=wp.float32, device=device, symmetric=True)
out_pe = wp.zeros(1, dtype=wp.int32, device=device)
out_npes = wp.zeros(1, dtype=wp.int32, device=device)

wp.launch(query_and_put, dim=1, inputs=[symmetric_buffer, out_pe, out_npes], device=device)
wp.synchronize_device(device)
nvshmem_bindings.barrier_all()

pe_value = int(out_pe.numpy()[0])
npes_value = int(out_npes.numpy()[0])
buffer_value = float(symmetric_buffer.numpy()[0])
assert pe_value == rank, (rank, pe_value)
assert npes_value == nranks, (rank, npes_value)
if rank == 1:
    assert buffer_value == 42.0, buffer_value

print(
    f"PE {rank}/{nranks} on {device.alias}: query=({pe_value}, {npes_value}), "
    f"symmetric_buffer={buffer_value}, remote_transport=none: PASSED",
    flush=True,
)

nvshmem_bindings.barrier_all()
del symmetric_buffer, out_pe, out_npes
gc.collect()
nvshmem.finalize()

if rank == 0:
    uid_path.unlink(missing_ok=True)
