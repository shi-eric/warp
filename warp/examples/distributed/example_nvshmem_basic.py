"""Basic NVSHMEM smoke test: PE queries, symmetric alloc, float_p.

Launch:
    mpirun -np 2 python -m warp.examples.distributed.example_nvshmem_basic
"""

import os
import tempfile

import nvshmem.core as nvshmem
from mpi4py import MPI

# Must set WARP_CACHE_ROOT before importing Warp (read at import time)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()
os.environ["WARP_CACHE_ROOT"] = os.path.join(tempfile.gettempdir(), f"warp_nvshmem_rank{rank}")
os.environ.setdefault("NVSHMEM_SYMMETRIC_SIZE", "1073741824")  # 1 GB

import warp as wp  # noqa: E402

wp.config.cuda_output = "cubin"

# Select GPU based on local rank (not global rank) to handle multi-node correctly
local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
local_rank = local_comm.Get_rank()
num_cuda_devices = wp.get_cuda_device_count()
if num_cuda_devices > 1:
    wp.set_device(f"cuda:{local_rank}")
else:
    wp.set_device("cuda:0")
device = wp.get_device()

# CUDA context must exist before nvshmem.init()
nvshmem.init(mpi_comm=comm, initializer_method="mpi")
my_pe = nvshmem.my_pe()
n_pes = nvshmem.n_pes()
print(f"[PE {my_pe}] Host: NVSHMEM initialized ({n_pes} PEs)", flush=True)


# === All builtins in one kernel to keep it in one Warp module ===


@wp.kernel
def nvshmem_test_kernel(
    out_pe: wp.array(dtype=wp.int32),
    out_npes: wp.array(dtype=wp.int32),
    buf: wp.array(dtype=float),
):
    # 1. PE queries
    pe = wp.nvshmem_my_pe()
    npes = wp.nvshmem_n_pes()
    out_pe[0] = pe
    out_npes[0] = npes

    # 2. PE 0 writes 42.0 into buf[0] on PE 1
    if pe == 0 and npes >= 2:
        wp.nvshmem_float_p(buf, 0, 42.0, 1)

    wp.nvshmem_quiet()
    wp.nvshmem_barrier_all()


# Allocate
out_pe = wp.zeros(1, dtype=wp.int32, device=device)
out_npes = wp.zeros(1, dtype=wp.int32, device=device)
buf = wp.zeros(4, dtype=wp.float32, device=device, symmetric=True)

# Launch
wp.launch(nvshmem_test_kernel, dim=1, inputs=[out_pe, out_npes, buf], device=device)
wp.synchronize_device(device)

# Host barrier to make sure all PEs have completed
from nvshmem.bindings import nvshmem as _nvshmem_bindings  # noqa: E402

_nvshmem_bindings.barrier_all()

# Check results
pe_val = out_pe.numpy()[0]
npes_val = out_npes.numpy()[0]
buf_val = buf.numpy()

print(f"[PE {my_pe}] Kernel: pe={pe_val}, n_pes={npes_val}, buf[0]={buf_val[0]}", flush=True)

passed = True
if pe_val != my_pe:
    print(f"[PE {my_pe}] FAIL: pe mismatch (expected {my_pe}, got {pe_val})", flush=True)
    passed = False
if npes_val != n_pes:
    print(f"[PE {my_pe}] FAIL: n_pes mismatch (expected {n_pes}, got {npes_val})", flush=True)
    passed = False
if my_pe == 1 and n_pes >= 2:
    if abs(buf_val[0] - 42.0) > 1e-6:
        print(f"[PE {my_pe}] FAIL: float_p didn't arrive (expected 42.0, got {buf_val[0]})", flush=True)
        passed = False

if passed:
    print(f"[PE {my_pe}] ALL PASSED", flush=True)

nvshmem.finalize()
MPI.Finalize()
