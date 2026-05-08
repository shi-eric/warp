# NVSHMEM Integration (Initial PoC)

**Status**: In Progress

## Motivation

Warp supports multi-GPU usage at the host level (arrays on different devices, per-device kernel launches, P2P access),
but has no in-kernel inter-GPU communication.
Data exchange between GPUs requires returning to the host, copying, and re-launching.

NVSHMEM is a partitioned global address space (PGAS) library that enables GPU threads to directly read/write
memory on remote GPUs without going through the CPU.
Integrating NVSHMEM into Warp kernels lets users write fused compute-communicate patterns
(e.g., domain-decomposed simulations where boundary exchange happens inside the kernel).

The existing `warp/examples/distributed/example_jacobi_mpi.py` demonstrates the host-level pattern with mpi4py.
This work enables an equivalent example using NVSHMEM for in-kernel communication,
with no mpi4py dependency.

## Requirements

| ID  | Requirement                                                       | Priority | Notes                                                        |
| --- | ----------------------------------------------------------------- | -------- | ------------------------------------------------------------ |
| R1  | Device-side NVSHMEM builtins callable from `@wp.kernel` functions | Must     | Minimal set: PE queries, scalar/bulk put, sync                |
| R2  | `libnvshmem_device.ltoir` linked via nvJitLink at runtime         | Must     | Built from NVSHMEM source for PoC, shipped as file            |
| R3  | `symmetric=True` parameter on `wp.array` for NVSHMEM allocation   | Must     | Native C via `dlopen(RTLD_NOLOAD)` into nvshmem4py's host lib |
| R4  | Automatic `nvshmemx_cumodule_init` after loading NVSHMEM modules  | Must     | Native C, same dlopen approach as R3                          |
| R5  | Clear error at compile time if nvshmem4py is not installed        | Must     | `import nvshmem` check during kernel compilation              |
| R6  | Jacobi solver example using pure NVSHMEM (no mpi4py)             | Should   | Example written but not yet end-to-end tested                 |
| R7  | Packman integration for NVSHMEM distribution                     | Must     | Headers + version.h for LTOIR build, Linux x86_64 + sbsa      |
| R8  | License file for NVSHMEM                                         | Must     | `licenses/libnvshmem-LICENSE.txt`                             |

**Non-goals** (explicitly out of scope for this PoC):

- Autodiff through NVSHMEM operations
- Block-cooperative or warp-cooperative variants (`nvshmemx_*_block`, `nvshmemx_*_warp`)
- Signaling (`put_signal`, `signal_op`, `wait_until`)
- Get operations (`nvshmem_float_g`, bulk get)
- Cooperative kernel launch
- Tile-level NVSHMEM operations
- Windows or macOS support
- Warp-level wrappers for init/finalize (users call nvshmem4py)
- Version compatibility checking between device LTOIR and nvshmem4py
- Embedding the LTOIR into `warp.so` (shipped as a file for now)

## Design

### Approach

Ship `libnvshmem_device.ltoir` in `warp/bin/`.
At runtime, Warp's codegen emits `extern "C"` forward declarations for NVSHMEM device functions,
compiles the kernel to LTOIR via NVRTC, and links it with `libnvshmem_device.ltoir` via nvJitLink.
Host-side operations (init, malloc, cumodule_init) call into `libnvshmem_host.so` via
`dlopen(RTLD_NOLOAD)`, reusing the library that nvshmem4py loaded during `nvshmem.init()`.

**Where the LTOIR comes from**: The NVSHMEM 3.6.x distribution does not include it.
A standalone script (`tools/build_nvshmem_device_ltoir.sh`) builds it via cmake from the
NVSHMEM source. It supports two modes: local build from a source checkout (`NVSHMEM_SRC` env var),
or a docker-based build using `nvcr.io/nvidia/nvhpc` (no local dependencies needed).
Starting with NVSHMEM 3.7, the LTOIR will be available in the binary distribution
(`developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/`).

`build_lib.py` does not build the LTOIR itself. It expects a pre-built file, passed via
`--nvshmem-ltoir` or already present in `warp/bin/`.

**Open question for production**: Should Warp embed the `.ltoir` in its wheel (adding ~7.5 MB),
or require users to install it separately? Embedding is simpler for users but couples Warp releases
to a specific NVSHMEM version. Requiring a separate install avoids the coupling but adds friction
(users would need to add NVIDIA's apt index or pip index and install additional packages).
This decision can be deferred until NVSHMEM 3.7 ships.

**TODO: Version diagnostics.** Record the NVSHMEM version that the shipped `.ltoir` was built from
(e.g., via `wp.nvshmem_version()` or embedding it in `warp.so`). This would help diagnose
ABI mismatches between the device LTOIR and the host library from nvshmem4py.
The coupling is on the device state struct layout (`nvshmemi_device_host_state_t`, 776 bytes
with `static_assert` size checks). Within NVSHMEM 3.x this is likely stable.

### Alternatives Considered and Rejected

**Direct `.bc` linking (original plan)**.
The NVSHMEM distribution ships `libnvshmem_device.bc` (LLVM bitcode).
We initially planned to pass it directly to nvJitLink.
nvJitLink cannot consume LLVM bitcode in any input type (`LTOIR`, `OBJECT`, `FATBIN`, `ANY`).
LLVM bitcode (magic `BC\xc0\xde`) is a different format from NVIDIA LTOIR (magic `\x7fNCE`).
Even `NVJITLINK_INPUT_ANY` (value 10, added in CUDA 13) rejects `.bc` files.

**Shim compilation**.
Compile a thin `extern "C"` wrapper `.cu` with real NVSHMEM headers, producing LTOIR via `nvcc -dlto -fatbin`.
The shim successfully compiled and the wrappers linked with kernel LTOIR via nvJitLink.
However, the shim's internal calls to NVSHMEM transport functions (e.g., `nvshmemi_transfer_rma_p`)
remained unresolved because the implementations only exist in the `.bc` file.
Linking the `.bc` into the shim at build time via `nvcc -dlink` produced a fully-resolved cubin
that no longer exported the wrapper symbols for nvJitLink to find.
The shim approach was ultimately unnecessary once we built the LTOIR from source.

**dlopen for `libnvshmem_host.so`**.
The NVSHMEM distribution ships only `libnvshmem_host.so` (shared), not `.a` (static).
Initially attempted `dlopen` at runtime, but the system may have multiple copies at different paths
(system install, nvshmem4py bundle) causing version mismatches.
Resolved by linking at build time against the Packman distribution's `.so` and shipping it in `warp/bin/`
with the existing `$ORIGIN` rpath. The NVSHMEM host headers cannot be included by g++ (CCCL dependency),
so the five needed C functions are forward-declared instead.

**Embedding LTOIR in `warp.so` via `ld -r -b binary`**.
Worked mechanically (the binary data was linked in) but the retrieved data
didn't match the original file byte-for-byte, causing nvJitLink to reject it.
Not worth debugging for a PoC. Ship the LTOIR as a file instead.

### Key Implementation Details

#### Build System

**LTOIR prerequisite**: `libnvshmem_device.ltoir` must be pre-built before running `build_lib.py`.
The standalone script `tools/build_nvshmem_device_ltoir.sh` handles this via cmake
(either from a local NVSHMEM source checkout or a docker container).
The cmake build uses `-DNVSHMEM_IBGDA_SUPPORT=OFF` to avoid InfiniBand header dependencies
and `-DCMAKE_CUDA_ARCHITECTURES=80` (NVSHMEM's minimum supported arch; nvJitLink recompiles
the LTOIR for the actual target arch at kernel link time).

**Packman dependency** (`deps/libnvshmem-deps.packman.xml`):
Provides headers for `WP_ENABLE_NVSHMEM` compilation.
Download URLs use `developer.download.nvidia.com` (not `developer.nvidia.com`).
The Packman archive extracts into a version-specific subdirectory
(e.g., `libnvshmem-linux-x86_64-3.6.5_cuda13-archive/`). The build system
navigates into this subdirectory to find `include/`.

**`build_lib.py` changes**:

- `find_nvshmem()`: checks `NVSHMEM_HOME`, then Packman. Returns `None` on failure instead of raising.
- `--use-nvshmem` / `--no-use-nvshmem` flags
- `--nvshmem-ltoir`: path to pre-built `libnvshmem_device.ltoir` (or `NVSHMEM_DEVICE_LTOIR` env var)
- `WP_ENABLE_NVSHMEM` preprocessor flag
- Copies LTOIR to `warp/bin/` if `--nvshmem-ltoir` is provided; uses existing file if already present

#### Device-Side Builtins

Six builtins registered in `builtins.py` with `namespace=""` and `export=False`:

| Builtin                                       | Warp Signature                                   | C Symbol in LTOIR     |
| --------------------------------------------- | ------------------------------------------------ | --------------------- |
| `wp.nvshmem_my_pe()`                          | `() -> int`                                      | `nvshmem_my_pe`       |
| `wp.nvshmem_n_pes()`                          | `() -> int`                                      | `nvshmem_n_pes`       |
| `wp.nvshmem_float_p(dest, offset, value, pe)` | `(array(float), int, float, int) -> None`        | `nvshmem_float_p`     |
| `wp.nvshmem_float_put(dest, src, nelems, pe)` | `(array(float), array(float), int, int) -> None` | `nvshmem_float_put`   |
| `wp.nvshmem_quiet()`                          | `() -> None`                                     | `nvshmem_quiet`       |
| `wp.nvshmem_barrier_all()`                    | `() -> None`                                     | `nvshmem_barrier_all` |

`namespace=""` prevents the default `wp::` prefix in generated C code.
`export=False` prevents CPU export stubs in `exports.h` (NVSHMEM functions are device-only).
`nvshmem_float_p` and `nvshmem_float_put` use custom `dispatch_func` to transform
`wp.array` arguments to raw pointer expressions (e.g., `&var_dest.data[var_offset]`).
Note: use `.data` (dot), not `->data` (arrow). `wp::array_t` is passed by value, not pointer.

#### Codegen and Linking

When any `wp.nvshmem_*` builtin is referenced in a kernel:

1. The `ModuleBuilder.uses_nvshmem` flag is set (detected by checking `func.key.startswith("nvshmem_")`).
2. At compile time, `import nvshmem` is checked. If it fails, `ImportError` is raised.
3. The generated `.cu` file includes `extern "C"` forward declarations
   for the six NVSHMEM symbols.
4. NVRTC compiles with `-dlto` and `--relocatable-device-code=true`
   (gated on `num_ltoirs > 0` in `warp.cu`, which requires `WP_ENABLE_NVSHMEM || WP_ENABLE_MATHDX`).
5. `libnvshmem_device.ltoir` is loaded from `warp/bin/` and appended to `ltoirs_to_link`.
6. nvJitLink links the kernel LTOIR + device LTOIR into the final cubin.

**Output format**: nvJitLink with LTOIR inputs requires cubin output, not PTX.
Set `wp.config.cuda_output = "cubin"` or the linking will fail.
(TODO: make this automatic when NVSHMEM builtins are detected.)

#### Module Registration

After `build.load_cuda()` returns a CUmodule handle,
if the module uses NVSHMEM, Warp calls `wp_nvshmem_cumodule_init(handle)` (native C function
that forward-declares and calls `nvshmemx_cumodule_init` from `libnvshmem_host.so`).
This populates the `__constant__` symbols in the cubin that the NVSHMEM runtime needs
(`nvshmemi_device_state_d`, `nvshmemi_ibgda_device_state_d`, `nvshmemi_device_lib_version_d`).

The native code checks `nvshmemx_init_status() >= 2` before calling `cumodule_init`.
If NVSHMEM is not initialized (e.g., compilation-only test), the call is skipped with a warning.

#### Symmetric Array Allocation

`symmetric=True` on `wp.array` routes allocation through `wp_nvshmem_malloc()`
and deallocation through `wp_nvshmem_free()` (native C functions linked against `libnvshmem_host.so`).

#### Host Library Linking

`libnvshmem_host.so` is shipped in `warp/bin/` and linked at build time.
`warp.so` finds it at runtime via the existing `$ORIGIN` rpath.
The five host functions we need (`nvshmem_malloc`, `nvshmem_free`, `nvshmemx_cumodule_init`,
`nvshmemx_cumodule_finalize`, `nvshmemx_init_status`) are forward-declared in `nvshmem.cpp`
because the NVSHMEM headers cannot be included by g++ (they pull in CCCL/CUDA C++ headers).

**Initialization order matters**: `wp.set_device()` (which creates the CUDA context)
must be called BEFORE `nvshmem.init()`. NVSHMEM's `cumodule_init` fails with
"nvshmem get cucontext failed" if there's no active CUDA context.

#### nvshmem4py API

The Python package is `nvshmem4py-cu13` (or `nvshmem4py-cu12`). Import as `import nvshmem`.

Key APIs used:
- `nvshmem.core.init(device=CudaDevice(rank), mpi_comm=comm, initializer_method="mpi")`: MPI bootstrap
- `nvshmem.core.my_pe()`, `nvshmem.core.n_pes()`: PE queries (host side)
- `nvshmem.core.init_status()`: returns `InitStatus` enum (0=not init, 1=bootstrapped, 2=initialized)
- `nvshmem.bindings.nvshmem.malloc(nbytes)`: symmetric allocation (returns int pointer)
- `nvshmem.bindings.nvshmem.free(ptr)`: symmetric deallocation
- `nvshmem.bindings.nvshmem.cumodule_init(handle)`: register CUmodule
- `nvshmem.bindings.nvshmem.barrier_all()`: host-side barrier (low-level, no stream arg)
- `nvshmem.core.barrier_all()`: higher-level barrier (requires a stream, may fail without one)

Note: `nvshmem.core.barrier_all()` requires a `cuda.core` stream object.
For simple use, `nvshmem.bindings.nvshmem.barrier_all()` works without one.

## Pitfalls and Lessons Learned

### nvJitLink Cannot Consume LLVM Bitcode

The original design assumed `libnvshmem_device.bc` could be passed to nvJitLink.
It cannot. nvJitLink rejects LLVM bitcode (`.bc`) in every input type,
including `NVJITLINK_INPUT_ANY`. LLVM bitcode and NVIDIA LTOIR are different formats
despite both being based on LLVM IR. The solution is to build `libnvshmem_device.ltoir`
from NVSHMEM source using `nvcc --ltoir`.

### The Packman Distribution Does Not Include LTOIR (Pre-3.7)

The NVSHMEM 3.6.x Packman distribution ships `.bc` (LLVM bitcode) and `.a` (host objects, ELF),
but NOT `.ltoir`. The LTOIR must be built from source via cmake with `NVSHMEM_BUILD_LTOIR_LIBRARY=ON`
and `NVSHMEM_IBGDA_SUPPORT=OFF` (to avoid InfiniBand header dependencies).
See `tools/build_nvshmem_device_ltoir.sh` for a reproducible build (local or docker-based).
Starting with NVSHMEM 3.7, the `.ltoir` will be included in the distribution.

### Multiple `libnvshmem_host.so` Versions on the System

The system may have multiple copies of `libnvshmem_host.so`:
an older system-wide install (e.g., `/lib/x86_64-linux-gnu/libnvshmem_host.so.3`),
a CUDA-version-specific install (`/usr/lib/x86_64-linux-gnu/nvshmem/13/`),
and the one bundled with nvshmem4py. Using `dlopen("libnvshmem_host.so")` from C code
finds whichever the system linker resolves first, which may differ from nvshmem4py's copy.
Solution: use `dlopen(RTLD_NOLOAD)` to reuse the library nvshmem4py already loaded.

### CUDA Context Must Exist Before NVSHMEM Init

`nvshmem.core.init()` requires an active CUDA context on the target device.
Call `wp.set_device(f"cuda:{rank}")` before `nvshmem.init()`.
Without this, `nvshmemx_cumodule_init` fails with "nvshmem get cucontext failed"
and `nvshmem_malloc` returns NULL.

### Warp Builtins Need `export=False`

Without `export=False`, the build system generates CPU export stubs in `exports.h`
that reference `wp::nvshmem_my_pe()` etc. These don't exist (NVSHMEM is device-only)
and cause compilation errors. The `exports.h` must be regenerated after adding the flag
(`python -c "from build_lib import generate_exports_header_file; generate_exports_header_file('.')"`).

### Array Dispatch Uses Dot, Not Arrow

Warp kernel arguments of type `wp::array_t<float>` are structs passed by value.
The `dispatch_func` must use `var_dest.data` (dot access), not `var_dest->data` (arrow/pointer).
Using arrow causes "operator -> applied to non-pointer type" NVRTC compilation errors.

### nvJitLink LTO Path Gated on `WP_ENABLE_MATHDX`

Warp's `warp.cu` gates the LTOIR compilation and nvJitLink linking paths on `#if WP_ENABLE_MATHDX`.
NVSHMEM-enabled kernels need this path even without mathdx.
The guard was extended to `#if WP_ENABLE_MATHDX || WP_ENABLE_NVSHMEM`.

### Per-Rank Kernel Cache Needed for Multi-Process Runs

When multiple MPI ranks compile Warp kernels simultaneously, they race on the shared kernel cache.
Each rank needs its own cache directory: `os.environ["WARP_CACHE_ROOT"] = f"/tmp/warp_rank{rank}"`.

### cubin Output Required for nvJitLink

When nvJitLink links LTOIR inputs, the output must be cubin, not PTX.
Warp defaults to PTX output on some configurations.
Set `wp.config.cuda_output = "cubin"` before loading NVSHMEM-enabled modules.
(TODO: make this automatic.)

## Testing Strategy

**Unit tests** (`warp/tests/test_nvshmem.py`):

- Test that NVSHMEM builtins compile without error (codegen + nvJitLink linking).
- Test `symmetric=True` array allocation error paths (CPU rejection, non-NVSHMEM build rejection).
- Multi-PE tests (PE queries, float_p) launched via `mpirun` or `nvshmrun` subprocess.

**End-to-end test** (`/tmp/test_nvshmem_basic.py`):
Single kernel testing PE queries, symmetric allocation, and `nvshmem_float_p` across 2 PEs.
Verified: PE 0 writes 42.0 to PE 1's buffer via in-kernel `nvshmem_float_p`.

**Build test**: `--no-use-nvshmem` produces a working build with stubs.
