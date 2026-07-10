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
| R2  | NVSHMEM device code linked via nvJitLink at runtime              | Must     | 3.7.1 ships raw LTOIR files and a multi-architecture fatbin   |
| R3  | `symmetric=True` parameter on `wp.array` for NVSHMEM allocation   | Must     | Native C via `dlopen(RTLD_NOLOAD)` into nvshmem4py's host lib |
| R4  | Automatic `nvshmemx_cumodule_init` after loading NVSHMEM modules  | Must     | Native C, same dlopen approach as R3                          |
| R5  | Clear error at compile time if nvshmem4py is not installed        | Must     | `import nvshmem` check during kernel compilation              |
| R6  | Jacobi solver example using pure NVSHMEM (no mpi4py)             | Should   | Example written but not yet end-to-end tested                 |
| R7  | Packman integration for NVSHMEM distribution                     | Must     | Currently 3.6.5; moving to 3.7.1 removes the LTOIR build step |
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

The current implementation ships `libnvshmem_device.ltoir` in `warp/bin/`.
At runtime, Warp's codegen emits `extern "C"` forward declarations for NVSHMEM device functions,
compiles the kernel to LTOIR via NVRTC, and links it with `libnvshmem_device.ltoir` via nvJitLink.
Host-side operations (init, malloc, cumodule_init) call into `libnvshmem_host.so` via
`dlopen(RTLD_NOLOAD)`, reusing the library that nvshmem4py loaded during `nvshmem.init()`.

**Where the LTOIR comes from**: The NVSHMEM 3.6.x distribution does not include it, so the PoC uses
`tools/build_nvshmem_device_ltoir.sh` to build it from source. The official NVSHMEM 3.7.1 CUDA 13
archive now includes architecture-specific raw LTOIR files and a multi-architecture LTOIR fatbin.
The source-build script remains useful for 3.6.x and custom NVSHMEM builds, but is no longer needed
when consuming the 3.7.1 distribution.

`build_lib.py` does not build the LTOIR itself. It expects a pre-built file, passed via
`--nvshmem-ltoir` or already present in `warp/bin/`.

**Recommended production direction**: Consume the official 3.7.1
`libnvshmem_device.ltoir.fatbin` instead of building and shipping a single raw LTOIR file. The fatbin
is multi-architecture and smaller than any individual raw LTOIR file. Warp's general nvJitLink path
already supports `NVJITLINK_INPUT_FATBIN`; the remaining work is to teach the NVSHMEM-specific build
and runtime path to discover, copy, and submit this file as a fatbin rather than as raw LTOIR.

Whether the fatbin belongs in the Warp wheel or in a separately installed NVSHMEM package remains a
packaging decision. Embedding approximately 3.9 MB is simpler for users but couples the wheel to an
NVSHMEM version. Discovering an external install avoids that payload and coupling, but requires
reliable package discovery and clear version diagnostics.

**TODO: Version diagnostics.** Record the NVSHMEM version that the shipped `.ltoir` was built from
(e.g., via `wp.nvshmem_version()` or embedding it in `warp.so`). This would help diagnose
ABI mismatches between the device LTOIR and the host library from nvshmem4py.
The coupling is on the device state struct layout (`nvshmemi_device_host_state_t`, 776 bytes
with `static_assert` size checks). Do not assume that all NVSHMEM 3.x releases are interchangeable;
the 3.7.1 testing found that Python package resolution can select a different host-library version.

### NVSHMEM 3.7.1 Distribution Findings

The [official NVSHMEM 3.7.1 Linux x86_64 CUDA 13 archive](https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.7.1_cuda13-archive.tar.xz)
contains:

| Artifact                               | Architectures                | Size      |
| -------------------------------------- | ---------------------------- | --------- |
| `libnvshmem_device.ltoir.fatbin`       | Multi-architecture           | 3,899,400 bytes |
| `libnvshmem_device_sm_75.ltoir`        | SM 75                        | 7,760,100 bytes |
| `libnvshmem_device_sm_80.ltoir`        | SM 80                        | 7,680,796 bytes |
| `libnvshmem_device_sm_89.ltoir`        | SM 89                        | 7,680,796 bytes |
| `libnvshmem_device_sm_90.ltoir`        | SM 90                        | 8,614,236 bytes |
| `libnvshmem_device_sm_100.ltoir`       | SM 100                       | 8,221,584 bytes |
| `libnvshmem_device_sm_120.ltoir`       | SM 120                       | 8,221,584 bytes |

The SM 120 raw LTOIR was verified by a full Warp build with CUDA 13.3 and an SM 120 Blackwell target.
The packaged multi-architecture fatbin was also passed directly through Warp's existing fatbin input
path and linked successfully into a cubin. This confirms that both packaged forms are usable; the
fatbin is the better distribution artifact because it avoids selecting one architecture at build
time and has the smallest payload.

CUDA 13.3 was installed and exercised only as an opt-in toolkit through
`--cuda-path /usr/local/cuda-13.3`. Warp's default build remains on CUDA 13.0 because CUDA 13.1
through 13.3 have a known Blackwell Warp-module compilation bug that is expected to be fixed in
CUDA 13.4. A representative opt-in build is:

```bash
export NVSHMEM_HOME=/path/to/libnvshmem-linux-x86_64-3.7.1_cuda13-archive
WARP_CACHE_PATH=/tmp/warp-cache-warp-worktree-1-nvshmem-3.7.1 \
WARP_CACHE_ROOT=/tmp/warp-cache-warp-worktree-1-nvshmem-3.7.1 \
uv run build_lib.py \
    --cuda-path /usr/local/cuda-13.3 \
    --nvshmem-ltoir "$NVSHMEM_HOME/lib/libnvshmem_device_sm_120.ltoir"
```

This command verifies the current raw-LTOIR workflow. A future fatbin-aware option should accept
`libnvshmem_device.ltoir.fatbin` directly and preserve its input type through nvJitLink.

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

**Direct linking to `libnvshmem_host.so`**.
The NVSHMEM distribution ships only a shared host library. Linking Warp against one packaged copy
would not guarantee that it matches the library loaded by nvshmem4py. The implementation therefore
uses `dlopen(RTLD_NOLOAD)` to find the already-loaded `libnvshmem_host.so` and resolves the five
required symbols with `dlsym`. This makes initialization order explicit and avoids loading a second,
potentially incompatible host runtime.

**Embedding LTOIR in `warp.so` via `ld -r -b binary`**.
Worked mechanically (the binary data was linked in) but the retrieved data
didn't match the original file byte-for-byte, causing nvJitLink to reject it.
Not worth debugging for a PoC. Ship the LTOIR as a file instead.

### Key Implementation Details

#### Build System

**LTOIR prerequisite**: `libnvshmem_device.ltoir` must be available before running `build_lib.py`.
For NVSHMEM 3.6.x, `tools/build_nvshmem_device_ltoir.sh` produces it via cmake from a local checkout
or a container. For NVSHMEM 3.7.1, pass one of the packaged architecture-specific files to
`--nvshmem-ltoir`. The cmake fallback uses `-DNVSHMEM_IBGDA_SUPPORT=OFF` to avoid InfiniBand header
dependencies and `-DCMAKE_CUDA_ARCHITECTURES=80`.

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

`ModuleBuilder.uses_nvshmem` must also survive a cache hit in a fresh process. Codegen does not run
on that path, so cache identity alone cannot reconstruct the in-memory flag. Warp stores the flag as
`__uses_nvshmem__` in the module metadata, restores it before loading the CUDA module, and removes the
private key before constructing `ModuleExec`. This ensures a cached module still receives
`nvshmemx_cumodule_init`.

#### Symmetric Array Allocation

`symmetric=True` on `wp.array` routes allocation through `wp_nvshmem_malloc()`
and deallocation through `wp_nvshmem_free()` (native C functions resolved from the already-loaded
`libnvshmem_host.so`).

#### Host Library Loading

`libnvshmem_host.so` is loaded by nvshmem4py. Warp uses `dlopen(RTLD_NOLOAD)` to reuse that exact
loaded library, then resolves `nvshmem_malloc`, `nvshmem_free`, `nvshmemx_cumodule_init`,
`nvshmemx_cumodule_finalize`, and `nvshmemx_init_status` with `dlsym`. The 3.7.1 smoke test verified
that calls resolved to `libnvshmem_host.so.3.7.1` from the selected archive. The functions are
forward-declared in `nvshmem.cpp` because the NVSHMEM headers cannot be included by g++ (they pull
in CCCL/CUDA C++ headers).

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
NVSHMEM 3.7.1 resolves this packaging gap by including raw LTOIR files for SM 75, 80, 89, 90,
100, and 120, plus `libnvshmem_device.ltoir.fatbin`.

### The Packaged Fatbin Needs the Correct nvJitLink Input Type

The 3.7.1 `libnvshmem_device.ltoir.fatbin` is not interchangeable with a raw `.ltoir` file at the
API boundary. It must be submitted as `NVJITLINK_INPUT_FATBIN`, while the architecture-specific
files use `NVJITLINK_INPUT_LTOIR`. Warp already supports both input types in its general linker path,
but the NVSHMEM-specific path currently opens `warp/bin/libnvshmem_device.ltoir` and appends its bytes
to the raw LTOIR list. Renaming or copying the fatbin to that filename is therefore insufficient.

### Keep the Host Library and Device Input on the Same NVSHMEM Version

In the test environment, `uv run --with nvshmem4py-cu13` resolved nvshmem4py 0.3.1 together with
`nvidia-nvshmem-cu13` 3.4.5, not NVSHMEM 3.7.1. Using that host library with 3.7.1 device input would
silently create the version mismatch described above. The 3.7.1 end-to-end test explicitly selected
the extracted 3.7.1 host library through `NVSHMEM_HOME` and the dynamic loader path; `/proc/self/maps`
confirmed that `libnvshmem_host.so.3.7.1` was loaded. Production tooling should either pin matching
packages or reject a detected host/device version mismatch.

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

### Constrain the Transport for a Local Single-PE Smoke Test

The 3.7.1 library selected the `ibrc` remote transport by default on the GKE test node and waited for
InfiniBand support that was not present. For a local, one-PE validation, the following configuration
avoided unrelated transport and collective dependencies:

```bash
export NVSHMEM_REMOTE_TRANSPORT=none
export NVSHMEM_DISABLE_LOCAL_ONLY_PROXY=1
export NVSHMEM_DISABLE_NCCL=1
export NVSHMEM_DISABLE_NVLS=1
export NVSHMEM_SYMMETRIC_SIZE=67108864
```

This is a smoke-test configuration, not a recommendation for multi-PE or production execution. A
multi-PE test must select a transport appropriate for the target system.

### Cache Identity Does Not Persist Runtime Initialization State

The module hash determines whether a compiled artifact is reusable, but it does not populate fields
on a new `Module` object. Before `uses_nvshmem` was stored in module metadata, a fresh process could
load a cached NVSHMEM cubin without calling `nvshmemx_cumodule_init`. The 3.7.1 cache-hit test exposed
this distinction and verifies both metadata restoration and the module-initialization attempt.

### cubin Output Required for nvJitLink

When nvJitLink links LTOIR inputs, the output must be cubin, not PTX.
Warp defaults to PTX output on some configurations.
Set `wp.config.cuda_output = "cubin"` before loading NVSHMEM-enabled modules.
(TODO: make this automatic.)

## Testing Strategy

**Unit tests** (`warp/tests/distributed/test_nvshmem.py`):

- Test that NVSHMEM builtins compile without error (codegen + nvJitLink linking).
- Test that a cache hit in a fresh `Module` restores `uses_nvshmem` and attempts module initialization.
- Test `symmetric=True` array allocation error paths (CPU rejection, non-NVSHMEM build rejection).
- Multi-PE tests (PE queries, float_p) launched via `mpirun` or `nvshmrun` subprocess.

**End-to-end test** (`/tmp/test_nvshmem_basic.py`):
Single kernel testing PE queries, symmetric allocation, and `nvshmem_float_p` across 2 PEs.
Verified: PE 0 writes 42.0 to PE 1's buffer via in-kernel `nvshmem_float_p`.

**NVSHMEM 3.7.1 validation**:

- Full standard Warp build with `--cuda-path /usr/local/cuda-13.3` and the packaged SM 120 raw LTOIR.
- Direct nvJitLink probe using the packaged multi-architecture fatbin as `NVJITLINK_INPUT_FATBIN`.
- One-PE UID initialization with the 3.7.1 host library, symmetric allocation, and a real Warp kernel
  returning PE 0 and one total PE.
- Fresh-process cache hit that loads the cubin and executes the NVSHMEM module-initialization path.
- Full standard rebuild with the default CUDA 13.0 toolkit after the opt-in test.

**Build test**: `--no-use-nvshmem` produces a working build with stubs.
