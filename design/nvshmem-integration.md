# NVSHMEM Integration

**Status**: Prototype implemented; default NVSHMEM 3.7.1 enablement pending CUDA 13.4 validation

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
| R7  | Packman integration for NVSHMEM distribution                     | Must     | Keep 3.6.5 until the default toolkit can consume 3.7.1        |
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

## Design

### Approach

The implementation embeds either `libnvshmem_device.ltoir.fatbin` or `libnvshmem_device.ltoir` in a
read-only section of `warp.so`. At runtime, Warp's codegen emits `extern "C"` forward declarations
for NVSHMEM device functions, compiles the kernel to LTOIR via NVRTC, and passes the embedded memory
directly to nvJitLink.
Host-side operations (init, malloc, cumodule_init) call into `libnvshmem_host.so` via
`dlopen(RTLD_NOLOAD)`, reusing the library that nvshmem4py loaded during `nvshmem.init()`.

**Where the LTOIR comes from**: The NVSHMEM 3.6.x distribution does not include it, so the PoC uses
`tools/build_nvshmem_device_ltoir.sh` to build it from source. The official NVSHMEM 3.7.1 CUDA 13
archive now includes architecture-specific raw LTOIR files and a multi-architecture LTOIR fatbin.
The source-build script remains useful for 3.6.x and custom NVSHMEM builds, but is no longer needed
when consuming the 3.7.1 distribution.

`build_lib.py` does not build the device input itself. It automatically discovers the packaged fatbin
under `NVSHMEM_HOME`, accepts one explicitly through `--nvshmem-fatbin`, or accepts a raw LTOIR file
through `--nvshmem-ltoir`. The selected bytes are staged under `_build`, embedded with an assembly
`.incbin` directive, and verified byte-for-byte through the accessor exported by the finished library.

**Implemented device-library direction**: Consume the official 3.7.1
`libnvshmem_device.ltoir.fatbin` instead of building and shipping a single raw LTOIR file. The fatbin
is multi-architecture and smaller than any individual raw LTOIR file. Warp's general nvJitLink path
and NVSHMEM-specific build path preserve it as `NVJITLINK_INPUT_FATBIN`. Selecting one device input
removes legacy sidecar inputs so an earlier build cannot silently supply device code from another
NVSHMEM version.

The approximately 3.9 MB fatbin increases `warp.so` and the wheel by the same payload that a sidecar
file would require. Embedding makes Warp and its NVSHMEM device code an atomic, versioned artifact.

**Version validation.** Warp records the NVSHMEM header version in its native library and queries
`nvshmemx_vendor_get_version_info` from the host library already loaded by nvshmem4py. The versions
must match exactly before `nvshmemx_cumodule_init` runs. The coupling is on device state and constant
symbol layouts; do not assume that all NVSHMEM 3.x releases are interchangeable.

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
    --cuda-path /usr/local/cuda-13.3
```

This command discovers `libnvshmem_device.ltoir.fatbin` in the 3.7.1 installation and preserves its
input type through nvJitLink. `--nvshmem-fatbin` can select a fatbin outside `NVSHMEM_HOME`; the raw
`--nvshmem-ltoir` workflow remains available for older or custom NVSHMEM builds.

The 3.7.1 CUDA 13 archive was produced with CUDA 13.2. CUDA 13.0 nvJitLink rejects its device fatbin
with `ERROR 4 in nvvmAddNVVMContainerToProgram`, so Warp cannot yet update the default Packman pin
from NVSHMEM 3.6.5 while retaining CUDA 13.0. `build_lib.py` reads the producer toolkit from
`nvshmem_version.h` and rejects an older Warp toolkit during the build instead of deferring this
failure to kernel compilation. CUDA 13.3 remains an opt-in validation path until CUDA 13.4 fixes the
Blackwell Warp-module compiler regression.

Until that default-toolkit constraint is resolved, the standard CUDA 13.0 build uses the Packman
NVSHMEM 3.6.5 headers and host integration but embeds no device payload unless a compatible raw
LTOIR file is supplied explicitly. The build emits a warning, the native device-library accessor
returns no payload, and tests that require NVSHMEM device builtins skip. Host-side NVSHMEM support,
including symmetric allocation after NVSHMEM initialization, remains compiled in.

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
uses `dlopen(RTLD_NOLOAD)` to find the already-loaded `libnvshmem_host.so` and resolves the six
required symbols with `dlsym`. This makes initialization order explicit and avoids loading a second,
potentially incompatible host runtime.

**Embedding LTOIR via `ld -r -b binary`**.
The initial embedding experiment linked data into `warp.so`, but the retrieved byte range did not
match the original file and nvJitLink rejected it. The production design instead uses `.incbin` with
explicit start and end symbols, then hashes the exported bytes after linking. This preserves the
single-binary benefit while making corruption a build failure.

### Key Implementation Details

#### Build System

**Device-library prerequisite**: For NVSHMEM 3.7.1, `build_lib.py` automatically uses the packaged
multi-architecture fatbin. For NVSHMEM 3.6.x, `tools/build_nvshmem_device_ltoir.sh` produces raw
LTOIR via cmake from a local checkout or a container. The cmake fallback uses
`-DNVSHMEM_IBGDA_SUPPORT=OFF` to avoid InfiniBand header dependencies and
`-DCMAKE_CUDA_ARCHITECTURES=80`.

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
- `--nvshmem-fatbin`: path to `libnvshmem_device.ltoir.fatbin` (or `NVSHMEM_DEVICE_FATBIN` env var)
- `WP_ENABLE_NVSHMEM` preprocessor flag
- Prefers a packaged fatbin and embeds the selected input into a read-only `warp.so` section
- Verifies the embedded bytes and input kind after linking, and removes legacy sidecar inputs
- Rejects packaged device inputs produced by a newer CUDA Toolkit than the one building Warp

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
4. When either external link inputs or the embedded NVSHMEM input is needed, NVRTC compiles with
   `-dlto` and `--relocatable-device-code=true`. The native path is available when either
   `WP_ENABLE_NVSHMEM` or `WP_ENABLE_MATHDX` is enabled.
5. Python tells `wp_cuda_compile_program` that the module uses NVSHMEM; it does not read or copy the
   embedded bytes.
6. Native code retrieves the embedded pointer, selects `NVJITLINK_INPUT_FATBIN` or
   `NVJITLINK_INPUT_LTOIR` from the compiled input kind, and links it with the kernel LTOIR.

**Output format**: nvJitLink with LTOIR inputs requires cubin output, not PTX.
Set `wp.config.cuda_output = "cubin"` or the linking will fail.

**Remaining work**: Automatically select or require cubin output when NVSHMEM builtins are detected,
so users cannot accidentally request an incompatible PTX output path.

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

Immediately before that call, Warp compares the NVSHMEM version compiled into `warp.so` with the
version reported by the loaded host library. A mismatch raises a `RuntimeError` with both versions;
native code also refuses the initialization call as a defense in depth.

#### Symmetric Array Allocation

`symmetric=True` on `wp.array` routes allocation through `wp_nvshmem_malloc()`
and deallocation through `wp_nvshmem_free()` (native C functions resolved from the already-loaded
`libnvshmem_host.so`).

#### Host Library Loading

`libnvshmem_host.so` is loaded by nvshmem4py. Warp uses `dlopen(RTLD_NOLOAD)` to reuse that exact
loaded library, then resolves `nvshmem_malloc`, `nvshmem_free`, `nvshmemx_cumodule_init`,
`nvshmemx_cumodule_finalize`, `nvshmemx_init_status`, and `nvshmemx_vendor_get_version_info` with
`dlsym`. The 3.7.1 smoke test verified that calls resolved to `libnvshmem_host.so.3.7.1` from the
selected archive. The functions are forward-declared in `nvshmem.cpp` because the NVSHMEM headers
cannot be included by g++ (they pull in CCCL/CUDA C++ headers).

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
files use `NVJITLINK_INPUT_LTOIR`. Warp compiles the selected input kind into `warp.so` alongside the
bytes and uses that value at runtime; renaming a fatbin to look like raw LTOIR remains invalid.

### Keep the Host Library and Device Input on the Same NVSHMEM Version

In the test environment, `uv run --with nvshmem4py-cu13` resolved nvshmem4py 0.3.1 together with
`nvidia-nvshmem-cu13` 3.4.5, not NVSHMEM 3.7.1. Using that host library with 3.7.1 device input would
create the version mismatch described above. The 3.7.1 end-to-end test explicitly selected
the extracted 3.7.1 host library through `NVSHMEM_HOME` and the dynamic loader path; `/proc/self/maps`
confirmed that `libnvshmem_host.so.3.7.1` was loaded. Warp now rejects this mismatch before module
initialization, but environments should still pin matching packages so initialization can succeed.

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

### The nvJitLink Dependency Must Follow Both Features

Warp originally gated both its nvJitLink helper and linker dependency on MathDx. NVSHMEM-enabled
kernels need nvJitLink even when MathDx is disabled. Both the compile-time guard and native linker
inputs now use `WP_ENABLE_MATHDX || WP_ENABLE_NVSHMEM`; a CUDA 13.3 build with
`--no-use-libmathdx` verified that an NVSHMEM kernel still compiles from the embedded fatbin.

### Per-Rank Kernel Cache Needed for Multi-Process Runs

When multiple processes compile Warp kernels simultaneously, they race on the shared kernel cache.
Each rank needs its own `WARP_CACHE_PATH` and `WARP_CACHE_ROOT`. The UID test creates rank-specific
directories under a temporary test directory before importing Warp.

### Constrain the Transport for a Local P2P Test

The 3.7.1 library selected the `ibrc` remote transport by default on the GKE test node and waited for
InfiniBand support that was not present. For a same-host test, `NVSHMEM_REMOTE_TRANSPORT=none`
disables network transport discovery while retaining the local CUDA IPC/P2P path:

```bash
export NVSHMEM_REMOTE_TRANSPORT=none
export NVSHMEM_DISABLE_LOCAL_ONLY_PROXY=1
export NVSHMEM_DISABLE_NCCL=1
export NVSHMEM_DISABLE_NVLS=1
export NVSHMEM_SYMMETRIC_SIZE=67108864
```

On the tested RTX PRO 6000 Blackwell system, two ranks were assigned to separate 1g.24gb MIG slices.
NVSHMEM classified them as multiple PEs on one physical GPU (MPG) and warned that only limited MPG
support was available without MPS. It nevertheless reported both PEs in its P2P connected list,
opened CUDA IPC handles between the slices, and completed device-side PE queries, a barrier, and an
`nvshmem_float_p` from PE 0 to PE 1. This configuration validates local P2P behavior only; multi-node
testing must select a network transport appropriate for the target system.

### Cache Identity Does Not Persist Runtime Initialization State

The module hash determines whether a compiled artifact is reusable, but it does not populate fields
on a new `Module` object. Before `uses_nvshmem` was stored in module metadata, a fresh process could
load a cached NVSHMEM cubin without calling `nvshmemx_cumodule_init`. The 3.7.1 cache-hit test exposed
this distinction and verifies both metadata restoration and the module-initialization attempt.

### cubin Output Required for nvJitLink

When nvJitLink links LTOIR inputs, the output must be cubin, not PTX.
Warp defaults to PTX output on some configurations.
Set `wp.config.cuda_output = "cubin"` before loading NVSHMEM-enabled modules.

## Testing Strategy

**Unit tests** (`warp/tests/distributed/test_nvshmem.py`):

- Test that NVSHMEM builtins compile without error (codegen + nvJitLink linking).
- Test that a cache hit in a fresh `Module` restores `uses_nvshmem` and attempts module initialization.
- Test `symmetric=True` array allocation error paths (CPU rejection, non-NVSHMEM build rejection).
- Multi-PE tests (PE queries, float_p) launched via `mpirun` or `nvshmrun` subprocess.
- Test two-PE UID bootstrap and `nvshmem_float_p` over local P2P with remote transports disabled.

**End-to-end validation**:

- A one-PE UID-bootstrap smoke test loads the NVSHMEM 3.7.1 host library, allocates symmetric Warp
  arrays, compiles a kernel from the embedded fatbin, and verifies device-side PE 0 of one PE.
- The checked-in `TestNvshmemMultiPE.test_uid_local_p2p` test verifies that PE 0 writes 42.0 to PE 1's
  symmetric buffer through in-kernel `nvshmem_float_p` without MPI or a network transport.

Run the non-IBRC test after building with the opt-in CUDA Toolkit and NVSHMEM 3.7.1 archive:

```bash
export NVSHMEM_HOME=/path/to/libnvshmem-linux-x86_64-3.7.1_cuda13-archive
WARP_CACHE_PATH=/tmp/warp-cache-nvshmem-uid-parent \
WARP_CACHE_ROOT=/tmp/warp-cache-nvshmem-uid-parent \
uv run --with nvshmem4py-cu13 -m unittest \
    warp.tests.distributed.test_nvshmem.TestNvshmemMultiPE.test_uid_local_p2p
```

The test discovers `NVSHMEM_HOME/bin/nvshmrun.hydra`, prepends the matching host-library directory,
uses an atomic temporary file to share the UID, and assigns one local rank to each CUDA device.

**NVSHMEM 3.7.1 validation**:

- Full standard Warp build with `--cuda-path /usr/local/cuda-13.3` and the packaged SM 120 raw LTOIR.
- Direct nvJitLink probe using the packaged multi-architecture fatbin as `NVJITLINK_INPUT_FATBIN`.
- Automatic discovery, byte-exact embedding, and native runtime routing of the packaged fatbin.
- Wheel build containing the embedded payload in `warp.so` with no NVSHMEM device sidecar.
- Full build and NVSHMEM kernel compile with MathDx disabled.
- Early rejection when CUDA 13.0 is asked to consume the CUDA 13.2-produced 3.7.1 device input.
- Rejection of a loaded host library whose NVSHMEM version differs from Warp's build version.
- One-PE UID initialization with the 3.7.1 host library, symmetric allocation, and a real Warp kernel
  returning PE 0 and one total PE.
- Checked-in two-PE UID test across two MIG slices with `NVSHMEM_REMOTE_TRANSPORT=none`, including a
  device-side write from PE 0 to PE 1 over CUDA IPC/P2P.
- Fresh-process cache hit that loads the cubin and executes the NVSHMEM module-initialization path.
- Full standard rebuild with the default CUDA 13.0 toolkit after the opt-in test; the native accessor
  reports no embedded payload and device-dependent tests skip as expected.

**Build test**: `--no-use-nvshmem` produces a working build with stubs.
