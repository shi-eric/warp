# NVSHMEM Integration

**Status**: Implemented. Warp links the device library of the installed NVSHMEM at runtime and requires NVSHMEM 3.9
or newer, or an explicit device library override. Decided and implemented 2026-10-30, replacing a prototype that
embedded the NVSHMEM 3.8.0 device library in `warp.so`; see
[Transition from the Prototype](#transition-from-the-prototype).

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

| ID  | Requirement                                                       | Priority | Notes                                                                     |
| --- | ----------------------------------------------------------------- | -------- | ------------------------------------------------------------------------- |
| R1  | Device-side NVSHMEM builtins callable from `@wp.kernel` functions | Must     | PE queries, element and bulk RMA, signals, sync                           |
| R2  | NVSHMEM device code linked via nvJitLink at runtime               | Must     | Input is `libnvshmem_device.ltoir.fatbin` from the installed NVSHMEM      |
| R3  | Kernels accept symmetric memory allocated through nvshmem4py      | Must     | Wrapped with `wp.array(ptr=...)`; freed explicitly by the user            |
| R4  | Automatic `nvshmemx_cumodule_init` after loading NVSHMEM modules  | Must     | Native C via `dlopen(RTLD_NOLOAD)` into the loaded host lib               |
| R5  | Clear error when NVSHMEM is not usable                            | Must     | No device library found (compile); host runtime missing or too old (load) |
| R6  | Jacobi solver example using pure NVSHMEM (no mpi4py)              | Should   | `example_jacobi_nvshmem_device.py`; still bootstraps with MPI             |
| R7  | No NVSHMEM artifacts in the Warp build or wheel                   | Must     | No Packman dependency, build flag, embedded payload, or license file      |
| R8  | NVSHMEM 3.9 or newer                                              | Must     | First release whose wheels ship the device LTOIR fatbin                   |

**Non-goals** for the initial implementation (the [Coverage Plan](#coverage-plan) schedules some of these as later
batches):

- Autodiff through NVSHMEM operations
- Block-cooperative or warp-cooperative variants (`nvshmemx_*_block`, `nvshmemx_*_warp`)
- Non-blocking variants (`nvshmem_*_nbi`)
- Cooperative kernel launch
- NVSHMEM's CuTe-based tile API (`nvshmemx_tile_api.hpp`)
- Windows or macOS support
- Warp-level wrappers for init, finalize, or symmetric allocation (users call nvshmem4py)
- NVSHMEM host libraries older than 3.9, including the version current PyTorch releases pin
- Including any NVSHMEM header during NVRTC compilation, whether the device implementation headers or the
  prototype-only `c_api` headers

## Design

### Approach

Warp's codegen emits `extern "C"` forward declarations for the NVSHMEM device entry points, compiles the kernel to
LTOIR with NVRTC, and links it with nvJitLink against `libnvshmem_device.ltoir.fatbin` from the NVSHMEM installation
that owns the host library loaded in the process. Host-side operations (`cumodule_init`, `cumodule_finalize`, status
and version queries) call into `libnvshmem_host.so` through `dlopen(RTLD_NOLOAD)`, reusing the library the
application loaded, typically through nvshmem4py.

Warp ships no NVSHMEM code, needs no NVSHMEM headers or archives to build, and has no NVSHMEM-specific build flag.
The feature is available in every Linux CUDA build that links nvJitLink.

### Decision: Link the Installed Device Library and Require NVSHMEM 3.9

Decided 2026-10-30. It replaces the 2026-10-25 decision to embed the NVSHMEM 3.8.0 device library in `warp.so`.

Warp statically embeds compilers and leaf runtimes: NVRTC, LLVM/Clang, libmathdx, and the CUDA runtime. Each takes
Warp-generated source in and produces a binary out, or talks only to the CUDA driver through a public compatibility
contract. Nothing else in the process has to agree with the version Warp picked, and each is mandatory for Warp to run
at all, so embedding is unconditional and paid once.

The NVSHMEM device library is different in kind. It is the device half of a split runtime. Its code reads
`nvshmemi_device_state_d` and the transport and team state that `nvshmemx_cumodule_init` populates from
`libnvshmem_host.so`, and those structures change between minor releases: linking 3.8.0 device code against the 3.4.5
library fails with `Size doesn't match for 'nvshmemi_device_state_d'`. The host library is chosen by the application's
environment, not by Warp. Conceptually the device library is a shared-library dependency that NVSHMEM ships as LTOIR
only because device code cannot be dynamically loaded.

| Component              | Role for Warp                   | Runtime counterpart           | Who picks the counterpart            |
| ---------------------- | ------------------------------- | ----------------------------- | ------------------------------------ |
| NVRTC, LLVM/Clang      | compiler for Warp's own codegen | CUDA driver, CPU              | nobody; the output is self-contained |
| libmathdx              | LTOIR generator for tile ops    | none; inlined into kernels    | nobody                               |
| CUDA runtime (static)  | leaf runtime                    | CUDA driver                   | Warp's toolkit choice                |
| `libnvshmem_host.so`   | runtime Warp calls into         | the application's NVSHMEM job | the application's environment        |
| NVSHMEM device library | device half of that runtime     | `libnvshmem_host.so`          | the application's environment        |

Embedding the device library was rejected for these reasons, in order of weight:

1. **Warp would own one half of a runtime while disowning the other.** Warp never loads, initializes, or allocates
   NVSHMEM and reuses whatever the application loaded (see [Host Library Loading](#host-library-loading)). Embedding
   the device half contradicts that posture and recreates, one layer down, the version mismatch that `RTLD_NOLOAD`
   was chosen to avoid.
2. **Every Warp release becomes an implicit NVSHMEM pin.** NVSHMEM supports a host library that is at least as new as
   the device library within a major version, so an embedded library is the minimum host version Warp accepts, and
   Warp has to implement and document that rule. The pin collides with other packages' pins: PyTorch 2.9 through 2.14
   pin `nvidia-nvshmem-cu13==3.4.5` exactly, and Warp's own `uv.lock` resolves to that version through torch, so the
   embedded 3.8.0 library made the feature unusable in Warp's default development environment. With the wheel's
   fatbin, host and device code come from one package and NVSHMEM's own `cumodule_init` check is the whole policy.
3. **NVSHMEM fixes would ship on Warp's release cadence.** Device-side fixes, support for new GPU architectures in
   the fatbin, and transport changes would reach users only through a Warp rebuild and release. Linked from the wheel,
   updating the NVSHMEM package is enough.
4. **Warp would redistribute NVSHMEM binaries.** That requires carrying and tracking a license file, and the license
   texts in the NVSHMEM redistributable manifest, archives, and wheels disagreed at the time of writing. An embedded
   binary is also invisible to dependency scanning, which only sees `uv.lock`. Embedding nothing ends both concerns.
5. **It gated the feature behind an opt-in build.** The prototype needs a roughly 485 MB Packman archive,
   `--use-nvshmem`, `.incbin` assembly with byte verification, producer-toolkit checks, and `WP_ENABLE_NVSHMEM`
   branches. It was off by default and no CI job built it, so release wheels would never have had the feature.
   Runtime linking needs only nvJitLink, which MathDx builds already carry.
6. **The ecosystem convention is the opposite.** NVSHMEM's own cuTile linkage provider and nvshmem4py locate the
   fatbin in the wheel, and NCCL ships `libnccl_device.ltoir` in its wheels for the same JIT consumers. Warp embedding
   would be the exception.

Wheel size is not the argument: the fatbin is about 4 MB. The coupling is.

Two constraints hold under either design and are checked at link time with a specific error:

- nvJitLink cannot consume LTOIR produced by a newer toolkit than the one it shipped with, so Warp's toolkit must be
  at least as new as the toolkit that produced the NVSHMEM fatbin. Embedding checked this at Warp's build; runtime
  linking surfaces it on the user's machine. Warp's default builds track current CUDA releases, which keeps the window
  small.
- Architecture coverage is NVSHMEM's. The 3.8.0 CUDA 13 fatbin carries LTOIR for SM 75, 80, 89, 90, 100, and 120;
  nvJitLink selects the entry that is compatible with the target.

### Where the Device Library Comes From

Since NVSHMEM 3.7, the redistributable archives include a multi-architecture `libnvshmem_device.ltoir.fatbin` and
per-architecture `libnvshmem_device_sm_*.ltoir` files. The PyPI wheels through 3.8.0 do not: they ship
`libnvshmem_device_sm_*.bc` (LLVM bitcode, which nvJitLink cannot consume) and `libnvshmem_device.a`. The NVSHMEM team
confirmed on 2026-10-30 that this is a packaging gap in 3.8.0 and that the 3.9 wheels always build and ship
`libnvshmem_device.ltoir.fatbin`; the `devel` branch of the public repository stages the fatbin and the
per-architecture LTOIR files in `python_libs_wheel/cmake/stagePlugins.cmake`. nvshmem4py 0.4.1 already exposes
`nvshmem.core.find_device_bitcode_library(language=DeviceLibLanguage.LTOIR)`, which resolves to that file.

Warp therefore requires NVSHMEM 3.9 or newer. Older hosts are rejected at module load with a message that names the
loaded version.

| Package                                                             | LLVM bitcode                | LTOIR                                                                                |
| ------------------------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------ |
| `nvidia-nvshmem-cu13` 3.7.0 through 3.8.0 (PyPI)                    | `libnvshmem_device_sm_*.bc` | none                                                                                 |
| NVSHMEM 3.7 and 3.8 redistributable archives, 3.8.0 Debian packages | `libnvshmem_device_sm_*.bc` | `libnvshmem_device.ltoir.fatbin`, `libnvshmem_device_sm_{75,80,89,90,100,120}.ltoir` |
| `nvidia-nvshmem-cu13` 3.9 (announced)                               | `libnvshmem_device_sm_*.bc` | `libnvshmem_device.ltoir.fatbin`, `libnvshmem_device_sm_*.ltoir`                     |

**CUDA 12.** In the CUDA 12 archives the file named `libnvshmem_device.ltoir.fatbin` is a single-architecture raw
LTOIR image (SM 90 in 3.8.0; the first configured architecture in 3.9), not a fatbin. CUDA 12 consumers must use the
per-architecture `libnvshmem_device_sm_<arch>.ltoir` file for the target architecture. CUDA 13 is the priority for
Warp; CUDA 12 handling is a follow-up.

### Decision: Hand-Written Device Declarations

Decided 2026-10-30. Warp declares the NVSHMEM device entry points it uses by hand, as `extern "C"` prototypes in
`warp/native/nvshmem_device.h` written against NVSHMEM's public C API, and includes no NVSHMEM header during NVRTC
compilation. The public C ABI is stable across releases, so the same declarations link against every device library
Warp supports.

The alternative was to include NVSHMEM's own prototype headers. The NVSHMEM `devel` branch (3.9) adds
`NVSHMEM_DEVICE_DECLARATIONS_ONLY`, under which `nvshmem.h` skips the device implementation and the prototypes live in
`c_api/nvshmem_api.h` and `c_api/nvshmemx_api.h`: about 180 device-callable declarations plus the
`NVSHMEMI_REPT_FOR_*` type-list macros. Under NVRTC those two headers compile in about 0.06 s and the result links
against the 3.8.0 fatbin, and they would have given Warp compiler-checked prototypes that match the installed device
library by construction, plus type lists from which to generate the typed builtins.

They were rejected because of what they include. The prototype headers include `cuda_runtime.h`, `cuda.h`,
`cuda_fp16.h`, `cuda_bf16.h`, and the libc headers unconditionally, and `nvshmem_api.h` defines host-only
`static inline` helpers (`nvshmemi_check_init_status`, `nvshmem_init`, `nvshmem_init_thread`, `nvshmem_finalize`,
`nvshmemx_init_attr`) that call `fprintf`, `strerror`, and `exit` outside any `__CUDACC_RTC__` guard. Warp's embedded
NVRTC provides none of those headers built in, and the CUDA toolkit headers are not available at runtime (see the
full-header alternative below). Making the prototype headers compile took about 80 lines of Warp-owned stub headers
shadowing those names: POD `__half` and `__nv_bfloat16`, opaque stream and module handles, fixed-width integer
typedefs, and five libc declarations. Those stubs are unacceptable for three reasons:

- **They would be permanent.** NVRTC has no built-in `cuda_fp16.h`, `cuda_bf16.h`, or `cuda_runtime.h`, so the
  stubs would ship for as long as Warp used the headers.
- **They are fragile at the user's NVSHMEM upgrade.** The libc stubs exist only to satisfy NVSHMEM's host-only
  helpers. A new helper, or a prototype that uses real content from a shadowed header, fails to compile on the user's
  machine when the NVSHMEM package changes, not in Warp's build.
- **They conflict with real headers.** A user who passes the CUDA toolkit include directory through the
  `extra_cuda_include_dirs` module option gets two definitions of `__half`.

Upstream `__CUDACC_RTC__` guards around the helpers and includes would not finish the job: the RMA prototypes take
`half` and `__nv_bfloat16` by value, so those types still need definitions under NVRTC, and NVSHMEM would have to
provide RTC-only minimal types for them. That is a larger ask than guarding a few helpers, for a convenience.

The accepted cost is that `extern "C"` declarations are unchecked at link time: a wrong prototype is silent memory
corruption rather than a compile error, and the block grows with builtin coverage (typed atomics, waits, and
reductions multiply it). Two mitigations:

- Declare with the header's own types rather than ABI-equivalent ones. The prototype's `ptrdiff_t` strides are
  currently declared as `long long`, which is the same size on LP64 but a different type.
- Add a CI consistency check in a job whose image has `nvcc` and the NVSHMEM wheel installed: a script reads the
  `extern "C"` block of `nvshmem_device.h` and emits one
  `static_assert(std::is_same_v<decltype(&::nvshmem_putmem), void (*)(void*, const void*, size_t, int)>)` per
  declared function, compiled against the wheel's real headers. A drifted prototype then fails CI before a release,
  and Warp depends on CUDA headers only in that CI job, never at runtime.

### Alternatives Considered and Rejected

**Embedding the device library in `warp.so` (the prototype).**
`build_lib.py` discovered the fatbin under `NVSHMEM_HOME` or a Packman archive, staged it under `_build`, embedded it
through an assembly `.incbin` directive with start and end symbols, and verified the exported bytes against the
source file after linking. At runtime, native code passed the embedded bytes to nvJitLink as `NVJITLINK_INPUT_FATBIN`
or `NVJITLINK_INPUT_LTOIR` depending on the compiled input kind. It worked, and the byte verification caught an
earlier `ld -r -b binary` attempt that produced a corrupted range. It was rejected for the reasons in the decision
above.

**Compiling against the full NVSHMEM device headers under NVRTC.**
The headers have `__CUDACC_RTC__` support and do compile under NVRTC 13.4 when the NVSHMEM include directory, the
CUDA toolkit include directory, and `include/cccl` are on the include path, and nvJitLink links the result against
either the fatbin or `libnvshmem_device.a`. The header compile takes about 3.2 s per module instead of 0.02 s with
forward declarations, and the headers need libcu++ (`cuda::atomic_ref`, `cuda::std::tuple`, `cuda::std::array`,
type traits) plus `cuda_runtime.h`, `cuda_fp16.h`, and `cuda_bf16.h`. Warp does not ship CUDA toolkit headers and
will not depend on the `nvidia-cuda-*` wheels for them, and stubbing libcu++ is not an option. Rejected.

**Including NVSHMEM's prototype headers under NVRTC.**
With `NVSHMEM_DEVICE_DECLARATIONS_ONLY` (3.9), the `c_api` prototype headers avoid libcu++ and compile under NVRTC in
about 0.06 s, but only with about 80 lines of Warp-owned stubs standing in for `cuda_runtime.h`, `cuda.h`,
`cuda_fp16.h`, `cuda_bf16.h`, and the libc headers. The stubs would be permanent and fragile. Rejected in favor of
hand-written declarations; see [Decision: Hand-Written Device Declarations](#decision-hand-written-device-declarations).

**Linking the wheel's `libnvshmem_device.a`.**
The static archive in the wheels holds relocatable cubins for SM 75, 80, 89, 90, 100, and 120, but the public
`nvshmem_*` functions are inline in the headers, so the archive contains only `nvshmemi_*` internals. A kernel built
from forward declarations fails to link against it with undefined `nvshmem_my_pe`, `nvshmem_float_p`, and
`nvshmem_quiet`. Using the archive requires the full headers, which the previous alternative rules out. Rejected.

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
The shim approach was ultimately unnecessary once the packaged LTOIR became available.

**Direct linking to `libnvshmem_host.so`**.
The NVSHMEM distribution ships only a shared host library. Linking Warp against one packaged copy
would not guarantee that it matches the library loaded by nvshmem4py. The implementation therefore
uses `dlopen(RTLD_NOLOAD)` to find the already-loaded `libnvshmem_host.so` and resolves the
required symbols with `dlsym`. This makes initialization order explicit and avoids loading a second,
potentially incompatible host runtime.

**`symmetric=True` on `wp.array`**.
An earlier prototype allocated symmetric memory through `wp.zeros(..., symmetric=True)` and freed it
in the array's deleter. `nvshmem_malloc` and `nvshmem_free` are collectives: every PE must make the
same calls in the same order with the same sizes, because that sequence determines each object's
offset in the symmetric heap. Tying the free to a Warp array's lifetime made it depend on every
rank holding references identically. Rank-specific references, views that keep a parent alive,
exception tracebacks, reference cycles, and `requires_grad` toggling can each break that. On two
GPUs, keeping one extra reference on PE 0 placed the next allocation at different heap offsets on
the two PEs, and the job hung. Dropping an array after `nvshmem.core.finalize()` made NVSHMEM abort
the process, and allocating with a device other than the PE's device produced an array whose
`device` did not match its memory. nvshmem4py reaches the same conclusion: it raises
`NvshmemError` when a buffer is released without an explicit `free()`.

### Key Implementation Details

#### Build System

Warp's build needs nothing from NVSHMEM. The only change is that nvJitLink is linked whenever CUDA is enabled on
Linux, not only when libmathdx is. The nvJitLink helper in `warp.cu` and the `-dlto` and
`--relocatable-device-code=true` NVRTC options are used whenever a module has link inputs, whether they come from
libmathdx or from NVSHMEM.

#### Device-Side Builtins

The builtins are type-generic and address elements with Warp indices, like the atomics:

| Builtin                                                                             | Operation                              |
| ----------------------------------------------------------------------------------- | -------------------------------------- |
| `wp.nvshmem_my_pe()`, `wp.nvshmem_n_pes()`                                          | PE index and count                     |
| `wp.nvshmem_p(dest, i[, j, k, l], value, pe)`                                       | Write one element on a remote PE       |
| `wp.nvshmem_g(source, i[, j, k, l], pe)`                                            | Read one element from a remote PE      |
| `wp.nvshmem_put(dest, source, pe)`                                                  | Copy a local array into a remote array |
| `wp.nvshmem_get(dest, source, pe)`                                                  | Copy a remote array into a local array |
| `wp.nvshmem_put_signal(dest, source, signals, signal_index, signal, signal_op, pe)` | Put, then update a remote signal       |
| `wp.nvshmem_signal_op(signals, i, signal, signal_op, pe)`                           | Update a remote signal                 |
| `wp.nvshmem_signal_wait_until(signals, i, cmp, value)`                              | Wait on a local signal                 |
| `wp.nvshmem_quiet()`, `wp.nvshmem_fence()`, `wp.nvshmem_barrier_all()`              | Completion, ordering, barrier          |

Arrays may be any Warp dtype, including vectors, matrices, and 16-bit floats, with up to four dimensions.
Element operations take one index per dimension. Bulk operations take arrays of the same dtype and shape;
slice them to transfer part of an array. The signal arrays are 1D `wp.uint64` arrays, and
`wp.NVSHMEM_SIGNAL_*` and `wp.NVSHMEM_CMP_*` match the NVSHMEM constants.

The Python registrations in `builtins.py` only declare signatures and check dtypes. The implementations
are `wp::` templates in `warp/native/nvshmem_device.h`, which `builtin.h` includes for CUDA compilation:

- Element addresses come from Warp's `index()`, so strides, negative indices, and debug bounds checks
  behave as for ordinary array access.
- Elements move through the unsigned-integer NVSHMEM functions sized to the element's alignment
  (`nvshmem_uint8_p` through `nvshmem_uint64_p` and their `g`, `iput`, and `iget` forms), which transfers
  any dtype bit for bit. Elements wider than 8 bytes move as several parts, so they are not written
  atomically.
- Bulk transfers use `nvshmem_putmem`/`nvshmem_getmem` when both arrays are packed, the strided `iput`/`iget`
  forms for other 1D arrays, and one transfer per element otherwise.
- `nvshmem_put_signal` uses `nvshmem_putmem_signal` for packed arrays. NVSHMEM has no strided form, so
  other arrays are put, completed with `nvshmem_quiet`, and then signaled.

All builtins use `export=False` because they are device-only. Builtins with side effects use
`skip_replay=True` so the backward pass does not repeat remote writes, waits, or barriers.

The `extern "C"` declarations in `nvshmem_device.h` are written by hand against NVSHMEM's public C API and use the
header's own parameter types. No NVSHMEM header is included during NVRTC compilation; see
[Decision: Hand-Written Device Declarations](#decision-hand-written-device-declarations) for the reasoning and the
CI check that guards the declarations against drift.

#### Codegen and Linking

When any `wp.nvshmem_*` builtin is referenced in a kernel:

1. The `ModuleBuilder.uses_nvshmem` flag is set when a builtin registered with `requires_nvshmem=True` is
   called. User functions never set it, whatever their name.
2. Compilation needs an NVSHMEM installation on the machine (for the device library) but no initialized NVSHMEM
   runtime. `wp.compile_aot_module` works on a build machine with the NVSHMEM wheel installed.
3. The generated source needs nothing NVSHMEM-specific: `nvshmem_device.h` declares the NVSHMEM entry
   points for every CUDA module, and only modules that call them are linked with the device library.
   Modules that use NVSHMEM builtins fail to compile for the CPU and with the `llvm_cuda` option.
4. NVRTC compiles with `-dlto` and `--relocatable-device-code=true`, and the device library is passed to
   nvJitLink as `NVJITLINK_INPUT_FATBIN` (or `NVJITLINK_INPUT_LTOIR` for a raw per-architecture file) through the
   same link-input arrays that carry libmathdx LTOIR.

**Output format**: Both cubin and PTX output work. With PTX output, nvJitLink performs LTO and emits
PTX (`-ptx`), which the driver then JIT-compiles. A two-PE remote-write test passed with
`wp.config.cuda_output = "ptx"` on NVSHMEM 3.8.0 and CUDA 13.4 with the prototype's embedded library.

#### Device Library Discovery

The device library must come from the same NVSHMEM installation as the host library the application loads, so that
`nvshmemx_cumodule_init` finds device code built for its state layout. Discovery order:

1. An explicit override, the `WARP_NVSHMEM_DEVICE_LIBRARY` environment variable, naming a
   `libnvshmem_device.ltoir.fatbin` or a raw `.ltoir` file. This serves custom NVSHMEM builds and CI before 3.9 wheels
   exist (the 3.8.0 archive's fatbin paired with a 3.8.0 host library). An override also skips the 3.9 host version
   floor, which exists only because older wheels lack the device library; NVSHMEM's own `nvshmemx_cumodule_init`
   check that host and device versions match still applies.
2. The directory of the loaded `libnvshmem_host.so`, found with `dladdr` on a symbol resolved through the
   `RTLD_NOLOAD` handle. The wheel, the Debian packages, and the archives all place `libnvshmem_device.ltoir.fatbin`
   next to the host library.
3. If no host library is loaded yet, the installed `nvidia-nvshmem-cu<major>` distribution for Warp's CUDA major
   version, located through `importlib.metadata`, so that compilation can precede `nvshmem.init()` and
   ahead-of-time compilation works on machines that never initialize NVSHMEM.
4. Otherwise compilation fails with an error that names the paths tried and states the NVSHMEM 3.9 requirement.

The selected file's identity (path, kind, source, distribution version, and SHA-256 digest) is recorded in the module
metadata as the `nvshmem` record; its presence is what marks a cached module as one that uses NVSHMEM. On a cache
hit, a different installed device library is
treated as a cache miss, so upgrading the NVSHMEM package recompiles affected modules instead of running device code
from the previous version. NVSHMEM accepts an older device library with a newer host library, so without this check
the stale cubin would load and run silently.

#### Module Registration

After `build.load_cuda()` returns a CUmodule handle,
if the module uses NVSHMEM, Warp calls `wp_nvshmem_cumodule_init(handle)` (native C function
that forward-declares and calls `nvshmemx_cumodule_init` from `libnvshmem_host.so`).
This populates the `__constant__` symbols in the cubin that the NVSHMEM runtime needs
(`nvshmemi_device_state_d`, `nvshmemi_ibgda_device_state_d`, `nvshmemi_device_lib_version_d`).

Loading fails with a `RuntimeError` that names the cause and how to fix it:

- Before the CUDA module is loaded: no NVSHMEM host library is loaded in the process, or the loaded host library is
  older than 3.9 and the module was not linked with an explicit override.
- After the CUDA module is loaded: `nvshmemx_cumodule_init` fails, most often because NVSHMEM is not
  initialized (native code checks `nvshmemx_init_status() >= 2`), or because the device library does not match the
  host library, which NVSHMEM checks itself through `nvshmemi_device_lib_version_d`. Warp unloads the CUDA module
  before raising.

Nothing is cached on failure, so loading again after NVSHMEM initialization succeeds. An earlier
prototype only warned and kept the module; kernels then ran against uninitialized NVSHMEM device state
and silently returned wrong values (for example, `nvshmem_n_pes()` returned 0), and the module stayed
unregistered for the rest of the process.

Warp has no version rule of its own beyond the 3.9 minimum. The prototype compared the loaded host library against
the version compiled into `warp.so` because the embedded device library fixed one side of the pair; with both halves
coming from one installation that comparison has nothing to check.

Before a registered CUDA module is unloaded, `ModuleExec` calls `nvshmemx_cumodule_finalize` to
unregister it. Both calls are local to the PE, not collectives, so the destructor can make them. If
NVSHMEM is already finalized, the native wrapper skips the call.

`ModuleBuilder.uses_nvshmem` must also survive a cache hit in a fresh process. Codegen does not run
on that path, so cache identity alone cannot reconstruct the in-memory flag. The `nvshmem` record in the module
metadata (see [Device Library Discovery](#device-library-discovery)) is that flag: a cached module whose metadata
carries it is registered with `nvshmemx_cumodule_init` after loading. Metadata written by the prototype carried only
a `__uses_nvshmem__` flag; a module with that key is rebuilt rather than loaded.

#### Symmetric Memory

Warp does not allocate symmetric memory. Users allocate it with nvshmem4py and wrap the pointer as a
Warp array that does not own the memory:

```python
buf = nvshmem.core.buffer(n * 4)  # collective: same size on every PE
a = wp.array(ptr=int(buf.handle), shape=n, dtype=wp.float32, device=device)
a.zero_()  # nvshmem4py does not zero memory; synchronize and barrier before remote writes
...
del a
nvshmem.core.free(buf)  # collective: every PE, same order, before nvshmem.core.finalize()
```

The free is an explicit collective call in program order, matching the NVSHMEM and nvshmem4py
programming model. The wrapping array must not be used after `free()`.

#### Host Library Loading

Warp never loads `libnvshmem_host.so` itself. The application loads and initializes it, typically
through nvshmem4py, and Warp uses `dlopen(RTLD_NOLOAD)` to reuse that exact loaded library. It then
resolves `nvshmemx_cumodule_init`, `nvshmemx_cumodule_finalize`, `nvshmemx_init_status`, and
`nvshmemx_vendor_get_version_info` with `dlsym`, and uses `dladdr` on one of them to find the library's directory
for device library discovery. The functions are forward-declared in `nvshmem.cpp` because the NVSHMEM headers cannot
be included by g++ (they pull in CCCL/CUDA C++ headers); with no embedded library there is no other reason to have
the headers at build time.

The host library's soname is `libnvshmem_host.so.<major>`; the `devel` branch keeps it at the project major version,
so `libnvshmem_host.so.3` remains correct for 3.9.

**Initialization order matters**: `wp.set_device()` (which creates the CUDA context)
must be called BEFORE `nvshmem.init()`. NVSHMEM's `cumodule_init` fails with
"nvshmem get cucontext failed" if there's no active CUDA context.

#### nvshmem4py API

The Python package is `nvshmem4py-cu13` (or `nvshmem4py-cu12`). Import as `import nvshmem`.
Warp itself does not import nvshmem4py; applications, tests, and examples use it to initialize NVSHMEM
and allocate symmetric memory. Any library that loads and initializes the NVSHMEM host runtime works.

Key APIs used:

- `nvshmem.core.init(device=CudaDevice(rank), mpi_comm=comm, initializer_method="mpi")`: MPI bootstrap
- `nvshmem.core.my_pe()`, `nvshmem.core.n_pes()`: PE queries (host side)
- `nvshmem.core.init_status()`: returns `InitStatus` enum (0=not init, 1=bootstrapped, 2=initialized)
- `nvshmem.core.buffer(nbytes)`: collective symmetric allocation (returns a `cuda.core` `Buffer`)
- `nvshmem.core.free(buffer)`: collective symmetric deallocation
- `nvshmem.bindings.nvshmem.cumodule_init(handle)`: register CUmodule
- `nvshmem.bindings.nvshmem.barrier_all()`: host-side barrier (low-level, no stream arg)
- `nvshmem.core.barrier_all()`: higher-level barrier (requires a stream, may fail without one)
- `nvshmem.core.find_device_bitcode_library(language=DeviceLibLanguage.LTOIR)`: path of the wheel's fatbin
  (nvshmem4py 0.4.1 and newer); Warp locates the file itself to avoid importing nvshmem4py, but the two agree

Note: `nvshmem.core.barrier_all()` requires a `cuda.core` stream object.
For simple use, `nvshmem.bindings.nvshmem.barrier_all()` works without one.

### Transition from the Prototype

The first version of the branch implemented the embedded design. Moving to the decided design removed:

- `deps/libnvshmem-deps.packman.xml`, `licenses/libnvshmem-LICENSE.txt`, and `tools/build_nvshmem_device_ltoir.sh`
- In `build_lib.py`: `--use-nvshmem`, `--nvshmem-ltoir`, `--nvshmem-fatbin`, NVSHMEM discovery and validation, the
  `.incbin` assembly generation, legacy sidecar cleanup, the producer-toolkit check, and the post-link byte
  verification
- In `build_dll.py`: NVSHMEM include paths, `WP_ENABLE_NVSHMEM`, and `WP_NVSHMEM_DEVICE_LIBRARY_KIND`; nvJitLink is
  linked for every Linux CUDA build
- In `nvshmem.cpp`: the `nvshmem_version.h` include, `wp_nvshmem_get_build_version`, the embedded-library accessor,
  and the host-version comparison
- In `warp.cu`: the `link_nvshmem` parameter and the embedded-library branch of `wp_cuda_compile_program`; the device
  library arrives through the existing link-input arrays
- In `context.py`: `_validate_nvshmem_version` and the tests that exercise it

It added `warp/_src/nvshmem.py` (device library discovery and the override), the metadata record of the linked
library, the 3.9 minimum check at load, and the `requires_nvshmem` attribute on `add_builtin` in place of the
name-prefix check. The builtins, module registration, unload handling, multi-PE tests, and examples carried over
unchanged.

## Coverage Plan

Decided 2026-10-30 from a survey of the NVSHMEM 3.8.0 device API and of the device functions that NVSHMEM's own
examples and performance tests call. Two prerequisites land before any new builtin, because each addition would
otherwise extend mechanisms the transition replaces: the explicit "requires NVSHMEM" attribute on `add_builtin` in
place of name-prefix detection, and routing the device library through the link-input arrays.

### Initial Set

The twelve builtins cover the point-to-point core. The device functions NVSHMEM's examples and perftests call most
often are, in order, `nvshmem_barrier_all`, `nvshmem_n_pes`, `nvshmem_my_pe`, `nvshmem_quiet`, typed `wait_until`,
typed `p`, `nvshmemx_signal_op`, typed `put_nbi`, typed `put`, `nvshmem_fence`, block-scope `put_nbi`, and typed `g`.
Of those, only typed `wait_until` and the non-blocking forms are missing. Two fixes apply to the initial set itself:

- **Thread-scope calls replicate in tiled kernels.** With `wp.launch_tiled`, every thread in the block executes the
  kernel body with the same `wp.tid()`, so a thread-scope builtin runs once per thread: `wp.nvshmem_p` issues
  `block_dim` identical puts, which is wasteful but benign, and an atomic would be wrong by a factor of `block_dim`.
  The rule is that thread-scope `nvshmem_*` builtins are for `wp.launch` kernels and `tile_nvshmem_*` builtins for
  tiled launches. Codegen can warn when a thread-scope NVSHMEM builtin appears in a module compiled with a
  `block_dim` above one.
- The `iput` and `iget` stride parameters are declared with the header's `ptrdiff_t`.

### Batches

| Batch | Builtins                                                                                                                                                                                                  | NVSHMEM functions                                                                                                                                                                                                           | Rationale                                                                                                                                                                                  |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1     | `nvshmem_ptr(arr, pe)` returning an array view of remote memory; `nvshmem_put_nbi`, `nvshmem_get_nbi`; `nvshmem_sync_all`; `nvshmem_wait_until`, `nvshmem_test` on integer arrays; `nvshmem_signal_fetch` | `nvshmem_ptr`, `nvshmem_putmem_nbi`, `nvshmem_getmem_nbi`, `nvshmem_sync_all`, `nvshmem_<T>_wait_until`, `nvshmem_<T>_test`, `nvshmem_signal_fetch`                                                                         | `nvshmem_ptr` is the most Warp-idiomatic addition: ordinary indexing and `tile_store` work on the view for P2P-reachable PEs. The rest complete the point-to-point model the examples use. |
| 1     | Teams: `wp.NVSHMEM_TEAM_WORLD`, `wp.NVSHMEMX_TEAM_NODE`; `nvshmem_team_my_pe`, `nvshmem_team_n_pes`, `nvshmem_team_translate_pe`; `nvshmem_barrier(team)`, `nvshmem_sync(team)`                           | `nvshmem_team_*`, `nvshmem_barrier`, `nvshmem_team_sync`                                                                                                                                                                    | `nvshmem_team_t` is an `int32`, so this is cheap, and node-local teams are what multi-node halo exchange keys on.                                                                          |
| 2     | Atomics: `nvshmem_atomic_add`, `fetch_add`, `inc`, `fetch_inc`, `set`, `fetch`, `swap`, `compare_swap`; `and`, `or`, `xor` and their fetch forms                                                          | `nvshmem_<T>_atomic_*`; `nvshmemx_{float,double}_atomic_{add,fetch_add}`                                                                                                                                                    | Counters, locks, and work queues. The first typed family, so it fixes the dtype table below.                                                                                               |
| 3     | Block-cooperative: `tile_nvshmem_put`, `tile_nvshmem_get` and their `_nbi` forms, `tile_nvshmem_put_signal`, `tile_nvshmem_barrier_all`, `tile_nvshmem_sync_all`, team barrier and sync                   | `nvshmemx_putmem_block`, `nvshmemx_getmem_block`, `nvshmemx_<T>_iput_block`, `nvshmemx_<T>_put_signal_block`, `nvshmemx_barrier_all_block`, `nvshmemx_sync_all_block`, `nvshmemx_barrier_block`, `nvshmemx_team_sync_block` | Bulk transfers from one thread are the current performance trap; these are the fix. See the next section.                                                                                  |
| 4     | Collectives: `tile_nvshmem_reduce` (sum, prod, min, max; and, or, xor), `tile_nvshmem_broadcast`, `tile_nvshmem_fcollect`, `tile_nvshmem_alltoall`, `tile_nvshmem_reducescatter`, all team-based          | `nvshmemx_<T>_<op>_reduce_block`, `nvshmemx_broadcastmem_block`, `nvshmemx_fcollectmem_block`, `nvshmemx_alltoallmem_block`, `nvshmemx_<T>_<op>_reducescatter_block`                                                        | Device-side collectives need every participating block resident on every PE, which Warp cannot guarantee without cooperative launch. Gated on that.                                        |
| later | `wait_until_all`, `_any`, `_some` and their vector forms; counted signals (3.8); `_warp` variants; QP and shared-memory lending APIs                                                                      |                                                                                                                                                                                                                             | Warp has no warp-level programming model, and the rest are niche.                                                                                                                          |

### Typed Families

The RMA builtins stay dtype-generic through the alignment-sized unsigned transfers. Atomics, waits, and reductions are
typed in NVSHMEM and need one dtype table shared by the Python constraints and the native dispatch, so unsupported
dtypes are rejected at codegen rather than as an unresolved symbol from nvJitLink. NVSHMEM's `int32`, `uint32`,
`int64`, and `uint64` names are used instead of `int`, `long`, and `long long`, which removes the LP64 ambiguity.

| Warp dtype                                 | NVSHMEM type name    | Atomics                                                                        | Waits and tests | Reductions                                        |
| ------------------------------------------ | -------------------- | ------------------------------------------------------------------------------ | --------------- | ------------------------------------------------- |
| `int32`, `uint32`, `int64`, `uint64`       | `int32` ... `uint64` | standard and bitwise                                                           | yes             | sum, prod, min, max, and, or, xor                 |
| `int16`, `uint16`                          | `short`, `ushort`    | standard only                                                                  | yes             | sum, prod, min, max (`short`); bitwise (`ushort`) |
| `float32`, `float64`                       | `float`, `double`    | `fetch`, `set`, `swap`; `add` and `fetch_add` through the `nvshmemx` extension | no              | sum, prod, min, max                               |
| `float16`, `bfloat16`                      | `half`, `bfloat16`   | no (the half extension is a build option)                                      | no              | sum, prod, min, max                               |
| `int8`, `uint8`, `bool`, vectors, matrices | none                 | no                                                                             | no              | `int8`/`uint8` bitwise only                       |

NVSHMEM documents the `nvshmemx` floating-point `add` and `fetch_add` as supported over GPU load/store paths such as
NVLink peer access and over the IBRC remote transport only.

### Block-Cooperative Calls Through `tile_*`

The mapping is direct because the contracts coincide. NVSHMEM requires that a `_block` function be called by every
thread in the block with identical arguments, and that is the contract a Warp tile op already assumes: in a tiled
launch all threads execute the body with the same `tid`, and tile builtins are collective. `wp.tile_nvshmem_put(dest,
source, pe)` is therefore a one-line native wrapper around `nvshmemx_putmem_block`, and `tile_nvshmem_barrier_all`
around `nvshmemx_barrier_all_block`. The `tile_nvshmem_*` names follow `tile_atomic_add(array, tile, offset)`, Warp's
convention for a block-cooperative operation on an array.

The operand types are the design question:

- **Array to array**, both in global memory, is the first form. It matches the current `nvshmem_put`: packed arrays go
  through `nvshmemx_putmem_block`, strided 1D arrays through the typed `iput_block`, and other layouts per element.
- **Tile to remote array** has a transport constraint. A shared-memory tile is a contiguous buffer only when its layout
  is the default packed one; `tile_view` and `tile_transpose` produce strided layouts, so the builtin must check the
  layout at compile time and stage otherwise. More fundamentally, shared memory is a valid put source only on paths
  where the GPU itself performs the copy. NVSHMEM 3.8 does this over P2P, including a TMA path where the kernel lends
  shared memory through `nvshmemx_ask_smem` and `nvshmemx_give_smem`. Remote transports issue RDMA from registered
  global memory, so a shared-memory source does not work portably. The plan is two steps: ship `nvshmem_ptr` first, so
  `tile_store` into a remote view gives tile-to-remote transfers for P2P-reachable PEs with no new builtin, and add a
  tile overload of `tile_nvshmem_put` later that stages through a caller-provided local array when the destination PE
  is not P2P-reachable.
- **NVSHMEM's own tile API** (`nvshmemx_tile_api.hpp`) is CuTe-based header-only templates, unreachable from
  header-free linking. The `_block` C API is the path.
- **Device-side collectives need co-residency.** NVSHMEM's guidance is to launch kernels that call device collectives
  with `nvshmemx_collective_launch`, a cooperative launch that fails instead of deadlocking when the grid cannot be
  resident. Warp launches with `cuLaunchKernel`. `tile_nvshmem_barrier_all` is safe when the grid fits on the device
  or a single block per PE participates; batch 4 waits for a cooperative launch option in Warp, which is a small,
  separately useful feature.
- **Warp-scope variants** are not planned. Warp exposes no warp-level programming model.

## Pitfalls and Lessons Learned

### nvJitLink Cannot Consume LLVM Bitcode

The original design assumed `libnvshmem_device.bc` could be passed to nvJitLink.
It cannot. nvJitLink rejects LLVM bitcode (`.bc`) in every input type,
including `NVJITLINK_INPUT_ANY`. LLVM bitcode and NVIDIA LTOIR are different formats
despite both being based on LLVM IR. The usable input is the LTOIR that NVSHMEM builds with `nvcc --ltoir`.

### The Packman Distribution Does Not Include LTOIR (Pre-3.7)

The NVSHMEM 3.6.x distribution ships `.bc` (LLVM bitcode) and `.a` (host objects, ELF),
but NOT `.ltoir`. The prototype built the LTOIR from source via cmake with `NVSHMEM_BUILD_LTOIR_LIBRARY=ON`
and `NVSHMEM_IBGDA_SUPPORT=OFF` (to avoid InfiniBand header dependencies).
NVSHMEM 3.7.1 resolves this packaging gap in the archives by including raw LTOIR files for SM 75, 80, 89, 90,
100, and 120, plus `libnvshmem_device.ltoir.fatbin`; the wheels follow in 3.9.

### The Packaged Fatbin Needs the Correct nvJitLink Input Type

The `libnvshmem_device.ltoir.fatbin` is not interchangeable with a raw `.ltoir` file at the
API boundary. It must be submitted as `NVJITLINK_INPUT_FATBIN`, while the architecture-specific
files use `NVJITLINK_INPUT_LTOIR`. Renaming a fatbin to look like raw LTOIR is invalid, and in the CUDA 12 archives
the file with the fatbin name is a raw single-architecture image (see
[Where the Device Library Comes From](#where-the-device-library-comes-from)).

### Public Symbols Live Only in the LTOIR Library

The public `nvshmem_*` device functions are `always_inline` wrappers in the headers that call `nvshmemi_*`
internals. The static archive `libnvshmem_device.a` therefore contains only the internals, and a kernel built from
forward declarations cannot link against it. The LTOIR library is built in a mode that exports the public functions as
`extern "C"` symbols, which is what makes header-free linking possible.

### The Device Headers Need libcu++ Under NVRTC

NVSHMEM's device headers compile under NVRTC, but only with the CUDA toolkit's `include` and `include/cccl`
directories on the include path: they use `cuda::atomic_ref`, `cuda::std::tuple`, `cuda::std::array`, and type traits,
and include `cuda_runtime.h`, `cuda_fp16.h`, and `cuda_bf16.h`. Trivial stubs for the standard integer types are not
enough. The prototype-only `c_api` headers avoid libcu++ but still include `cuda_runtime.h`, `cuda.h`, `cuda_fp16.h`,
`cuda_bf16.h`, and libc. Warp's NVRTC compilation sees only `warp/native`, by design, and Warp's embedded NVRTC has no
built-in copy of any of those headers.

### Host and Device Library Versions

NVSHMEM accepts a host library that is at least as new as the device library within a major version, and rejects the
rest in `nvshmemx_cumodule_init`. The internal state layout changes between minor versions (3.8.0 device code against
a 3.4.5 library fails on the size of `nvshmemi_device_state_d`). Taking both halves from the same installation makes
the rule moot for ordinary use; it still applies when the override points at a device library from a different
installation. Observed on 2026-10-30 with the 3.4.5 host library and the 3.8.0 fatbin: the module compiles and
links, `nvshmemx_cumodule_init` prints "NVSHMEM version in CUmodule or CUlibrary does not match with NVSHMEM host
library version" and returns 7, and Warp unloads the module and raises
`nvshmemx_cumodule_init failed (7)`. The failure is loud and the process continues.

### Multiple `libnvshmem_host.so` Versions on the System

The system may have multiple copies of `libnvshmem_host.so`:
an older system-wide install (e.g., `/lib/x86_64-linux-gnu/libnvshmem_host.so.3`),
a CUDA-version-specific install (`/usr/lib/x86_64-linux-gnu/nvshmem/13/`),
and the one bundled with nvshmem4py. Using `dlopen("libnvshmem_host.so")` from C code
finds whichever the system linker resolves first, which may differ from nvshmem4py's copy.
Solution: use `dlopen(RTLD_NOLOAD)` to reuse the library nvshmem4py already loaded, and take the device library from
that library's directory.

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

### Raw Pointer Arguments Ignored Strides

The first builtins were typed (`nvshmem_float_p`, `nvshmem_int_put`, and so on) and built pointer
expressions such as `&var_dest.data[var_offset]` in Python dispatch functions. That ignored array strides:
on two GPUs, writing element 1 of `buf[::2]` on a remote PE updated `buf[1]` instead of `buf[2]`. The
generic builtins compute addresses with Warp's array indexing in native code, and the multi-PE test
covers strided views.

### The nvJitLink Dependency Must Follow Both Features

Warp originally gated both its nvJitLink helper and linker dependency on MathDx. NVSHMEM-enabled
kernels need nvJitLink even when MathDx is disabled. Both the compile-time guard and native linker
inputs now use `WP_ENABLE_MATHDX || WP_ENABLE_NVSHMEM`; a CUDA 13.3 build with
`--no-use-libmathdx` verified that an NVSHMEM kernel still compiles. Under the decided design nvJitLink is linked for
every Linux CUDA build.

### Multi-Process Runs Can Share the Kernel Cache

Warp builds each module in a process- and thread-specific directory and renames it into place, so
ranks that compile the same module concurrently can share one kernel cache. Earlier versions of the
examples set `WARP_CACHE_ROOT` per rank, but Warp reads `WARP_CACHE_PATH`, so that setting had no
effect. The UID test still assigns a per-rank `WARP_CACHE_PATH` to keep test artifacts isolated.

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

Transport selection and MPG support are independent. `NVSHMEM_REMOTE_TRANSPORT=none` disables
network transports such as `ibrc`; it does not select an MPG support level.

### MIG, MPG, and MPS Support Levels

- A processing element (PE) is one process participating in an NVSHMEM job.
- Multi-Instance GPU (MIG) partitions one physical GPU into isolated CUDA-visible devices. On the
  tested system, each PE was assigned to a different 1g.24gb MIG slice, but NVSHMEM still grouped the
  two PEs under their shared physical GPU and classified the job as multiple processes per GPU (MPG).
- CUDA Multi-Process Service (MPS) allows work from multiple CUDA process contexts to execute
  concurrently on a GPU.
- In **limited MPG** mode, MPS is absent or the configured active-thread percentages sum to more
  than 100%.
  NVSHMEM documents support for point-to-point RMA and a limited set of host- or stream-initiated
  barriers and synchronization APIs. Device-side synchronization and collectives are not part of
  this support level.
- In **full MPG** mode, MPS is running and the active-thread percentages for PEs sharing a GPU total
  at most 100%. NVSHMEM then enables its complete API, including device-side synchronization and
  collectives. On a MIG system, enabling MPS requires one server and distinct pipe directory per MIG
  GPU instance rather than a single environment-variable change.

See the NVSHMEM [MPG installation guidance][nvshmem-mpg] and the CUDA
[MIG with MPS workflow][mig-mps] for the configuration requirements.

The RTX PRO 6000 Blackwell result is **limited-MPG empirical validation**. MPS was not running, and
NVSHMEM explicitly warned that only limited MPG support was available. It nevertheless reported both
PEs in its P2P connected list, opened CUDA IPC handles between the MIG slices, and completed
device-side PE queries, an `nvshmem_float_p` from PE 0 to PE 1, and a device-side barrier. The RMA is
within the documented limited-MPG API set; the successful device-side barrier is an observed behavior
on this configuration, not a claim that NVSHMEM supports device-side synchronization in limited MPG
mode. A full-MPG validation would require a separately configured MPS-enabled CI job and is not
covered by this test.

This configuration validates same-host CUDA IPC/P2P without a network transport. Multi-node testing
must select a network transport appropriate for the target system.

[nvshmem-mpg]: https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/nvshmem-install-proc.html#using-nvshmem-with-multiple-processes-per-gpu
[mig-mps]: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/getting-started-with-mig.html#mig-with-cuda-mps

### Link Time Comes From Inlined NVSHMEM Device Code

The first compile of a module that uses NVSHMEM builtins is dominated by nvJitLink. Measured with
caching disabled on NVSHMEM 3.8.0, CUDA 13.4, and an SM 86 target, for kernels without backward passes:

| Kernel uses                                                    | Cold link | Cubin   |
| -------------------------------------------------------------- | --------- | ------- |
| `nvshmem_my_pe` only                                           | 2.4 s     | 32 KB   |
| `nvshmem_p` or `nvshmem_g`                                     | 3.4 s     | ~180 KB |
| `nvshmem_put` or `nvshmem_get`                                 | 3.7 s     | ~204 KB |
| `nvshmem_quiet` or `nvshmem_fence`                             | 3.0 s     | ~116 KB |
| `nvshmem_signal_op`                                            | 3.1 s     | 141 KB  |
| `nvshmem_barrier_all`                                          | 4.5 s     | 292 KB  |
| `nvshmem_put_signal`                                           | 10.9 s    | 703 KB  |
| `p`, `g`, `put`, `get`, `put_signal`, signal wait, and `quiet` | 23.7 s    | 1.2 MB  |

Any NVSHMEM module pays about 2.4 s for link-time optimization of the NVSHMEM device library's 8 MB
of IR. NVSHMEM's device functions are always inlined, so each distinct operation adds 100 to 300 KB of
machine code, while further calls to an operation already in the kernel add almost nothing. ptxas time
grows faster than code size: the last row spends 6.3 s in link-time optimization and 17.4 s in ptxas.
Warp's kernel cache stores the result, so the cost is paid once per module and cache. With the default
`enable_backward=True`, the module also links a backward kernel; NVSHMEM modules that do not need autodiff should set
`enable_backward=False`.

nvJitLink options do not reduce this materially:

- `-kernels-used` has nothing to prune. The NVSHMEM device library contains no kernels, and omitting
  Warp's backward kernels made no measurable difference.
- `-optimize-unused-variables`, keeping `nvshmemi_device_state_d`, `nvshmemi_device_lib_version_d`,
  `nvshmemi_ibgda_device_state_d`, and `nvshmemi_gpunetio_device_state_d` through `-variables-used`,
  saved 1 to 6% and up to 16 KB of cubin.
- `-split-compile=0` saved 4 to 8%, because one kernel dominates.
- `-Xptxas=-O1` saved about 16% at the cost of slower device code.

Measure cold links with `-no-cache`. nvJitLink caches results in `~/.nv/ComputeCache`, keyed by LTOIR
bytes that include the directory NVRTC ran in, so compiles from a previously used working directory can
appear several times faster than cold compiles. To compare options without rebuilding Warp, compile
with `WARP_DUMP_LTOIR=1`, which writes `nvrtc_output.ltoir` and the device library to the working directory, and
link those files through `cuda.bindings.nvjitlink` with `-time`.

**Decision (2026-10-25): keep the `nvshmem_put_signal` fallback.** Contiguity is known only at runtime,
so every kernel that calls `nvshmem_put_signal` compiles both the `nvshmem_putmem_signal` path and the
fallback for other layouts (put, quiet, then signal). The fallback accounts for roughly half of that
builtin's link time; the earlier float-only builtin had no fallback but sent the wrong elements for
strided views. Correct results for every layout are worth the one-time compile cost.

To limit the cost in multi-PE jobs, compile modules once rather than on every rank at the same time,
for example with `wp.compile_aot_module` on a machine with the NVSHMEM wheel installed, or by loading on one rank
before the others.

### Unloaded Modules Must Be Unregistered

NVSHMEM records each registered module by the address of its `nvshmemi_device_state_d` global and only
writes device state to addresses it has not initialized yet. When Warp unloaded a registered module
without `nvshmemx_cumodule_finalize`, the CUDA driver placed the next module's state global at the freed
address. NVSHMEM treated the new module as already initialized, so its kernels read zeroed NVSHMEM
state (`nvshmem_n_pes()` returned 0) without any error. This happens whenever a module is reloaded, for
example after editing a kernel. `TestNVSHMEMMultiPE.test_reload_after_unload` covers it.

### Cache Identity Does Not Persist Runtime Initialization State

The module hash determines whether a compiled artifact is reusable, but it does not populate fields
on a new `Module` object. Before `uses_nvshmem` was stored in module metadata, a fresh process could
load a cached NVSHMEM cubin without calling `nvshmemx_cumodule_init`, and its kernels read zeroed NVSHMEM
state. `TestNVSHMEMMultiPE.test_cached_module_in_new_process` runs the same program twice against one
kernel cache and checks that the cached second run still returns correct NVSHMEM values.

The same gap applies to the device library itself: the module hash is computed before codegen knows that a module
uses NVSHMEM, and the kernel cache is keyed by Warp version, so a cubin linked against one NVSHMEM device library is
otherwise reused after the NVSHMEM package is upgraded. Recording the linked library in the metadata (see
[Device Library Discovery](#device-library-discovery)) closes it.

Forcing the rebuild was not enough on its own. When the cache directory already exists, `Module._compile` only
installs its outputs if the cached binary and metadata are missing, on the assumption that another process won the
race with identical outputs. The first implementation of the check recompiled against the new library and then loaded
the stale binary. A rebuild caused by a changed device library now replaces the cached files; the test that caught it
compares the digest recorded in the cache with the library in use rather than trusting the "(compiled)" message.

## Testing Strategy

**Unit tests** (`warp/tests/distributed/test_nvshmem.py`) exercise NVSHMEM the way user programs do,
rather than mocking Warp's module-loading internals, so they survive refactoring of those internals:

- Single-process tests compile NVSHMEM kernels ahead of time, check load-time errors without an NVSHMEM
  runtime, and check CPU and dtype errors and user functions named with the `nvshmem_` prefix.
- Multi-PE tests launched through `mpirun` or `nvshmrun` cover PE queries, element, bulk, and signal
  operations, a module loaded from the kernel cache in a new process, and a module loaded after
  another one is unloaded.
- The UID test checks two-PE UID bootstrap and a remote write as limited-MPG empirical validation over
  local P2P with remote transports disabled.
- The error-message test for unusable runtimes patches Warp's native bindings, because it needs failure codes a test
  machine cannot produce. The prototype's version-rule tests go away with the rule.
- Discovery tests run without a GPU against temporary directories: the override path, the library next to a host
  library, a fake installed distribution, and the error when nothing is found. A multi-PE test switches the override
  between the archive's fatbin and a raw per-architecture LTOIR and checks that the cached module is rebuilt and that
  the recorded digest follows the library in use.
- A CI-only declaration check, outside `warp/tests`: generate `static_assert`s from the `extern "C"` block of
  `nvshmem_device.h` and compile them with `nvcc` against the installed NVSHMEM wheel's headers (see
  [Decision: Hand-Written Device Declarations](#decision-hand-written-device-declarations)).

Run the NVSHMEM tests with an NVSHMEM wheel that ships the device library (3.9 or newer):

```bash
WARP_CACHE_PATH=/tmp/warp-cache-nvshmem \
uv run --with nvshmem4py-cu13 --with "nvidia-nvshmem-cu13>=3.9" --with mpi4py -m unittest \
    warp.tests.distributed.test_nvshmem
```

Until such a wheel exists, point the override at the fatbin from the 3.8.0 archive and use the 3.8.0 host library,
so both halves still come from one release:

```bash
export NVSHMEM_HOME=$PWD/_build/target-deps/libnvshmem/libnvshmem-linux-x86_64-3.8.0_cuda13-archive
export WARP_NVSHMEM_DEVICE_LIBRARY=$NVSHMEM_HOME/lib/libnvshmem_device.ltoir.fatbin
WARP_CACHE_PATH=/tmp/warp-cache-nvshmem \
uv run --with nvshmem4py-cu13 --with nvidia-nvshmem-cu13==3.8.0 --with mpi4py -m unittest \
    warp.tests.distributed.test_nvshmem
```

The MPI-launched tests need a working `mpirun`. With HPC-X Open MPI on a single host without
InfiniBand, `OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader` avoids an `MPI_Init` failure that is unrelated
to NVSHMEM.

The test discovers `NVSHMEM_HOME/bin/nvshmrun.hydra`, prepends the matching host-library directory,
uses an atomic temporary file to share the UID, and assigns one local rank to each CUDA device.

**Prototype validation with the embedded library** (two RTX 3090 GPUs with full peer access, CUDA 13.4, driver 615,
NVSHMEM 3.8.0):

- Full standard Warp build with Packman CUDA-13 NVSHMEM 3.8.0; the embedded 4,230,304-byte fatbin
  was verified byte-for-byte.
- All `test_nvshmem.py` tests, including both MPI-launched multi-PE tests and the UID local P2P test,
  across two physical GPUs.
- `example_nvshmem_basic.py`, `example_nvshmem_put_bw.py`, and `example_jacobi_nvshmem_device.py` with
  two ranks.
- A two-PE `nvshmem_float_p` test with PTX output.

**Prototype validation with NVSHMEM 3.7.1**:

- Full standard Warp build with `--cuda-path /usr/local/cuda-13.3` and the packaged SM 120 raw LTOIR.
- Direct nvJitLink probe using the packaged multi-architecture fatbin as `NVJITLINK_INPUT_FATBIN`.
- Full build and NVSHMEM kernel compile with MathDx disabled.
- Early rejection when CUDA 13.0 is asked to consume the CUDA 13.2-produced 3.7.1 device input
  (`ERROR 4 in nvvmAddNVVMContainerToProgram`).
- One-PE UID initialization with the 3.7.1 host library, symmetric allocation, and a real Warp kernel
  returning PE 0 and one total PE.
- Checked-in two-PE UID test across two MIG slices with `NVSHMEM_REMOTE_TRANSPORT=none`, including a
  device-side write from PE 0 to PE 1 over CUDA IPC/P2P. NVSHMEM reported limited MPG because MPS was
  absent; the test does not claim full-MPG API coverage.
- Fresh-process cache hit that loads the cubin and executes the NVSHMEM module-initialization path.

The device library linking path itself is independent of where the bytes come from: the same nvJitLink call accepted
the embedded bytes, the archive's fatbin passed as a file, and the raw per-architecture LTOIR.

## Follow-up

To raise with the NVSHMEM team:

- **Device library in the PyPI wheels.** Confirmed for 3.9 on 2026-10-30. Nothing further unless the release slips.
- **License files.** Confirmed Apache 2.0 for the 3.7.0 and later redistributable binaries; the redistributable
  manifest's `LICENSE.txt` and the wheel license files are to be updated. No longer affects Warp, which redistributes
  nothing.
- **NVRTC-ready prototype headers: not requested.** Warp keeps hand-written declarations (see
  [Decision: Hand-Written Device Declarations](#decision-hand-written-device-declarations)). Revisit only if NVSHMEM
  ships prototype headers that compile under NVRTC with no CUDA toolkit or libc headers at all, including RTC-only
  definitions of `half` and `__nv_bfloat16`.

For Warp:

- Add the CI declaration check that compiles the hand-written declarations against the installed wheel's headers.
- Decide the public capability query. `wp.is_nvshmem_enabled()` reported the build flag; without a flag the useful
  question is whether a device library was found.
- Add a CI job that runs the single-process tests, and the multi-PE tests on a two-GPU runner.
- Handle CUDA 12, where the per-architecture raw LTOIR must be selected; validate aarch64 and multi-node runs with a
  network transport. Validation so far covers CUDA 13 on x86_64 with GPUs in a single node.
- Expand builtin coverage in the order of the [Coverage Plan](#coverage-plan), starting with the two prerequisites it
  names.
- Add a cooperative launch option to `wp.launch` before the collectives batch.
