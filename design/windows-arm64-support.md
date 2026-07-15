# Windows ARM64 Support (CPU + GPU Backends)

**Status**: CPU support is productized on ``main``: native builds, public LLVM
SDK consumption, ``win_arm64`` wheels, and ``windows-11-arm`` CI are complete.
The stacked CUDA branch adds the CUDA cross-build path. GPU runtime validation
still needs Windows ARM64 hardware with an NVIDIA GPU; MathDx remains
externally blocked.

## Motivation

Windows-on-ARM devices with NVIDIA GPUs are arriving, and CUDA 13.4 is the
first toolkit to ship Windows ARM64 target libraries (``lib/arm64``,
``bin/arm64``, ARM64 NVRTC/nvptxcompiler static libs). Warp needs to run on
these devices with both backends: the CUDA backend (``warp.dll``) and the CPU
backend (``warp-clang.dll``, which embeds LLVM/Clang to JIT-compile CPU
kernels).

GitHub provides ``windows-11-arm`` hosted runners for public repositories.
Warp now uses them for native CPU builds and tests, backed by a public LLVM
22.1.8 SDK and ``win_arm64`` wheel support. The remaining enablement gap is the
CUDA backend, whose build became feasible with the Windows ARM64 libraries in
CUDA 13.4.

This document captures the full scope of Windows-on-ARM enablement so that
follow-up work (much of it in separate sessions/repos) starts from a shared
foundation. The original exploration lived on
``ershi/cross-compile-windows``. Generic target plumbing now lives on
``ershi/win-arm64-cross-compilation``; CUDA enablement is stacked on it in
``ershi/win-arm64-cuda-13-4``.

## Current State

The CUDA prototype was verified 2026-07-08 on an x86-64 Windows 11 machine;
the CPU productization described below has since landed on ``main``:

- **GPU backend cross-compiles and links.** The generic cross-compilation
  branch adds ``--target-arch {x86_64,aarch64}`` to ``build_lib.py``; the
  stacked CUDA branch supplies CUDA 13.4 target routing. The command

  ```
  uv run build_lib.py --target-arch aarch64 --no-standalone --no-use-libmathdx
      --cuda-path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
  ```

  produces ``warp/bin/arm64/warp.dll`` — a verified ARM64 PE32+ binary
  (machine type ``AA64``, ~2300 ``wp_*`` exports) statically embedding ARM64
  NVRTC. Key mechanics: ``vcvarsall.bat amd64_arm64`` for the MSVC
  cross-toolchain, ``nvcc -target-dir arm64``, ``/MACHINE:ARM64``,
  ``ntdll.lib`` (ARM64 ``nvrtc_static.lib`` references
  ``__imp_RtlGetLastNtStatus``), and CUDA libs from ``lib/arm64``. Device code
  (PTX/cubins) is host-arch independent, so the standard desktop ``gencode``
  set applies; Blackwell-class ARM64 devices are covered by ``compute_121``
  PTX.

- **CPU backend is productized.** Commit ``32f2617b4`` (GH-1755) added native
  Windows ARM64 builds, ``win_arm64`` wheels, hosted ARM64 CI, and the COFF
  JIT fixes. The full CPU suite runs on ``windows-11-arm``.

- **Public LLVM distribution is complete.** The Conan-based pipeline in
  ``tools/llvm/`` builds LLVM/Clang 22.1.8 SDKs for all supported platforms,
  including ``windows-arm64`` with a two-pass native-tablegen cross build.
  ``deps/llvm-deps.packman.xml`` consumes the checksummed SDKs from the
  ``llvm-sdk-22.1.8-warp.1`` GitHub release. See
  ``design/llvm-sdk-conan-build.md`` for the authoritative build and release
  design.

- **libmathdx has no windows-aarch64 package** (``deps/libmathdx-deps.packman.xml``
  lists windows-x86_64, linux-x86_64, linux-aarch64 only), so tile operations
  (``WP_ENABLE_MATHDX``) are disabled on Windows ARM64. The build degrades
  gracefully and fetches by target architecture, so a future package is picked
  up automatically once added to the deps file.

## Requirements

| ID  | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| R1  | ``warp.dll`` for Windows ARM64 (CUDA backend) | Must | Done on ``ershi/win-arm64-cuda-13-4`` via cross-compile; CUDA 13.4+ required |
| R2  | ``warp-clang.dll`` for Windows ARM64 (CPU backend) | Must | Done on ``main`` and validated in ``windows-11-arm`` CI |
| R3  | Public, reproducible LLVM/Clang SDK for all supported platforms | Must | Done via ``tools/llvm/`` and GitHub Releases |
| R4  | CI builds and CPU tests on GitHub ``windows-11-arm`` runners without rebuilding LLVM | Must | Done on ``main`` using the public SDK |
| R5  | ``win_arm64`` wheel packaging and publication | Should | Packaging support done on ``main`` |
| R6  | Native builds on ARM64 Windows hosts (dev machines/devices) | Should | Done on ``main`` via centralized architecture normalization and native MSVC selection |
| R7  | Tile operations (MathDx) on Windows ARM64 | Could | Blocked externally on a windows-aarch64 libmathdx release |

**Non-goals**:

- ARM64EC or 32-bit ARM support.
- GPU test execution in public CI (GitHub hosted ARM64 runners have no GPU).
- Further changes to the completed cross-platform LLVM SDK pipeline.

## Design

The original roadmap split the work into LLVM distribution, Warp integration,
and CI. The CPU-side workstreams are now complete on ``main``. The remaining
stacked-branch work is CUDA cross-compilation and eventual GPU validation.

### Workstream A: LLVM/Clang SDK (complete)

Warp chose a public Conan recipe in ``tools/llvm/`` rather than extending the
internal ``conan-transition`` package. The checked-in pipeline builds the same
stripped, static LLVM/Clang 22.1.8 SDK for Linux, macOS, and Windows targets.
For Windows ARM64 it performs the required native-tablegen pre-pass before the
target build. GitHub Actions assembles SDK releases, and Packman verifies and
downloads their checksummed GitHub Release assets. The full design and release
procedure live in ``design/llvm-sdk-conan-build.md``.

### Workstream B: Warp build system and packaging

The CPU integration is complete on ``main``. Architecture normalization is
centralized in ``warp/_src/build_architecture.py``; native ARM64 MSVC
selection, public SDK consumption, and ``win_arm64`` packaging are all part of
the supported build. ``ershi/win-arm64-cross-compilation`` adds the generic
x86-64-to-ARM64 route, and ``ershi/win-arm64-cuda-13-4`` adds CUDA support.

- **Output layout**: cross builds emit to ``warp/bin/arm64/`` so they can
  coexist with host binaries during development; native ARM64 builds emit to
  ``warp/bin/`` like any other platform. The wheel build should always package
  binaries at ``warp/bin/`` (``setup.py`` requires that). A future packaging
  path for cross-built GPU artifacts must copy from ``warp/bin/arm64/``.

- **Runtime CPU JIT**: ``warp-clang`` JIT-compiles CPU kernels at runtime via
  ORC LLJIT (``warp/native/clang/clang.cpp``). The target triple on Windows
  ARM64 is ``aarch64-pc-windows-msvc``, which emits native COFF objects.
  JITLink has **no** COFF/AArch64 backend — ``JITLink/COFF.cpp`` dispatches only
  ``IMAGE_FILE_MACHINE_AMD64`` — so Warp links these objects with the legacy
  RTDyld layer (``RuntimeDyldCOFFAArch64``), forced on regardless of the
  caller's preference. The tempting alternative, an ``aarch64-pc-windows-elf``
  triple to reach JITLink's mature ELF backend, is unusable: a Windows-OS triple
  forces PIC, which degrades ``CodeModel::Large`` to a small-model ADRP+ADD
  ``:lo12:`` pair, and ``AArch64ELFObjectWriter`` then aborts the process on the
  Windows-flavored ADD-imm12 relocation it cannot represent
  (``AArch64ELFObjectWriter.cpp``). Two further fixes make the native-COFF +
  RTDyld path correct: the JIT uses ``CodeModel::Small`` so ``__chkstk`` lowers
  to a ``bl`` that RTDyld backs with a range-extension stub (Large would emit a
  truncated ADRP+ADD that faults), and the ``SectionMemoryManager`` reserves one
  contiguous slab per object so intra-module page-relative references stay within
  ADRP reach. This path is validated: the full CPU test suite is green on
  ``windows-11-arm``, so the CPU backend carries no outstanding JIT risk.

### Workstream C: CI

- **CPU build and test jobs are complete.** ``.github/workflows/ci.yml``
  builds native Windows ARM64 artifacts, runs the CPU suite, and covers CMake
  consumers on ``windows-11-arm``. These jobs use the public LLVM SDK and do
  not install CUDA.

- **GPU validation** stays manual/internal: an ARM64 Windows device with an
  NVIDIA GPU running the CUDA test suite, eventually as an internal GitLab
  nightly when hardware is available in the runner fleet.

- Keep GitLab (`.gitlab-ci.yml`) and GitHub (`.github/workflows/`)
  lightweight jobs in sync per repo convention where both exist.

### Historical Alternatives Considered

The implemented SDK pipeline selected a public Conan recipe in this repository
and GitHub Release assets. The alternatives below record the tradeoffs that
led to that result.

- **GitHub Release assets or GHCR (oras) for the LLVM prebuilt.** GitHub
  Releases became the selected distribution channel. GHCR would have mirrored
  the Linux builder-image pattern but added another consumer mechanism.

- **Rebuild LLVM in CI guarded by ``actions/cache``.** A cache miss (7-day
  eviction, 10 GiB repo budget) injects a multi-hour LLVM build into PR
  latency; unacceptable.

- **Teach ``build_llvm.py`` to cross-compile LLVM from x86-64** (two-stage
  native-tblgen bootstrap). Solves only the one-off local build and
  duplicates what the Conan recipe must do anyway; native ARM64 runners make
  it unnecessary for bootstrap purposes.

- **Hand-built Packman upload.** The older 15.0.7/18.1.3 packages had no
  public reproducible recipe or CI. The checked-in Conan pipeline and GitHub
  Release workflow replaced this process.

## Testing Strategy

- **Artifact sanity (host-side, automated)**: PE machine type ``AA64`` and
  export presence via ``dumpbin`` for both DLLs; already scriptable on x86-64.
- **CPU JIT validation (complete)**: the CPU suite runs on
  ``windows-11-arm`` and covers the COFF/AArch64 JIT path.
- **GPU suite on device**: manual first (``wp.init`` device enumeration,
  NVRTC kernel compile, full ``-s autodetect`` run), then internal nightly.
- **Wheel install test**: ``pip install`` of the ``win_arm64`` wheel on a
  device, ``import warp``, run an example; interop tests (Torch/JAX) skip.

## Phasing

1. **CUDA cross-build — done on ``ershi/win-arm64-cuda-13-4``**:
   cross-compile ARM64 ``warp.dll`` from x86-64 with CUDA 13.4.
2. **CPU backend and native host support — done on ``main``**: build and test
   ``warp-clang.dll`` on ``windows-11-arm``.
3. **LLVM SDK and CPU productization — done on ``main``**: public Conan
   pipeline, GitHub Release assets, Packman consumption, ``win_arm64`` wheels,
   CI, changelog, and docs.
4. **GPU follow-ups**: validate on Windows ARM64 NVIDIA hardware, add GPU CI
   when hardware is available, and enable MathDx when libmathdx ships a
   Windows ARM64 package.

## Open Questions

- **CUDA toolkit availability in CI**: public availability and installer
  support of the Windows ARM64 CUDA 13.4+ toolkit for hosted runners.
- **GPU validation hardware**: which Windows ARM64 system with an NVIDIA GPU
  should own the initial runtime validation and eventual nightly coverage?
- **MathDx availability**: when will libmathdx publish a Windows ARM64 package,
  and should the GPU build remain MathDx-disabled until then?

## Appendix: debugging the JIT on Windows ARM64

The CPU-JIT bring-up faults were access violations inside JIT-generated code,
with no source, no symbols, and no local ARM64 hardware. The workflow that
finally cracked them is worth recording for the next hard runtime bug on a
platform we can only reach through CI:

- **Dispatch-only debugging on ``windows-11-arm``.** With no ARM64 device on
  hand, every experiment ran as a GitHub Actions job on the hosted
  ``windows-11-arm`` runner. Keep the iteration loop tiny: dispatch one focused
  job, read one artifact, adjust, repeat. The debugger and symbols all have to
  come from the runner image, not a workstation.

- **Staged subprocess A/B repro.** Reduce the failure to the smallest kernel
  that still faults, then drive it from a parent process that launches the
  repro as a child under a debugger. Flip exactly one variable per run (code
  model, memory-manager reservation, ``enable_tiles_in_stack_memory``, ...) and
  compare the two children. This A/B structure is what tied each fault to a
  single root cause instead of a vague crash.

- **Remove ``PYTHONFAULTHANDLER`` before capturing dumps.** Python's fault
  handler installs a first-chance handler that consumes the access violation to
  print its own traceback, which then prevents Windows Error Reporting from
  writing a crash dump. Unset ``PYTHONFAULTHANDLER`` (and any in-process
  ``faulthandler.enable()``) for dump-capture runs so the AV propagates to WER
  or the attached debugger.

- **Live ``cdb`` with a first-chance AV handler.** The decisive signal was the
  faulting PC. Attach ``cdb`` across the process launch with
  ``cdb -g -G -o -c ".childdbg 1; sxe av; g" ...`` so it follows the child
  (``.childdbg 1``), breaks on the **first-chance** access violation
  (``sxe av``) rather than after the stack is unwound, and prints the exact
  instruction pointer. Matching that PC's low bits against
  ``warp-clang.dll+0x1000`` is what proved the fault was a truncated ADRP
  page-relative branch, pointing straight at the code-model / relocation fixes.

- **ARM64 ``cdb`` is already on the runner.** No install step is needed: the
  ``windows-11-arm`` image ships the ARM64 Debugging Tools for Windows under
  ``C:\Program Files (x86)\Windows Kits\10\Debuggers\arm64\cdb.exe``.
