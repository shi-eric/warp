# Targeted CUDA Artifact Export for API Capture

**Status**: Implemented (experimental)

**Issue**: [GH-1837](https://github.com/NVIDIA/warp/issues/1837)

## Motivation

API Capture (APIC) saves the operation stream and memory state of a graph to a
``.wrp`` file, with compiled kernel modules in a companion ``_modules``
directory. By default, ``capture_save()`` copies the module that executed during
capture. This makes saving inexpensive, but a captured CUBIN is tied to a narrow
set of CUDA architectures. Deploying the graph on a different GPU can therefore
require capturing again on that GPU or arranging a separate export for every
deployment target.

PTX provides a better portability boundary. Baseline PTX compiled for a virtual
architecture such as ``compute_75`` can be JIT-compiled by the CUDA driver for
devices with the same or a newer compute capability. Warp can also link its
generated CUDA code with MathDx LTO-IR and fatbin inputs before emitting PTX, so
the deployment artifact remains a self-contained code object. Loading it does
not require the original Python program, generated CUDA source, Warp headers, or
link-time inputs.

Users still need the option to export CUBIN when they value exact-target startup
latency over portability. Target architecture and output format are independent
choices: a PTX target describes a compatibility floor, while a CUBIN target
selects native code for the requested architecture.

This design adds an opt-in targeted export mode to ``capture_save()``. It builds
one final PTX or CUBIN per captured module for an explicitly requested target.
Untargeted saving retains its current binary-copy behavior.

## Terminology

- **Captured module**: The ``ModuleExec`` whose kernel was recorded in the APIC
  operation stream.
- **Frozen compile record**: Producer-side state created by the original module
  build: target-neutral CUDA source, native compiler options, target-dependent
  link recipes, metadata expressions, and integrity information.
- **Targeted export**: A save with an explicit ``target_arch`` that recompiles
  captured module variants for a requested CUDA target.
- **Baseline PTX**: PTX without an architecture- or family-specific suffix. Its
  target architecture is the minimum compute capability on which it can load.
- **Target artifact**: The final PTX or CUBIN and its target-specific kernel
  metadata produced for one captured module variant.
- **Capture artifact**: The binary originally loaded on the device that captured
  the graph.

## Requirements

| ID | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| R1 | Export every captured CUDA module for an explicit target architecture | Must | Targeted export is opt-in |
| R2 | Support PTX and CUBIN as independent output choices | Must | CUBIN remains the targeted default |
| R3 | Emit exactly one selected binary per captured module | Must | Do not package source or an alternate-format fallback |
| R4 | Produce self-contained PTX after resolving all generated link inputs | Must | Includes MathDx LTO-IR and fatbin inputs |
| R5 | Preserve captured module semantics and symbol identities | Must | Only target architecture and output format may change; export must not execute Warp Python code generation |
| R6 | Serialize kernel metadata from the target artifact | Must | Do not reuse target-dependent metadata from the capture artifact |
| R7 | Fail when the requested target cannot represent a captured feature | Must | Never raise or lower the requested target silently |
| R8 | Load targeted artifacts through the existing native APIC loader | Must | Python and standalone C++ use the same path |
| R9 | Preserve existing untargeted save/load behavior | Must | No recompilation when no target is requested |
| R10 | Diagnose format and architecture incompatibility before graph construction when possible | Must | CUDA remains authoritative for cases that cannot be checked reliably |

**Non-goals:**

- Packaging exported user-module CUDA source for compilation on the deployment
  system.
- Packaging generated ``.cu`` files, Warp headers, compilation recipes, LTO-IR,
  fatbins, or other intermediate build inputs.
- Re-evaluating Python definitions, globals, closures, or ``wp.static()``
  expressions during targeted export.
- Selecting a deployment device during export.
- Automatically discovering the lowest target accepted by every captured
  kernel.
- Silently replacing unavailable target-specific code with another algorithm.
- Packaging several CUBINs or a CUBIN plus PTX for each module.
- Making a CUDA capture runnable on CPU, or changing CPU object portability.
- Guaranteeing compatibility with a driver, CUDA feature set, or device resource
  limit that the selected target does not support.

## Design

### Public API

``capture_save()`` gains a ``use_ptx`` keyword alongside ``target_arch``:

```python
def capture_save(
    graph: Graph,
    path: str,
    inputs: dict | None = None,
    outputs: dict | None = None,
    *,
    target_arch: int | None = None,
    use_ptx: bool = False,
) -> None: ...
```

The behavior is:

| ``target_arch`` | ``use_ptx`` | Result |
| --- | --- | --- |
| ``None`` | ``False`` | Copy the captured binaries, preserving existing behavior |
| ``N`` | ``False`` | Compile and package CUBIN for ``sm_N`` |
| ``N`` | ``True`` | Compile and package PTX for ``compute_N`` |
| ``None`` | ``True`` | Reject because the PTX compatibility floor is unspecified |

Targeted options require a CUDA APIC graph. ``target_arch`` must be a positive
integer supported by Warp's bundled CUDA compiler. Keeping ``use_ptx=False`` as
the default preserves the targeted CUBIN behavior introduced with
``target_arch`` and matches ``compile_aot_module(..., use_ptx=...)`` naming.

``target_arch`` always means the CUDA compiler target. For baseline PTX it is
also a minimum runtime compute capability. For CUBIN it is a native-code target
with CUDA's narrower binary compatibility. Warp's existing
``cuda_arch_suffix`` policy is resolved and validated for the requested target.
An ``a`` suffix restricts PTX to an architecture-specific target, while an ``f``
suffix restricts it to a compatible architecture family. Warp preserves those
suffixes and does not describe the result as generally forward-compatible.

Examples:

```python
# Portable to compatible devices with compute capability 7.5 or newer.
wp.capture_save(graph, "simulation", target_arch=75, use_ptx=True)

# Native code for sm_89, using CUDA's CUBIN compatibility rules.
wp.capture_save(graph, "simulation", target_arch=89)
```

### Frozen CUDA compile record

Generated CUDA source is the semantic boundary for targeted export. It already
contains resolved ``wp.static()`` values, overload choices, kernel bodies, and
symbol identities. Export compiles that frozen source without executing Warp
Python code generation again.

The immutable version 2 ``CudaCompileRecord`` contains the module hash, active
``block_dim``, source basename and digest, native compiler options, declared
external dependency digests, typed MathDx recipes, and kernel descriptors.
``DotRecipe``, ``FftRecipe``, and ``SolverRecipe`` hold their own concrete native
arguments rather than a generic argument dictionary. They contain no AST,
callback, closure, or live code-generation object.

``CudaKernel`` is the authoritative description of forward/backward symbols,
shared-memory expressions, and cluster requirements. Resolving it for a target
produces a ``CompiledKernel`` with numeric shared-memory sizes. Legacy metadata
is derived from these descriptors rather than maintained independently. APIC
retains the captured executable's resolved descriptors and compile record.

Identical source contents share an immutable in-memory snapshot through a
weak-reference pool. A captured record therefore survives later cache changes,
while releasing its last owner releases the snapshot. Targeted export validates
frozen source and declared dependency digests; a changed or missing dependency
requires rebuilding and recapturing. Untargeted saving needs only the captured
binary and resolved descriptors.

### Target-neutral source and materialization

MathDx source symbols are target-neutral; their LTO cache identities retain the
architecture. The original code-generation pass decides whether an operation
uses MathDx or a scalar fallback. Export preserves that decision: a scalar
fallback stays scalar, and a retained MathDx recipe must succeed for the
requested target.

``build.materialize_cuda_record()`` is the common materializer for JIT, AOT,
and APIC. It returns LTO-IR, fatbins, compiler definitions, and resolved kernel
descriptors. Initial compilation reuses the successful MathDx results obtained
while selecting implementations rather than running those generators again.
Later targets materialize the same typed recipes for their own architecture.

Target-dependent results such as cuFFTDx workspace sizes are expressed through
stable recipe definitions and shared-memory expressions. Resolved per-kernel
shared-memory macros also drive register-spilling decisions, so native code and
serialized launch metadata use the same target values. Materialization neither
rewrites the frozen source nor reruns Python code generation.

### Shared compiler and immutable artifact cache

``cuda_build.compile_cuda()`` owns native compilation and artifact caching for
JIT, AOT, and APIC. It validates the record, materializes the requested target,
then links the frozen source and native inputs into final PTX or CUBIN. Its
``CudaArtifact`` returns the binary location, output kind, target and suffix,
resolved kernels and materialization definitions, and the originating record.

The producer cache separates immutable source, records, and complete artifacts:

```text
<cache_dir>/cuda/
    sources/<source-sha256>.cu
    records/<record-fingerprint>.json
    targets/<inputs-fingerprint>.json
    artifacts/<complete-fingerprint>/
        module.ptx                 # or module.cubin
        artifact.json
```

The complete fingerprint covers the record, target, suffix, binary kind,
materialized LTO-IR/fatbins, definitions, resolved kernels, and compiled binary
digest. ``artifact.json`` stores those inputs and the binary digest. Artifacts
are compiled in a staging directory and published under their fingerprint.
Target indexes map compilation inputs to the latest artifact. Forced builds
that produce different bytes create separate artifacts, preserving binaries
already retained by loaded executables.

Ordinary ``wp_*`` module cache entries retain small ``.cache.json`` indexes
pointing to these artifacts. A cache lookup validates the artifact, source,
record, and dependencies once, then returns the validated object directly to
module loading. Ordinary binary cache hits do not rematerialize MathDx inputs.

Explicit-directory AOT compilation still exports conventional binary, ``.meta``,
and source copies. ``load_aot_module()`` requires the conventional binary and
metadata from an explicit directory. It may attach a frozen record from a
matching producer index after verifying those files, but only the default cache
layout supports index-only loading. Raw binary loads without such a record
remain supported for execution and untargeted APIC export; targeted export
requires the record. LLVM CUDA targeted export remains unsupported.

### Per-save metadata and publication

``capture_save()`` prepares each export independently:

1. Select captured binaries or obtain targeted artifacts from the shared
   compiler. Validate their frozen forward/backward symbol identities.
2. Construct module, resolved kernel, and named binding arrays owned by this
   save. Snapshot memory regions and register mesh records as before.
3. Copy exactly the selected binaries into a staged companion directory and
   serialize the staged ``.wrp`` using a borrowed ``APICExportDescriptor``.
4. Publish the new companions while retaining the old directory, then replace
   the ``.wrp``. Restore the old companions if ordinary publication fails.

The native state owns recorded operations, memory, meshes, and live replay
resources. It no longer retains export module/kernel/binding tables or exposes
registration APIs for them. ``wp_apic_state_save()`` receives the descriptor as
its fifth argument, validates its identities and references, and borrows its
arrays and strings only for that call. A null descriptor supports raw state
saves without export metadata. Repeated saves cannot accumulate old binding
names or overwrite another save's target metadata.

Targeted and untargeted saves use the same packaging path and replace the
companion directory with exactly the selected files. Module filenames use full
hashes, so custom or duplicate producer basenames do not collide:

```text
simulation.wrp
simulation_modules/
    wp_<full-module-hash>.sm75.ptx
```

CUBIN uses ``.cubin``; CPU objects use ``wp_<full-module-hash>.o``. Source,
metadata cache files, and link inputs remain producer-side artifacts.
Compilation or native serialization failures leave the existing bundle in
place. Ordinary publication failures restore the previous companion directory.
The ``.wrp`` and directory are separate sibling paths: publication does not
provide crash-atomic replacement or a consistent snapshot for concurrent readers
between the two replacements.

### Serialized module metadata

APIC wire format version 17 uses a format-neutral module record containing:

- module hash and human-readable module name;
- ``binary_filename``;
- ``binary_kind``: PTX, CUBIN, or CPU object;
- ``target_arch``; and
- ``arch_suffix``.

The header's graph-level device and target information remains available as a
summary, while per-module records are authoritative for module loading and
diagnostics. All modules in a targeted export use the requested target and
format. Untargeted exports describe the actual captured artifact for each
module rather than recomputing its target from mutable global configuration.

The ownership redesign keeps wire format version 17 unchanged. The loader
continues to accept supported older versions and infers their binary kind from
the filename extension. Module validation requires basenames with supported
extensions matching ``binary_kind`` and well-formed architectures and suffixes.
Custom companion basenames remain valid; architecture metadata need not be
encoded in the name. The save descriptor validates module/kernel references and
recorded launch entry points before serialization.

Native accessors use ``binary`` rather than ``cubin`` where they can return
PTX, CUBIN, or CPU objects. Export metadata ownership changes the native save
signature without changing the serialized representation.

### CUDA loading

Python ``capture_load()`` and standalone C++ loading continue to use the same
native ``wp_apic_load_graph()`` path. Targeted artifacts introduce no guest
compiler recipe, source path, include directory, or new standalone API.

The native loader:

1. Parses and validates the file, module records (including companion filenames
   and binary kinds), and operation stream.
2. Queries the physical architecture associated with the supplied CUDA
   context.
3. Rejects baseline PTX when the physical architecture is below its target.
4. Loads every module through Warp's existing native CUDA module loader.
5. Allocates memory, reconstructs objects, resolves kernel functions, and
   lazily builds the CUDA graph only after all modules load. Resolved
   ``CUfunction`` handles are cached per kernel direction and reused for launch
   setup; unused forward/backward directions are not resolved or configured.

For PTX, the existing module loader uses the CUDA driver JIT when the driver can
consume the generated PTX and Warp's embedded ``nvPTXCompiler`` path otherwise.
For architecture-specific PTX, family-specific PTX, and CUBIN, the CUDA loader
remains the final compatibility authority. Warp does not duplicate a partial
architecture table that could disagree with the driver.

There is no alternate-format resolution. A targeted PTX export cannot fall back
to a producer CUBIN, and a targeted CUBIN export cannot find an unrequested PTX.

### Conditional graph helpers

``capture_if()`` and ``capture_while()`` use small Warp-native helper kernels to
set CUDA graph condition handles. These helpers are generated by the replay
runtime and are not captured user modules. Loaded graph reconstruction must not
compile them as CUBIN for the serialized producer target, which can be
incompatible with the load device.

The lazy conditional helper loader first checks its per-context module cache.
Only a cache miss derives the target and output kind from the physical load
context using the native equivalent of Warp's normal CUBIN/PTX selection policy.
An architecture directly supported by the embedded NVRTC can use CUBIN. A newer
architecture can use PTX for a supported virtual
target and let the driver JIT it. The resulting helper module remains cached per
context. APIC scans operations, including nested branches, and preloads helpers
before CUDA capture only when conditional operations are present. Ordinary
launches avoid conditional-helper target selection.

This guest-local compilation is runtime implementation infrastructure, not
source fallback for exported user modules. It needs no serialized user source or
compile recipe.

### Error handling

Targeted export validates public arguments before compilation. It then fails
the entire targeted module preparation phase if any module:

- lacks a complete frozen compile record;
- has invalid frozen source contents or a missing or modified declared external
  dependency;
- has an unsupported compile-record schema or invalid metadata expression;
- requests a cluster, architecture suffix, intrinsic, MathDx implementation, or
  other feature unavailable at ``target_arch``;
- cannot materialize all recorded link recipes and their target metadata;
- is missing a recorded kernel symbol or resolved metadata entry; or
- fails native compilation or linking.

Errors identify the module name and hash, requested format and target, and the
underlying validation or compiler diagnostic. Warp never silently changes the
target, format, architecture suffix, or implementation.

Load errors identify the binary filename and kind, declared target and suffix,
physical device architecture, and CUDA diagnostic. An obviously older device is
rejected before resource allocation for baseline PTX. CUDA decides compatibility
for CUBIN and suffixed PTX, and its error is retained rather than replaced by a
generic APIC message.

## Alternatives Considered

### Regenerate from captured Python code-generation state

The first targeted-export implementation retained a ``ModuleHasher``, kernel
objects, resolved options, definition hashes, and deferred ``wp.static()`` state,
then ran ``ModuleBuilder`` again during ``capture_save()``. This regenerated
target-specific MathDx source and link inputs, but it also reopened arbitrary
Python computation whose results had already been frozen into the captured
executable.

Soundly detecting or replaying every value that can affect a deferred static
expression requires modeling callbacks, attribute descriptors, aliases,
overloaded operations, reflection, and mutable Warp objects. That complexity is
unnecessary when the original generated source already records the result.
Treating the source and a native compile recipe as the frozen boundary preserves
the executable directly and removes Python semantic reconstruction from export.

### Add an architecture macro to existing MathDx symbols

An architecture macro can token-paste the requested target into declarations
and call sites. It cannot generate the corresponding LTO-IR or resolve
target-dependent shared-memory requirements, so it still needs declarative link
and metadata recipes. Keeping architecture-bearing symbols also makes generated
source and diagnostics harder to inspect. Target-neutral internal symbols with
architecture in the LTO cache key provide the same uniqueness with a simpler
source contract.

### Compile the existing source without a record

The module directory's current ``.cu`` is sufficient for ordinary kernels only
when the correct source variant and native flags are known. It does not contain
separately linked MathDx inputs or target-dependent metadata. Making the source
part of a versioned compile record closes those gaps without re-entering Python
code generation.

### Package CUDA source and compile on load

Generated source allows the guest to choose its physical architecture. It also
requires a matching semantic compile recipe, compatible Warp headers, an
embedded compiler, and every external link input. MathDx kernels cannot be
reproduced from ``.cu`` alone because their LTO-IR and fatbin inputs are created
during target-dependent code generation. Final linked PTX provides the desired
forward portability without moving Warp's Python code generator or build state
to the guest.

### Package CUBIN and PTX together

A producer CUBIN could reduce startup latency on a matching guest while PTX
provides fallback elsewhere. This requires multi-artifact module manifests,
selection order, duplicate integrity and diagnostics rules, and larger bundles.
CUDA's driver cache already amortizes PTX JIT cost. The first targeted export
therefore emits exactly the requested format; multi-artifact bundles can be
added separately if deployment measurements justify them.

### Package several target CUBINs

Several CUBINs can cover a known fleet without driver JIT. They do not satisfy
the single-minimum-target portability goal, and they expand the public API and
native resolver beyond what GH-1837 requires.

## Testing Strategy

Tests prioritize the public export boundary: artifacts, failures, and
loaded-graph results. Focused record/cache tests also check source ownership,
schema validation, and cache integrity; these internals are not compatibility
promises.

The behavior matrix covers:

- argument validation for CUDA and CPU graphs;
- CUBIN export for a physical target and baseline PTX export for a minimum
  target, including PTX-to-device JIT playback;
- rejection when baseline PTX requires a newer architecture than the load
  device;
- self-contained PTX and CUBIN playback for a MathDx matrix product and
  target-dependent cuFFTDx workspace;
- distinct module identities and block-dimension variants;
- preservation of captured ``wp.static()`` results after Python state changes;
- loud, transactional failure when a requested target cannot represent a
  captured feature;
- targeted PTX playback of ``capture_if()`` and ``capture_while()``;
- repeated saves with changed bindings and recovery after a failed save;
- exact companion-directory replacement and publication rollback; and
- untargeted explicit binaries, custom basenames, working-directory changes,
  and version 16 file compatibility.

### Verification scope

Native APIC metadata and CUDA conditional code require rebuilding Warp. Run the
complete APIC test module, plus the targeted CUDA record/cache, AOT, shared-memory
spilling, and external-build tests in ``test_module_hashing.py``,
``test_module_aot.py``, ``test_cuda_smem_spilling.py``, and
``test_external_build.py``. Run pre-commit checks for changed files. Set
``WARP_CACHE_PATH`` to a unique path for this worktree for every Warp build and
test command. Broader tests are required only if failures or changes outside the
APIC/module-compilation surface reveal a wider impact.
