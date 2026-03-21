# Robust Compile Guards

**Status**: Implemented

## Motivation

Warp's CPU JIT compiler (embedded Clang) spends most of its cold-compile time
parsing native headers (`builtin.h` pulls in mesh, volume, tile, noise, etc.)
rather than compiling the actual kernel code.  For example, a simple 2D array
assignment kernel takes ~1.6s to cold-compile even though the kernel body is
trivial.

An automated investigation (using autoresearch) found that selectively excluding
unused headers via `#define WP_NO_XXX` before `#include "builtin.h"` could
reduce CPU cold-compile time by up to 80%.  Individual headers contribute
anywhere from +27ms (`vec.h`) to +816ms (`float16` adjoint instantiations).

The challenge is determining *which* headers a given kernel actually needs, and
doing so in a way that is robust against future changes to the builtin library.

## Requirements

| ID  | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| R1  | New builtins that omit a guard fail at import time, not at user compile time | Must | |
| R2  | Invalid guard names fail at import time | Must | |
| R3  | Significant CPU cold-compile speedup | Must | |
| R4  | Extend compile guards to CUDA (NVRTC) path | Should | |
| R5  | No manual tables to maintain | Must | |

## Design

### Guard Detection

Three layers determine which features a module needs, plus a safety-net
fallback:

1. **`compile_guard` on `add_builtin()`** -- Every builtin declares which header
   it needs.  Required parameter with no default (`TypeError` at import if
   omitted), validated against `VALID_COMPILE_GUARDS` (`ValueError` for typos).
   Concrete specializations of generic builtins inherit the parent's guard.

2. **Kernel/function/struct type inspection** -- `ModuleBuilder` inspects
   argument types, return types, and struct field types for vec/mat/quat/
   transform/float16/float64, adding the corresponding guards to
   `required_guards`.

3. **Builtin resolution in `Adjoint.add_call()`** -- When the codegen resolves a
   builtin call, `func.compile_guard` is added to `required_guards`.  This is
   transitive across `@wp.func` call chains.

4. **Safety-net source scan** -- After codegen produces the C++ source, a
   lightweight pattern scan catches types created inside `@wp.func` bodies
   that the annotation and type-inspection layers cannot see (e.g., custom
   matrix sizes from FEM geometry, `wp.constant` values).

### Guard Emission

Guard emission inverts the collected set: features NOT in `required_guards` get
`#define WP_NO_XXX`.  Both `cpu_module_header` and `cuda_module_header` accept a
`{compile_guards}` placeholder, so the optimization applies to both CPU (Clang)
and CUDA (NVRTC) compilation.

### Self-Sufficient Native Headers

Each native header `#include`s the types it depends on directly, rather than
relying on `builtin.h`'s include ordering:

- `mat.h`, `hashgrid.h`, `intersect.h`, `texture.h`, `noise.h`, `bvh.h`,
  `rand.h` → `#include "vec.h"`
- `spatial.h`, `svd.h`, `volume.h`, `tile.h` → `#include "mat.h"` (gets
  `vec.h` transitively)

With `#pragma once` on all headers, the C++ preprocessor handles transitive
dependencies automatically.  This means no Python-side dependency table is
needed -- if `noise.h` includes `vec.h`, then including `noise.h` automatically
makes vec types available regardless of whether `WP_NO_VEC` is defined.

### `tile_storage.h` Extraction

`tile_shared_storage_t` (~110 lines) was extracted from `tile.h` (~5300 lines)
into a standalone `tile_storage.h`.  This is included unconditionally from
`builtin.h` (outside the `WP_NO_TILE` guard), so every kernel gets tile shared
storage setup.  `WP_NO_TILE` then fully excludes the heavy tile operations
(registers, reductions, scans, sorts, matmul, FFT).

### Key Implementation Details

**Guard constants** live in `codegen.py` (not `context.py`) to avoid circular
imports.  `COMPILE_GUARD_ALWAYS = ""` is the sentinel for always-included
builtins; `require_guard()` skips empty strings.

**Mixed-group builtins** -- The "Geometry" and "Random" groups in `builtins.py`
contain builtins with different guards (e.g., `mesh_*` -> `WP_NO_MESH`,
`bvh_*` -> `WP_NO_BVH`).  Similarly, "Vector Math" contains both vec-only and
mat-producing builtins (`identity`, `outer`, `transpose`, etc. use `WP_NO_MAT`).

## Benchmark Results

Cold-compile time for a 2D array assignment kernel, median of 5+ samples:

| Configuration | Time | Speedup |
| --- | --- | --- |
| CPU without guards | 1.65s | baseline |
| CPU with guards | 0.33s | **5.0x** |
| CUDA without guards | 0.41s | baseline |
| CUDA with guards | 0.15s | **2.8x** |

Isolated kernels by feature category (each in its own Warp module, 5 samples):

| Kernel | CPU main | CPU branch | CPU speedup | CUDA main | CUDA branch | CUDA speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scalar only | 1.617s | 0.329s | **4.9x** | 0.410s | 0.146s | **2.8x** |
| Vector math | 1.639s | 0.380s | **4.3x** | 0.447s | 0.192s | **2.3x** |
| Noise + random | 1.663s | 0.432s | **3.8x** | 0.421s | 0.190s | **2.2x** |
| Mat + quat + transform | 1.706s | 0.536s | **3.2x** | 0.492s | 0.288s | **1.7x** |
| Volume sampling | 1.721s | 0.611s | **2.8x** | 0.775s | 0.614s | **1.3x** |
| Mesh queries | 1.815s | 0.824s | **2.2x** | 1.025s | 0.912s | **1.1x** |

### Warp FEM Examples

Compile time for FEM examples (3 samples each):

| Example | CPU main | CPU branch | CPU speedup | CUDA main | CUDA branch | CUDA speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fem.navier_stokes | 67.1s | 22.6s | **2.97x** | 23.7s | 15.7s | **1.51x** |
| fem.stokes | 55.6s | 17.8s | **3.12x** | 19.2s | 12.0s | **1.60x** |
| fem.deformed_geometry | 50.6s | 16.1s | **3.14x** | 19.9s | 13.4s | **1.49x** |
| fem.diffusion_3d | 42.7s | 13.1s | **3.26x** | 15.2s | 9.2s | **1.65x** |
| fem.convection_diffusion | 31.6s | 9.4s | **3.36x** | 11.0s | 6.8s | **1.62x** |

### Newton Physics Engine

Newton is a real-world Warp consumer with 30+ compiled modules.  Aggregate
cold-compile time for all Newton modules (3 samples):

| | CPU main | CPU branch | Speedup | CUDA main | CUDA branch | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All Newton modules | 58.9s | 22.1s | **2.7x** | 163.6s | 86.9s | **1.9x** |

## Testing Strategy

Unit tests in `warp/tests/test_compile_guards.py` (22 tests):

- **Validation**: `add_builtin()` raises `TypeError` without `compile_guard`,
  `ValueError` for invalid names.
- **Sweep**: Every registered builtin has a valid `compile_guard`.
- **C++ consistency**: Every guard in `VALID_COMPILE_GUARDS` has a matching
  `#ifndef` in the native headers.
- **Guard computation**: `compute_compile_guards()` tested with empty, full,
  and single-feature scenarios.
- **Source scan**: `scan_source_for_guards()` tested for vec/mat/quat/float16/
  float64 detection, no-match, and idempotency.
- **Type inspection**: `_inspect_type_for_guards()` tested for vec, mat, quat,
  transform, float16, float64, array-of-vec, and scalar-is-noop.
- **Return type inspection**: `build_function()` inspects function return types
  (TDD-verified: test fails without the fix, passes with it).

Full test suite: 6622 tests pass, 0 errors, 16 skipped (matching baseline).

## Alternatives Investigated

**String-scanning approach** -- Scanning the generated C++ source for identifier
substrings (e.g., `"mesh_query"` implies mesh.h is needed) to determine which
guards to emit.  Rejected because the scan table is disconnected from the
`add_builtin()` registrations — a new builtin whose C++ identifier isn't in the
table silently breaks user kernels.  The safety-net source scan (layer 4 above)
is a limited version of this approach, used only as a fallback for types that
the annotation system cannot detect.

**Python-side dependency table** -- A `_COMPILE_GUARD_DEPS` dict mapping header
inclusion constraints (e.g., "WP_NO_VEC can only be emitted if WP_NO_MAT is
also emitted, because mat.h uses vec types").  This required manual maintenance
and was a source of bugs (missing entries caused compile errors).  Eliminated by
making native headers self-sufficient with direct `#include` directives.

**Header restructuring** -- Splitting `builtin.h` into per-feature includes.
Rejected because 12+ headers re-include `builtin.h`, the scalar math section is
tightly interdependent, and the `WP_NO_XXX` model already produces identical
preprocessor output.

**Defaulting `compile_guard` to `COMPILE_GUARD_ALWAYS`** -- Rejected because it
silently over-includes everything, causing gradual performance regression when
new builtins are added without thought to which header they need.
