# JIT Compile Header Exclusion

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

| ID  | Requirement | Priority | Status |
| --- | --- | --- | --- |
| R1  | Invalid header names fail at import time, not at user compile time | Must | Done — `add_builtin()` raises `ValueError` for invalid `native_header` values |
| R2  | Significant CPU cold-compile speedup | Must | Done — up to 5x on CPU, 2.8x on CUDA |
| R3  | Extend compile guards to CUDA (NVRTC) path | Should | Done — both `cpu_module_header` and `cuda_module_header` emit guards |
| R4  | Minimize manual tables | Should | Partial — small tables remain (`VALID_NATIVE_HEADERS`, `_GENERIC_TYPE_HEADERS`) but they are co-located in `codegen.py` and covered by sweep tests |

## Design

### Guard Detection

Three layers determine which features a module needs:

1. **`native_header` on `add_builtin()`** -- Every builtin declares which header
   it needs (e.g. `"mesh"` for mesh.h).  Optional parameter defaulting to
   `None` (always included), validated against `VALID_NATIVE_HEADERS`
   (`ValueError` for invalid names).  Concrete specializations of generic
   builtins inherit the parent's header.

2. **Kernel/function/struct type inspection** -- `ModuleBuilder` inspects
   argument types, return types, and struct field types for vec/mat/quat/
   transform/float16/float64, adding the corresponding headers to
   `required_headers`.

3. **Builtin resolution in `Adjoint.add_call()`** -- When the codegen resolves a
   builtin call, `func.native_header` is added to `required_headers`.  This is
   transitive across `@wp.func` call chains.

4. **Variable creation in `Adjoint.add_var()`** -- When codegen creates a new
   variable (local, temporary, or constant), `_inspect_type_for_headers()` is
   called on its type.  This catches types created inside `@wp.func` bodies
   (e.g., custom matrix sizes from FEM geometry) that are not visible in
   function signatures.

### Guard Emission

Guard emission inverts the collected set: headers NOT in `required_headers` get
`#define WP_NO_XXX` (e.g. header `"mesh"` becomes `#define WP_NO_MESH`).  Both
`cpu_module_header` and `cuda_module_header` accept a `{compile_guards}`
placeholder, so the optimization applies to both CPU (Clang) and CUDA (NVRTC)
compilation.

### Self-Sufficient Native Headers

Each native header `#include`s the types it depends on directly, rather than
relying on `builtin.h`'s include ordering:

- `mat.h`, `hashgrid.h`, `intersect.h`, `texture.h`, `noise.h`, `bvh.h`,
  `rand.h` → `#include "vec.h"`
- `spatial.h` → `#include "mat.h"` (gets `vec.h` transitively)
- `svd.h`, `volume.h`, `tile.h` → `#include "mat_ops.h"` (gets `mat.h` and
  `vec.h` transitively)

With `#pragma once` on all headers, the C++ preprocessor handles transitive
dependencies automatically.  This means no Python-side dependency table is
needed -- if `noise.h` includes `vec.h`, then including `noise.h` automatically
makes vec types available regardless of whether `WP_NO_VEC` is defined.

### `tile_storage.h` Extraction

`tile_shared_storage_t` (~150 lines) was extracted from `tile.h` (~6000 lines)
into a standalone `tile_storage.h`.  This is included unconditionally from
`builtin.h` (outside the `WP_NO_TILE` guard), so every kernel gets tile shared
storage setup.  `WP_NO_TILE` then fully excludes the heavy tile operations
(registers, reductions, scans, sorts, matmul, FFT).

### `WP_NO_BACKWARD` Guard

When `wp.config.enable_backward` is `False`, the codegen emits
`#define WP_NO_BACKWARD` before `#include "builtin.h"`.  This skips:

- All adjoint function stubs in the generated C++ (wrapped in
  `#ifndef WP_NO_BACKWARD` in the codegen template)
- Adjoint operator instantiations in native headers (`array.h`, `mat_ops.h`,
  `quat.h`, `spatial.h`, `intersect_adj.h`, etc.)

This is separate from the per-builtin `native_header` system — it is a
module-level flag driven by configuration rather than feature detection.
FEM workloads benefit the most since they always set `enable_backward=False`.

### `mat.h` Split

`mat.h` was split into two files:

- **`mat.h`** -- Type definition (`mat_t<>` template, constructors, element
  access).  Included whenever `WP_NO_MAT` is not defined.
- **`mat_ops.h`** -- Operations (arithmetic, transpose, determinant, inverse,
  SVD helpers).  Included by headers that need mat operations (`svd.h`,
  `volume.h`, `tile.h`, `spatial.h`).

This allows modules that only *declare* mat-typed arguments (without calling mat
operations) to skip the heavier `mat_ops.h`.

### Key Implementation Details

**Header constants** live in `codegen.py` (not `context.py`) to avoid circular
imports.  `None` is the sentinel for always-included builtins;
`require_header()` skips `None` values.

**Mixed-group builtins** -- The "Geometry" and "Random" groups in `builtins.py`
contain builtins with different headers (e.g., `mesh_*` -> `"mesh"`,
`bvh_*` -> `"bvh"`).  Similarly, "Vector Math" contains both vec-only and
mat-producing builtins (`identity`, `outer`, `transpose`, etc. use `"mat"`).

## Benchmark Results

All benchmarks measured on L40 GPU, CUDA Toolkit 12.8, driver 570.158.01.  Each sample is a true
cold compile (kernel cache fully wiped between samples).  Run sequentially
with `CUDA_CACHE_DISABLE=1`.  121 example/device pairs were tested (22
Warp FEM + 55 Newton examples, CPU and CUDA); representative subsets are
shown below.

### Isolated Kernels

Each kernel is in its own Warp module so compile guards apply independently.
Median of 5 samples:

| Kernel | CPU main | CPU branch | CPU speedup | CUDA main | CUDA branch | CUDA speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scalar only | 1.639s | 0.338s | **4.9x** | 0.409s | 0.146s | **2.8x** |
| Vector math | 1.675s | 0.387s | **4.3x** | 0.442s | 0.193s | **2.3x** |
| Mat + quat + transform | 1.763s | 0.560s | **3.2x** | 0.484s | 0.290s | **1.7x** |
| Volume sampling | 1.742s | 0.614s | **2.8x** | 0.773s | 0.623s | **1.2x** |
| Mesh queries | 1.848s | 0.813s | **2.3x** | 1.026s | 0.910s | **1.1x** |

### Warp FEM Examples

Compile time for FEM examples, median of 3 samples.  22 examples tested;
representative subset shown here:

| Example | CPU main | CPU branch | CPU speedup | CUDA main | CUDA branch | CUDA speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fem.stokes_transfer | 83.7s | 23.3s | **3.59x** | 26.4s | 15.5s | **1.70x** |
| fem.navier_stokes | 68.3s | 21.3s | **3.20x** | 23.6s | 15.7s | **1.51x** |
| fem.stokes | 56.4s | 16.5s | **3.41x** | 19.8s | 12.2s | **1.62x** |
| fem.diffusion_3d | 43.1s | 12.3s | **3.50x** | 15.1s | 9.2s | **1.65x** |
| fem.convection_diffusion | 32.0s | 8.5s | **3.75x** | 11.0s | 6.6s | **1.66x** |
| fem.burgers | 28.8s | 7.3s | **3.94x** | 8.6s | 4.7s | **1.84x** |

Across all 22 FEM examples: **2.8x–3.9x CPU**, **1.3x–1.9x CUDA**.

### Newton Physics Engine

Newton is a real-world Warp consumer with 30+ compiled modules.  55 Newton
examples tested across 9 categories (basic, cable, cloth, contacts, diffsim,
IK, MPM, robot, softbody); representative subset shown here, median of 3
samples:

| Example | CPU main | CPU branch | CPU speedup | CUDA main | CUDA branch | CUDA speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| robot_anymal_d | 91.3s | 45.5s | **2.01x** | 91.1s | 80.6s | **1.13x** |
| robot_ur10 | 75.0s | 33.2s | **2.26x** | 55.0s | 45.9s | **1.20x** |
| selection_articulations | 85.5s | 37.0s | **2.31x** | 108.7s | 99.4s | **1.09x** |
| cloth_hanging | 28.4s | 18.3s | **1.55x** | 47.8s | 44.5s | **1.07x** |
| mpm_granular | — | — | — | 37.0s | 27.6s | **1.34x** |
| diffsim_ball | 13.9s | 8.1s | **1.73x** | 10.8s | 9.9s | **1.09x** |

Across all 55 Newton examples: **1.5x–2.3x CPU**, **1.0x–1.4x CUDA**.  Newton
CUDA improvements are smaller because NVRTC compile time dominates — the
CPU-side header exclusion is less impactful when most time is spent in the GPU
compiler.  MPM examples benefit most on CUDA since they use fewer Warp features.

No regressions detected across any of the 121 example/device pairs.

## Testing Strategy

Unit tests in `warp/tests/test_compile_guards.py` (19 tests):

- **Validation**: `add_builtin()` raises `ValueError` for invalid header names.
- **Sweep**: Every registered builtin has a valid `native_header`.
- **C++ consistency**: Every header in `VALID_NATIVE_HEADERS` has a matching
  `#ifndef WP_NO_XXX` in the native headers.
- **Guard computation**: `compute_compile_guards()` tested with empty, full,
  and single-feature scenarios.
- **`add_var()` header tracking**: `Adjoint.add_var()` tested for vec, mat,
  and scalar types to verify headers are collected during codegen.
- **Type inspection**: `_inspect_type_for_headers()` tested for vec, mat, quat,
  transform, float16, float64, array-of-vec, and scalar-is-noop.
- **Return type inspection**: `build_function()` inspects function return types
  (TDD-verified: test fails without the fix, passes with it).

Full test suite: 6622 tests pass, 0 errors, 16 skipped (matching baseline).

## Alternatives Investigated

**String-scanning approach** -- Scanning the generated C++ source for identifier
substrings (e.g., `"mesh_query"` implies mesh.h is needed) to determine which
guards to emit.  Rejected because the scan table is disconnected from the
`add_builtin()` registrations — a new builtin whose C++ identifier isn't in the
table silently breaks user kernels.  An earlier version used a limited source
scan as a safety net, but this was replaced by tracking headers in
`Adjoint.add_var()` which catches the same cases (types created inside
`@wp.func` bodies) without maintaining fragile string patterns.

**Python-side dependency table** -- A `_COMPILE_GUARD_DEPS` dict mapping header
inclusion constraints (e.g., "WP_NO_VEC can only be emitted if WP_NO_MAT is
also emitted, because mat.h uses vec types").  This required manual maintenance
and was a source of bugs (missing entries caused compile errors).  Eliminated by
making native headers self-sufficient with direct `#include` directives.

**Header restructuring** -- Splitting `builtin.h` into per-feature includes.
Rejected because 12+ headers re-include `builtin.h`, the scalar math section is
tightly interdependent, and the `WP_NO_XXX` model already produces identical
preprocessor output.

**Required `native_header` parameter** -- Making `native_header` a required
positional parameter (no default) so that omitting it raises `TypeError` at
import time.  This was the original design, but relaxed to an optional
parameter defaulting to `None` (always included) to reduce churn when adding
new builtins.  The `test_all_builtins_have_native_header` sweep test mitigates
the regression risk by catching any builtin with `native_header=None` that
should have a specific header.
