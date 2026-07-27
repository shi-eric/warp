# Robust Compile Guards

**Status**: Shelved (2026-07-27)

> [!IMPORTANT]
> This document records the branch's implemented design, but it is no longer
> an approved implementation or merge plan. Matched measurements against exact
> `main` found only a 0.44% aggregate CPU improvement with default PCH and a
> 1.16% CUDA improvement, with credible regressions in some Newton workloads.
> That product value does not justify the design's permanent maintenance cost.
> Do not merge or refactor this branch on the basis of this document.
>
> The unlisted
> [branch status and measurement record](https://gist.github.com/shi-eric/5fdf6e215795af2461f9bb94aa22499e)
> contains the final decision, exact revisions, application results, and
> artifact hashes. If compile guards are revisited, start a small coarse
> Geometry-and-Tile experiment from then-current `main` rather than reviving
> this complete branch.

## Motivation

Warp's CPU JIT compiler and NVRTC spend a substantial part of cold compilation
parsing and instantiating native declarations that a kernel does not use.
`builtin.h` includes geometry queries, dense math, Tile operations, stochastic
functions, and their adjoints even for a scalar kernel.

Defining `WP_NO_*` macros before including `builtin.h` allows the native
preprocessor to exclude unused feature families. Post-refactor measurements
showed a 5.92x reduction in CPU compilation time without a precompiled header
(PCH) and a 2.23x reduction in CUDA compilation time for a scalar kernel.
Supported FEM workloads showed statistically credible CUDA reductions; CPU
intervals spanned zero. The available Newton checkout was incompatible with
the measured Warp commit and produced no valid post-refactor timing.

The optimization is only sound when Warp can determine every native feature a
module requires. It must also remain maintainable as builtins and native headers
change. A missing requirement is a correctness bug, while an unnecessarily
included family is only a performance cost. The design therefore favors
conservative inclusion and makes native headers responsible for their own
dependencies.

This document specifies the target state for refactoring the existing branch.
It supersedes the branch's former raw-string guard model and its separate
Random, Noise, and Float64 guards.

## Decision summary

- Use 14 conceptual exclusion families: 13 feature families and the independent
  Backward dimension.
- Represent feature requirements in Python as positive `CompileFamily` enum
  members, not negative `WP_NO_*` strings.
- Require every builtin registration to state a family or explicitly state
  `None` for unconditional declarations.
- Keep Mesh, BVH, Intersection, Hash grid, Volume, Texture, Vector, Matrix,
  Quaternion, SVD, Tile, and Float16 separate.
- Combine Random and Noise into one Stochastic family.
- Remove the Float64 family and always include Float64 operations.
- Make guarded native headers directly include their prerequisites; do not keep
  a Python dependency-closure table.
- Use identical canonical feature exclusions for CPU and CUDA source.
- When CPU PCH is enabled, build and reuse one full, unguarded `builtin.h` PCH.
  Module family sets never participate in PCH identity.
- Include a deterministic compile-guard schema fingerprint in module cache
  identity.
- Validate correctness with real compilations, especially on CPU with PCH
  disabled. Do not use timing assertions in CI.

## Scope

### In scope

- Feature detection from builtin calls, types, transitive Warp functions, and
  generated source.
- Canonical macro emission for CPU and CUDA.
- Native feature-header boundaries and dependencies.
- CPU PCH compatibility and cache identity.
- Unit, native-compilation, cache, and integration verification.
- Evidence and acceptance measurements for the refactor.

### Out of scope

- Choosing when CPU PCH should be enabled.
- Guard-aware or feature-specific PCH variants.
- Exposing compile guards as public configuration.
- Automatically changing the family taxonomy when native headers change.
- Retrying a failed guarded compilation with every feature enabled.

The CPU PCH enablement policy requires a separate follow-up design. This
document records the measured PCH break-even behavior but does not turn it into
an enablement threshold.

## Requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| R1 | Every supported module configuration compiles with all required native declarations present. | Must |
| R2 | The implementation has exactly the 14 conceptual families specified here. | Must |
| R3 | Invalid or omitted family metadata fails during builtin registration. | Must |
| R4 | Explicit `None` is reserved for builtins whose native declarations are unconditional. | Must |
| R5 | Detection covers resolved builtins, transitive Warp functions, relevant type graphs, and generated-source fallbacks. | Must |
| R6 | Native feature headers own native dependencies; Python has no dependency-closure table. | Must |
| R7 | CPU and CUDA receive the same deterministic feature-family exclusions. | Must |
| R8 | Backward exclusion reflects effective module and kernel options and remains independent of builtin metadata. | Must |
| R9 | Module cache identity changes when the compile-guard schema changes. | Must |
| R10 | CPU PCH is full, unguarded, and independent of module family sets. | Must |
| R11 | Post-refactor paired measurements cover scalar, family, multi-module, and application workloads; credible regressions are reviewed before implementation status changes. | Must |
| R12 | CI verifies functionality without timing-based assertions. | Must |

## Core invariants

### Conservative correctness

Warp emits an exclusion macro only when complete module analysis finds no use
of that family. Every builtin registration must explicitly select a
`CompileFamily` or `None`. `None` is valid only when the builtin's native
declaration is unconditional. Omitting the argument or supplying an invalid
value fails during registration.

Guard selection is deterministic and module-local. It depends on resolved
builtin calls, transitive `@wp.func` bodies, concrete types, generated source,
and effective backward options. It does not depend on execution history,
previously compiled modules, or the order in which modules load.

### Native dependency ownership

Native headers directly include the headers they need. Python does not expand a
set of required families through a dependency graph. A required feature may
bring in another feature transitively, but the reverse relationship must not be
introduced merely to simplify family selection.

### PCH separation

A CPU PCH contains a full, unguarded `builtin.h`. Feature macros are emitted in
individual module sources and do not affect PCH creation, lookup, or reuse.
This prevents a combinatorial PCH cache and invalid reuse of a PCH that omitted
declarations required by a later module.

## Family taxonomy

`CompileFamily` contains the 13 feature families below. `WP_NO_BACKWARD` is the
fourteenth conceptual family, but it is derived from module options and is not
assignable to a builtin.

| Family | Exclusion macro | Primary native scope | Detection sources |
| --- | --- | --- | --- |
| Mesh | `WP_NO_MESH` | `mesh.h` | Mesh builtins and types |
| BVH | `WP_NO_BVH` | `bvh.h` | BVH builtins and types |
| Intersection | `WP_NO_INTERSECT` | `intersect.h`, `intersect_adj.h` | Intersection builtins |
| Hash grid | `WP_NO_HASHGRID` | `hashgrid.h` | Hash-grid builtins and types |
| Volume | `WP_NO_VOLUME` | `volume.h` | Volume builtins and types |
| Texture | `WP_NO_TEXTURE` | `texture.h` | Texture builtins and types |
| Vector | `WP_NO_VEC` | `vec.h` | Vector types and builtins |
| Matrix | `WP_NO_MAT` | `mat_ops.h` | Matrix types and builtins |
| Quaternion | `WP_NO_QUAT` | `quat.h`, `spatial.h` | Quaternion, transform, and spatial types and builtins |
| SVD | `WP_NO_SVD` | `svd.h` | SVD builtins |
| Tile | `WP_NO_TILE` | Heavy Tile operation headers | Tile builtins |
| Float16 | `WP_NO_FLOAT16_OPS` | Float16 operation and adjoint specializations | Float16 types and builtins |
| Stochastic | `WP_NO_STOCHASTIC` | `rand.h`, `noise.h` | Random and noise builtins |
| Backward | `WP_NO_BACKWARD` | Native adjoints and generated reverse functions | Effective `enable_backward` options |

The following previous macros are removed without compatibility aliases:

- `WP_NO_RAND`
- `WP_NO_NOISE`
- `WP_NO_FLOAT64_OPS`

These macros and builtin registration metadata are internal implementation
details, not public configuration.

## Python representation

### Positive family identifiers

Python code stores positive requirements:

```python
required_families.add(CompileFamily.MESH)
```

It must not store a negative macro string and call that string "required."
Each `CompileFamily` member owns its native exclusion macro and any conservative
generated-source patterns associated with the family.

`add_builtin()` uses a private sentinel to distinguish omission from an
intentional unconditional declaration:

```python
compile_family=_UNSET_COMPILE_FAMILY
```

The registration rules are:

- `_UNSET_COMPILE_FAMILY` raises during registration;
- `None` means the native declaration is intentionally unconditional;
- a `CompileFamily` member assigns a guarded feature family;
- a string, a member of another enum, or any other value raises.

Every direct `add_builtin()` call therefore records an explicit classification.
Concrete specializations of a generic builtin inherit the generic builtin's
family and do not repeat the argument.

The enum and its supporting type mappings live in `codegen.py` to avoid the
existing early-import cycle between code generation and context registration.

### Detection flow

`ModuleBuilder.required_families` begins empty and only grows:

1. When code generation resolves a builtin overload, it adds that overload's
   `compile_family`.
2. Building kernels, functions, and structs recursively inspects:
   - argument and return types;
   - array element types;
   - nested struct fields;
   - vector, matrix, quaternion, transform, and spatial types;
   - Float16 scalar types.
3. Calls through `@wp.func` are built transitively in the same builder, so
   builtin requirements inside user functions reach the containing module.
4. After function and kernel code generation, a conservative source scan adds
   type families that static inspection missed, such as types materialized
   through constants or generated FEM code.

The source scan may add a family but may never remove one. False positives are
safe performance costs. Float64 has no type mapping or source pattern because
it is always included.

### Emission

After detection completes, Warp iterates `CompileFamily` in declaration order.
It emits the exclusion macro for every absent feature family exactly once.
Enum declaration order is the canonical output order.

Backward is evaluated separately. Warp emits `WP_NO_BACKWARD` only when the
module disables backward and no kernel override enables it. This macro suppresses
both native adjoints and generated reverse functions.

The same canonical macro block prefixes CPU and CUDA module source.

## Module cache identity

Warp selects a cached binary from `ModuleHasher` before it generates C++ source.
Consequently, the generated macro block alone cannot make different guard
schemas use different cache identities.

Every module hash includes a deterministic compile-guard schema fingerprint.
The fingerprint covers:

- every family name, macro, and generated-source pattern;
- the generic-type and scalar-type family mappings;
- every registered builtin overload's key, stable signature, native function
  name, and assigned family or `None`.

Records are serialized in a canonical sorted order before hashing, so builtin
registration order does not affect the fingerprint. The fingerprint changes
when a family, mapping, scan rule, or builtin assignment changes.

Within one Warp code version, the module source and options plus this schema
fingerprint determine the emitted family set. Old cache entries may remain on
disk after the refactor, but their old hash makes them unreachable.

## Native-header contract

`builtin.h` owns top-level family selection. Feature headers own the
dependencies required after selection.

### Required dependency directions

- Mesh includes `mat_ops.h`, `bvh.h`, `intersect.h`, and `rand.h`.
- BVH includes `intersect.h` and `vec.h`.
- Intersection includes `vec.h` and `mat_ops.h`.
- Hash grid and Texture include `vec.h`.
- Volume and SVD include `mat_ops.h`.
- Quaternion and Spatial include matrix types.
- Tile includes `rand.h`, `mat_ops.h`, and `tile_storage.h`.
- Noise includes `rand.h` and `vec.h`.
- Random includes `array.h` and `vec.h`.

Every guarded header directly includes its prerequisites. Correctness must not
depend on sibling feature-header order in `builtin.h`.

Family macros gate top-level inclusion from `builtin.h`; they do not prohibit a
required header from including another family transitively. For example,
`WP_NO_STOCHASTIC` may be defined while Mesh directly includes `rand.h`. Once a
feature header is included as a dependency, its own family macro must not
suppress the header's entire contents.

Cross-cutting macros may suppress subsets of an included header:

- `WP_NO_BACKWARD` suppresses adjoint declarations and definitions.
- `WP_NO_FLOAT16_OPS` suppresses Float16 operation specializations, not the
  Float16 type or ABI declarations.

Float64 operations are unconditional.

### Stochastic block

`builtin.h` uses one Stochastic block:

```cpp
#ifndef WP_NO_STOCHASTIC
#include "rand.h"
#include "noise.h"
#endif
```

`noise.h` still includes `rand.h` directly. `#pragma once` makes the repeated
top-level include harmless while preserving standalone correctness.

### Tile and Intersection exceptions

`tile_storage.h` remains outside `WP_NO_TILE`; generated kernels require its
lightweight storage declarations even when they use no heavyweight Tile
operations.

`intersect_adj.h` is included only when Intersection is present and Backward is
enabled.

## CPU PCH contract

When CPU PCH is enabled:

1. Warp creates a PCH from a full `builtin.h` with no feature or Backward
   exclusion macros.
2. PCH storage is runtime- and thread-local, preventing cross-process,
   cross-compiler, and concurrent-writer reuse.
3. Within that storage, the compatibility key includes `block_dim`, debug mode,
   `verify_fp`, `enable_tiles_in_stack_memory`, and a digest of normalized extra
   CPU compiler flags.
4. The key never includes module feature families.
5. Modules with different feature sets and the same compatibility key reuse one
   full PCH.
6. Module macros appear after the PCH is loaded. They cannot remove declarations
   already present in the PCH.
7. `WP_NO_BACKWARD` still avoids generated reverse functions even though the PCH
   already contains native adjoint declarations.

Do not introduce guard-aware PCH variants without new measurements and a
separate design review.

## Failure behavior

- Omitted or invalid family metadata fails during builtin registration.
- Explicit `None` is accepted only as the registration's assertion that the
  native declaration is unconditional.
- An unknown value in `required_families` fails before native compilation.
- A false-positive source match includes too much native code and remains
  correct.
- A false-negative detection or missing native dependency is a correctness bug.
  Warp reports the native compilation failure and does not retry unguarded.

An automatic unguarded retry would hide metadata and dependency defects, make
cache behavior depend on failure history, and allow correctness coverage to
decay.

This prohibition does not apply to PCH recovery. If PCH creation fails, Warp
compiles the same guarded source without PCH. If an existing PCH is corrupt or
incompatible, Warp deletes that PCH and retries the same guarded source without
it. The retry changes only the PCH mechanism, never the feature-family set.

## Verification

### Registration and emission tests

- Family macros are unique.
- Omitting `compile_family` fails during direct builtin registration.
- Explicit classifications accept only `CompileFamily` members or `None`.
- Generic builtin specializations inherit their parent's family.
- Empty, full, and single-family sets emit the expected canonical block.
- Python and native sources contain none of the removed macro names.
- Changing a family, mapping, source pattern, or builtin assignment changes the
  schema fingerprint.

### Detection tests

- Kernel and function arguments and returns contribute families.
- Arrays, nested structs, and transitive `@wp.func` calls contribute families.
- Vector, Matrix, Quaternion/Transform/Spatial, and Float16 paths are covered.
- Source scanning only adds families and has explicit false-positive coverage.
- Float64 contributes no family.
- Mixed module and kernel backward settings produce the specified Backward
  decision.

### Real compilation tests

For every family, compile a minimal real kernel on every backend that supports
that feature. CPU-supported families must compile with CPU PCH disabled.
CUDA-only families, including Texture, must compile on CUDA. Tests must:

- cover Random-only and Noise-only kernels independently;
- exercise Intersection, BVH, and Mesh separately;
- exercise Vector, Matrix, Quaternion, and SVD separately;
- cover Tile, Float16, and unconditional Float64 behavior;
- include representative forward-only and backward-enabled modules;
- assert the expected required-family set and emitted macros before compiling.

Asserting the family set prevents an accidentally unguarded translation unit
from making a compilation test pass. Tests must not clear kernel or LTO caches;
the suite runner owns cache isolation.

### PCH and cache tests

- Compile a sequence of CPU modules with different family sets and verify that
  PCH mode creates and reuses one compatible full PCH.
- Verify that changing a PCH-affecting option selects a different PCH and that
  module family changes alone do not.
- Verify that a corrupt or incompatible PCH is removed before the same guarded
  source is retried without PCH.
- Compile the same CPU-supported modules with PCH disabled.
- Verify module hashes change when the relevant guard schema changes.
- Use functional assertions only, never timing thresholds.

### Integration verification

Because the refactor changes native headers, rebuild Warp before running tests.
Run the compile-guard tests, the registered family compilation matrix, and the
full Warp suite on CPU and CUDA. Since `builtins.py` changes, run the
documentation build to regenerate `warp/__init__.pyi`. Run pre-commit checks on
all changed files.

## Performance evidence

### Controlled ablation setup

The follow-up study used fresh compiler workers, a unique Warp cache for every
timing, an alternating paired order, and seven samples per cell. CPU family
measurements disabled PCH so header parsing remained observable. CUDA used its
normal PCH setting and disabled the CUDA driver cache. Reported intervals are
95% paired Student t intervals.

For a scalar kernel, excluding all current families reduced compilation time:

- CPU without PCH: 6.11x speedup, or an 83.63% reduction.
- CUDA with PCH: 2.27x speedup, or a 55.88% reduction.

### Coarse family effects

| Family | CPU, PCH off | CUDA, PCH on | Interpretation |
| --- | ---: | ---: | --- |
| Geometry | 196.22 ms `[191.46, 200.99]` | 113.84 ms `[112.13, 115.56]` | Strong on both |
| Tile | 61.45 ms `[57.91, 64.99]` | 41.68 ms `[39.82, 43.54]` | Strong on both |
| Backward | 56.87 ms `[53.38, 60.36]` | 39.91 ms `[37.33, 42.49]` | Strong on both |
| Dense math | 39.76 ms `[32.72, 46.80]` | 28.35 ms `[27.05, 29.65]` | Strong on both |
| Stochastic | 33.09 ms `[31.58, 34.60]` | 20.54 ms `[19.41, 21.67]` | Strong on both |
| Float16 operations | 58.99 ms `[55.81, 62.17]` | 1.66 ms `[-3.01, 6.32]` | CPU benefit |
| Float64 operations | 1.00 ms `[-2.34, 4.33]` | -0.01 ms `[-3.03, 3.00]` | No demonstrated benefit |

All demonstrated effects were positive in every pair. Float16 on CUDA and
Float64 on both backends had mixed signs and intervals spanning zero. Float16
is retained because CPU cold compilation matters; Float64 is removed.

### Workload-aware partition study

A second study compiled minimal real kernels for the geometry and dense-math
features under nested policies containing 6 through 14 conceptual families.
Every policy retained Tile, Backward, Float16, and combined Stochastic.
Float64 remained unconditional.

| Families | Policy | CPU suite | CPU tax vs. 14 | CUDA suite | CUDA tax vs. 14 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 6 | Coarse geometry and dense math | 2914.6 ms | 771.8 ms (36.0%) | 2421.6 ms | 527.3 ms (27.8%) |
| 9 | Four geometry, one dense math | 2443.3 ms | 300.6 ms (14.0%) | 2069.9 ms | 175.6 ms (9.3%) |
| 10 | Four geometry, two dense math | 2308.4 ms | 165.7 ms (7.7%) | 1982.5 ms | 88.2 ms (4.7%) |
| 11 | Four geometry, three dense math | 2287.8 ms | 145.1 ms (6.8%) | 1979.1 ms | 84.8 ms (4.5%) |
| 12 | Five geometry, three dense math | 2206.6 ms | 63.9 ms (3.0%) | 1939.9 ms | 45.6 ms (2.4%) |
| 13 | Six geometry, three dense math | 2169.8 ms | 27.1 ms (1.3%) | 1915.3 ms | 21.1 ms (1.1%) |
| 14 | Final taxonomy | 2142.7 ms | baseline | 1894.3 ms | baseline |

The later splits provide modest per-module savings that accumulate across
multi-module programs. The final split from 13 to 14 families saved 27.1 ms on
the CPU suite and 21.1 ms on the CUDA suite.

The repository inventory covered 1,519 decorated Warp functions in 253 test and
example files. Geometry leaves were usually used independently. Matrix and
Quaternion co-occurred far less often than either appeared alone. Noise always
co-occurred with Random, which supports combined Stochastic and matches the
native dependency finding.

### CPU PCH interaction

With a full CPU PCH, per-header feature deltas spanned zero because the
declarations were already precompiled. Backward still saved 19.11 ms
`[6.84, 31.38]` because it also suppresses generated reverse code.

Seven paired workers compiled a rotating sequence of seven modules. Every
worker created one PCH with the same filename, size, and content hash across
all module family sets.

| Modules compiled | PCH-off total | PCH-on total | PCH time saved |
| ---: | ---: | ---: | ---: |
| 1 | 234.86 ms | 1105.51 ms | -870.66 ms |
| 2 | 464.96 ms | 1173.95 ms | -708.99 ms |
| 3 | 692.80 ms | 1241.84 ms | -549.04 ms |
| 4 | 920.47 ms | 1309.62 ms | -389.15 ms |
| 5 | 1149.79 ms | 1377.45 ms | -227.65 ms |
| 6 | 1377.98 ms | 1445.27 ms | -67.29 ms `[-136.35, 1.77]` |
| 7 | 1607.93 ms | 1513.03 ms | 94.90 ms `[71.85, 117.95]` |

PCH lost for one through five mixed modules, was inconclusive at six, and
demonstrated a benefit at seven on this host and workload. This result motivates
a separate enablement-policy study; it does not define a universal threshold.

### Historical application measurements

Measurements from the original branch demonstrate that guards can matter in
larger applications, but they predate the final taxonomy and are not acceptance
measurements for this refactor. CPU PCH support was not part of these
measurements.

| Workload | CPU main | CPU guarded | CPU speedup | CUDA main | CUDA guarded | CUDA speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FEM Navier-Stokes | 67.1 s | 22.6 s | 2.97x | 23.7 s | 15.7 s | 1.51x |
| FEM Stokes | 55.6 s | 17.8 s | 3.12x | 19.2 s | 12.0 s | 1.60x |
| FEM Diffusion 3D | 42.7 s | 13.1 s | 3.26x | 15.2 s | 9.2 s | 1.65x |
| All Newton modules | 58.9 s | 22.1 s | 2.7x | 163.6 s | 86.9 s | 1.9x |

### Post-refactor validation (2026-07-26)

Commit `15d36a1eb27a4fa5c0fea8645b7ef744913298ec` was measured on
`ershi-interactive-autorun`, running Linux 6.8.0 on one 96-core,
192-thread AMD EPYC 9B45 socket. CUDA measurements used one NVIDIA RTX PRO
6000 Blackwell Server Edition GPU (compute capability 12.0, 97,887 MiB), with
the 580.126.20 driver and CUDA 13.0 driver API and toolkit. The environment
used Python 3.12.13 and Warp 1.13.0.dev0.

Each comparison used seven alternating, paired, fresh-process samples and a
unique Warp cache per worker. CUDA driver caching was disabled. Family and
partition measurements used CPU PCH off and CUDA PCH on; application
measurements used PCH on for both CPU and CUDA. Intervals below are paired 95%
Student t intervals. No raw physical cell had a coefficient of variation above
10%, no outlier was discarded, and no cell required recollection. The scalar,
family, partition, PCH, and supported application collections completed
28/28, 168/168, 1568/1568, 14/14, and 84/84 workers, respectively. Their
maximum raw-cell coefficients of variation were 1.29%, 1.55%, 2.47%, 0.89%,
and 0.61%.
The rotating PCH order made the derived one-, two-, and three-module PCH-off
prefix totals vary by 25.19%, 14.31%, and 10.76%; these are not separately
recollectable cells, and their paired intervals below retain that variation.

The scalar comparison forced all 13 feature families plus Backward against the
production guarded kernel:

| Device | Production | Forced unguarded | Time saved | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU, PCH off | 175.98 ms | 1041.59 ms | 865.61 ms `[855.69, 875.53]` | 5.92x |
| CUDA, PCH on | 150.84 ms | 335.68 ms | 184.84 ms `[182.22, 187.46]` | 2.23x |

For each coarse family, the table reports the additional compile time from
forcing that family into an otherwise minimal scalar kernel. Every CPU effect
and every CUDA effect except Float16 excluded zero:

| Family | CPU, PCH off | CUDA, PCH on |
| --- | ---: | ---: |
| Geometry | 195.25 ms `[193.14, 197.35]` | 113.72 ms `[111.36, 116.08]` |
| Tile | 61.77 ms `[59.01, 64.53]` | 42.25 ms `[40.51, 43.99]` |
| Backward | 57.37 ms `[55.78, 58.95]` | 40.66 ms `[38.14, 43.18]` |
| Dense math | 43.49 ms `[41.12, 45.87]` | 28.33 ms `[26.00, 30.65]` |
| Stochastic | 33.44 ms `[32.03, 34.85]` | 19.68 ms `[16.69, 22.67]` |
| Float16 operations | 58.81 ms `[57.66, 59.95]` | 0.50 ms `[-1.14, 2.14]` |

The first post-refactor partition artifact used a rotated seven-policy block
and shared one final observation among comparators. It therefore did not
alternate every logical comparator/final pair. That 679-worker artifact,
`_guard_post_refactor_partition_results.json`, SHA-256
`f03cb19091979d76492e9e106487416d012dbd5b2e48c3ee72aa1dc43d46c6e8`,
is superseded and contributes no result below.

The replacement workload-aware partition study held Tile, Backward, Float16,
and Stochastic constant. Each non-identical comparator/final observation was
its own adjacent pair, and the two sides reversed order on every successive
sample. The complete plan produced 784 timed pairs in 1,568 successful workers
with 1,568 unique caches. Another 602 sample-level comparisons were explicit
zero-effect aliases whose comparator and final expansions the analyzer
independently proved identical. There were no failures or recollections. The
replacement artifact
`_guard_post_refactor_partition_paired_results.json`, SHA-256
`4721825d46501fd379c5d50d1a0aa2ef7cebfc45a889d3bd1db5cce8a6e92c3a`,
had a maximum physical-cell coefficient of variation of 2.47%.

Suite effects sum the paired per-workload deltas within each sample; proven
aliases contribute exactly zero. No unpaired absolute suite mean is used.
Every coarser-policy tax below excluded zero:

| Policy | CPU tax vs. final | CUDA tax vs. final |
| --- | ---: | ---: |
| Coarse geometry and dense math | 775.44 ms `[769.72, 781.15]` | 507.73 ms `[501.50, 513.96]` |
| Four geometry, one dense math | 304.61 ms `[295.61, 313.61]` | 166.59 ms `[161.30, 171.89]` |
| Four geometry, two dense math | 168.45 ms `[165.99, 170.92]` | 90.11 ms `[82.86, 97.35]` |
| Four geometry, three dense math | 150.26 ms `[145.76, 154.76]` | 84.17 ms `[75.65, 92.70]` |
| Five geometry, three dense math | 67.15 ms `[60.57, 73.72]` | 39.96 ms `[37.09, 42.82]` |
| Six geometry, three dense math | 27.45 ms `[25.43, 29.46]` | 18.44 ms `[15.58, 21.30]` |

The replacement confirms that the final Matrix/Quaternion split produces a
small but statistically credible accumulated benefit on both devices. It
found no statistically credible partition regression.

The CPU sequence study reproduced the full-PCH compatibility invariant. Each
of seven PCH-on workers compiled seven different family sets while creating
exactly one PCH. All workers used the same filename, 3,855,052-byte size, and
SHA-256 digest
`0580a13082fcd69682fd2cf6866b155f5726884a98839342a018583936b34855`.

| Modules compiled | PCH-off total | PCH-on total | PCH time saved |
| ---: | ---: | ---: | ---: |
| 1 | 238.31 ms | 1103.70 ms | -865.39 ms `[-924.20, -806.58]` |
| 2 | 469.48 ms | 1172.03 ms | -702.55 ms `[-771.47, -633.64]` |
| 3 | 699.04 ms | 1240.06 ms | -541.01 ms `[-613.08, -468.95]` |
| 4 | 928.95 ms | 1308.95 ms | -379.99 ms `[-451.35, -308.63]` |
| 5 | 1156.44 ms | 1376.94 ms | -220.50 ms `[-280.65, -160.35]` |
| 6 | 1383.56 ms | 1445.07 ms | -61.51 ms `[-112.58, -10.44]` |
| 7 | 1611.11 ms | 1513.24 ms | 97.87 ms `[83.42, 112.32]` |

The observed crossover was between six and seven modules on this host and
workload. This does not change the PCH enablement policy.

Application measurements compared production guards with all feature-family
guards suppressed at the same commit. Backward behavior was unchanged. The
table reports unguarded compile time minus production compile time, so a
positive value is time saved by production guards:

| Workload | Device | Production | Time saved | Speedup |
| --- | --- | ---: | ---: | ---: |
| FEM Diffusion 3D | CPU | 5.487 s | 0.004 s `[-0.024, 0.031]` | 1.001x |
| FEM Navier-Stokes | CPU | 9.728 s | 0.032 s `[-0.009, 0.073]` | 1.003x |
| FEM Stokes | CPU | 7.018 s | 0.020 s `[-0.022, 0.062]` | 1.003x |
| FEM Diffusion 3D | CUDA | 6.046 s | 0.474 s `[0.449, 0.499]` | 1.078x |
| FEM Navier-Stokes | CUDA | 11.301 s | 0.535 s `[0.489, 0.580]` | 1.047x |
| FEM Stokes | CUDA | 8.026 s | 0.523 s `[0.492, 0.554]` | 1.065x |

The CPU application intervals spanned zero, while all three CUDA application
benefits were statistically credible. Among supported, valid comparisons,
there were no statistically credible regressions attributable to the
production guards in any scalar, family, partition, or FEM comparison. The
PCH-on startup tax through six modules and benefit at seven remain a separate
enablement-policy input. Newton remains an incomplete acceptance input.

Newton could not provide valid post-refactor timing evidence. The available
Newton checkout at commit
`92f63c84e2c7f7ca6621e2c47355e3cb39378ad1` declares
`warp-lang>=1.16.0.dev20260716`, but the frozen Warp commit identifies as
1.13.0.dev0. Support probes for `robot_ur10`, `basic_shapes`,
`cloth_hanging`, and `diffsim_ball` failed under both guard policies on CPU
and CUDA before completion: the first expected `warp.config.log_level`, while
the other three expected `warp.DeterministicMode`. The probes recorded partial
compilation totals before those API exceptions, but produced no valid
aggregate timing. The 16 failed probe records, partial totals, and diagnostics
remain diagnostic only and were not converted into timing samples.

This evidence does not change the recommended 13 feature families plus
Backward. Float64 remains unconditional, Stochastic remains combined, and the
PCH compatibility invariant remains a full PCH independent of module family
sets. The raw collections and machine-readable analysis are stored locally as
`_guard_post_refactor_*_results.json`,
`_cpu_pch_post_refactor_sequence_results.json`,
`_guard_post_refactor_newton_support_probe.json`, and
`_guard_post_refactor_analysis_fix_round_1.json`. For partition evidence,
`_guard_post_refactor_partition_paired_results.json` is authoritative and the
earlier partition artifact is superseded. All remain uncommitted benchmark
artifacts.

## Performance acceptance protocol

The completed post-refactor study followed the protocol below. Reuse it after
future material changes to the family taxonomy or guard implementation:

1. Re-run paired fresh-process measurements with unique Warp caches.
2. Measure CPU guards with PCH disabled and CUDA with its normal PCH setting.
3. Collect at least seven paired samples per comparison and report paired 95%
   confidence intervals.
4. Treat a regression whose paired 95% interval excludes zero as statistically
   credible; investigate it rather than hiding it in an aggregate.
5. Re-run representative FEM workloads and, when a compatible checkout is
   available, Newton workloads.
6. Update this evidence section with production-refactor measurements.

CI must not assert timing ratios. The acceptance decision is evidence-based and
considers both individual family workloads and accumulated multi-module costs.

## Migration

The old and new taxonomies must not coexist:

1. Add failing tests for the final taxonomy, Noise-only CPU compilation without
   PCH, unconditional Float64 behavior, and schema-sensitive module hashing.
2. Introduce `CompileFamily`, the required explicit registration sentinel,
   positive `required_families`, canonical emission, and the schema fingerprint.
3. Explicitly classify every direct builtin registration and convert type and
   source mappings to family members.
4. Add Stochastic, fix `noise.h`, remove the Random, Noise, and Float64 macros,
   and preserve the documented native dependency directions.
5. Integrate the full, unguarded CPU PCH implementation.
6. Run the real family compilation matrix before broader verification.
7. Rebuild Warp, regenerate `warp/__init__.pyi`, run pre-commit, and run the full
   suite.
8. Collect post-refactor timing evidence and update this document.

The refactor is implemented at commit
`2009d425439cfd9ea3a3f313774171173b416e55`. The exact product tree ran
12,554 tests: 12,550 passed and four skipped, with no failures. The focused
compile-guard and PCH matrix also passed with native libraries built against
LLVM 18.1.3, 21.1.0, and 22.1.8.

The post-refactor validation values above remain measurements of commit
`15d36a1eb27a4fa5c0fea8645b7ef744913298ec`. Subsequent fixes serialized
schema mutation and repaired compiler diagnostics and version compatibility;
they did not change the family taxonomy, emitted guards, or PCH compatibility
invariant.

## Alternatives rejected

### Five or six coarse families

Coarse families captured most scalar-kernel benefit but imposed measured
grouping taxes of up to 36.0% on CPU and 27.8% on CUDA across ordinary
single-feature modules. Small independent savings also accumulated across
multi-module suites.

### Separate Random and Noise families

A Noise-only kernel failed to compile on CPU without PCH because `noise.h`
called functions from `rand.h` without including it. CPU PCH masked this defect.
Repository usage also found Noise and Random together, so the additional
variable had no demonstrated value.

### A Float64 family

The measured Float64 effect was indistinguishable from zero on both backends.
Keeping the variable would add metadata, native conditionals, and tests without
demonstrated compilation benefit.

### Raw negative macro strings in Python

The old `required_guards` set contained values such as `WP_NO_MESH`, so
"requiring" a value meant suppressing its emission. Positive typed families
make the state legible and make misspelled strings invalid by construction.

### Optional family metadata

Using `None` as both the default and the explicit unconditional classification
cannot distinguish an intentional core builtin from a forgotten assignment. A
forgotten assignment can exclude the header containing that builtin rather
than fail open. The private omission sentinel makes every registration's intent
explicit while allowing generic specializations to inherit it.

### Python-side dependency closure

A Python dependency table duplicates the native include graph and can silently
become stale. Direct native includes let the C++ preprocessor resolve
transitive dependencies.

### Source scanning as the primary detector

Scanning generated C++ identifiers for every feature would duplicate builtin
registration metadata and could miss new native entry points. The source scan
remains only a conservative fallback for type materialization.

### Guard-aware CPU PCH variants

Feature-specific PCHs would fragment cache reuse and risk loading a PCH that
omitted declarations required by another module. The measured full PCH reused
one identical artifact across different family sets.

### Generated-source-only cache identity

Cache lookup occurs before generated C++ is written. A schema fingerprint is
required to invalidate cached modules when guard metadata changes.

### Automatic unguarded retry

Retrying after native compilation fails would mask missing metadata and header
dependencies and make cache behavior depend on compilation history. Retrying
the same guarded source without a failed PCH remains allowed.

### Choosing a PCH enablement threshold here

The observed break-even point came from one host and one mixed module sequence.
Selecting a default or automatic policy requires broader workloads and systems
and belongs in a separate design.
