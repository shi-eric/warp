## Description

Replace the fragile string-scanning compile guard system with `add_builtin()` annotations and kernel type inspection. This makes compile guards robust against new builtins (omitting `compile_guard` is a `TypeError` at import time) and extends them to the CUDA/NVRTC path.

The prior system scanned generated C++ source for string patterns to decide which headers to exclude. A new builtin whose C++ identifier wasn't in the scan table would silently break user kernels. The new system uses three authoritative sources:

1. **`compile_guard` on `add_builtin()`** — required parameter, validated against `VALID_COMPILE_GUARDS`
2. **Type inspection** — kernel args, function args/returns, struct fields checked for vec/mat/quat/float16/float64
3. **Safety-net source scan** — catches types created inside `@wp.func` bodies (e.g., FEM custom matrix sizes)

Additionally:
- **CUDA compile guards** — extends the optimization to the NVRTC path (previously CPU-only)
- **Self-sufficient native headers** — each header now `#include`s the types it uses directly (e.g., `noise.h` includes `vec.h`), eliminating the brittle `_COMPILE_GUARD_DEPS` dependency table
- **`tile_storage.h` extraction** — separates the lightweight `tile_shared_storage_t` class from the heavy `tile.h` (~5300 lines), allowing `WP_NO_TILE` to fully exclude tile operations

Design doc: `design/robust-compile-guards.md`

## Checklist

- [x] I am familiar with the [Contributing Guidelines](https://nvidia.github.io/warp/user_guide/contribution_guide.html).
- [x] New or existing tests cover these changes.
- [x] The documentation is up to date with these changes.

## Test plan

22 new unit tests in `warp/tests/test_compile_guards.py` covering:
- `add_builtin()` validation (TypeError for missing guard, ValueError for invalid)
- Sweep of all registered builtins for valid guards
- C++ header consistency (every guard has a `#ifndef` in native headers)
- `compute_compile_guards()` logic (empty, full, single feature)
- `scan_source_for_guards()` pattern detection (vec/mat/quat/float16/float64, no-match, idempotent)
- `_inspect_type_for_guards()` for all type categories (vec, mat, quat, transform, float16, float64, array-of-vec, scalar-is-noop)
- `require_guard("")` is a no-op
- Function return type inspection (TDD-verified)

Full test suite: 6622 tests pass, 0 errors, 16 skipped (matching baseline).

### Compile-time benchmarks

All benchmarks use subprocess-per-sample isolation with `WARP_CACHE_PATH`
wiped between samples and `CUDA_CACHE_DISABLE=1`. Measured on L40 GPU,
Linux. Full data and methodology in `_mr_benchmark_data.md` and
`_benchmark_methodology.md`.

**Isolated kernels** (5 samples each, one module per process):

| Kernel | CPU main | CPU branch | CPU speedup | CUDA main | CUDA branch | CUDA speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scalar only (trivial) | 1.661s | 0.344s | **4.8x** | 0.510s | 0.248s | **2.1x** |
| Vector math | 1.674s | 0.392s | **4.3x** | 0.550s | 0.308s | **1.8x** |
| Mat + quat + transform | 1.764s | 0.559s | **3.2x** | 0.672s | 0.389s | **1.7x** |
| Volume sampling | 1.764s | 0.626s | **2.8x** | 0.895s | 0.726s | **1.2x** |
| Mesh queries | 1.852s | 0.822s | **2.3x** | 1.129s | 1.018s | **1.1x** |

**Core Warp examples** (3 samples each, single-module workloads):

| Example | CPU speedup | CUDA speedup | Kernel features |
| --- | ---: | ---: | --- |
| optim.particle_repulsion | **3.8x** | **2.1x** | vec3 basic math |
| core.graph_capture | **3.5x** | **1.8x** | vec2/vec3, noise |
| core.wave | **3.5x** | **1.7x** | scalars |
| core.dem | **3.1x** | **1.6x** | vec3 |
| core.fluid | **2.8x** | **1.4x** | vec3, hash grid |
| core.sph | **2.5x** | **1.3x** | vec3, hash grid |
| core.mesh | **2.3x** | **1.2x** | mesh query |

CUDA speedup correlates with kernel simplicity — simpler kernels use
fewer native headers, so guards exclude more source per NVRTC call.

**FEM examples** (3 samples each, multi-module workloads):

| Example | CPU speedup | CUDA speedup |
| --- | ---: | ---: |
| fem.burgers | **3.9x** | 1.05x |
| fem.convection_diffusion | **3.7x** | 1.09x |
| fem.stokes_transfer | **3.6x** | 1.05x |
| fem.diffusion | **3.6x** | 1.03x |
| fem.navier_stokes | **3.2x** | 1.02x |

FEM compiles dozens of modules per process. PCH amortizes header parsing
across modules, making CUDA baselines fast enough that per-module guard
savings are marginal. CPU (no PCH) still benefits fully.

**Newton examples** — 28 CPU, 50 CUDA examples benchmarked. CPU: **1.5x–2.3x**
speedup. CUDA: within noise (~1.0x) for the same PCH-amortization reason.
No regressions on any workload. See `_benchmark_comparison.md` for full tables.

<details>
<summary>Reproduce benchmarks</summary>

```bash
# From the benchmark worktree (warp-worktree-2):
# Baseline (uses ../warp = main):
python3 _bench_comprehensive.py --label baseline --suite fem --samples 3
python3 _bench_comprehensive.py --label baseline --suite newton --samples 3

# Branch (uses ../warp-worktree-3):
python3 _bench_comprehensive.py --label branch --suite fem --samples 3
python3 _bench_comprehensive.py --label branch --suite newton --samples 3

# Compare:
python3 _bench_compare.py _benchmark_baseline_all.json _benchmark_branch_all.json -o _benchmark_comparison.md

# Core examples:
python3 _run_core_benchmarks.py

# Isolated kernels:
python3 _run_kernel_benchmarks.py
```

See `_benchmark_methodology.md` for pitfalls (PCH reuse, cache isolation,
CUDA driver cache, session survival for long runs).
</details>

## New feature / enhancement

Every `add_builtin()` call now requires a `compile_guard` parameter declaring which C++ header the builtin needs:

```python
from warp._src.codegen import COMPILE_GUARD_ALWAYS

# Scalar math — always available
add_builtin("log", ..., compile_guard=COMPILE_GUARD_ALWAYS)

# Mesh operations — needs mesh.h
add_builtin("mesh_query_point", ..., compile_guard="WP_NO_MESH")

# New builtins that omit compile_guard get TypeError at import time
add_builtin("my_new_func", ...)  # TypeError!
```
