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

Each kernel is compiled in its own Warp module (so guards apply independently), cold-compiled 5 times with cache cleared between samples. Measured on L40 GPU, Linux.

| Kernel | CPU main | CPU branch | CPU speedup | CUDA main | CUDA branch | CUDA speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scalar only (trivial) | 1.617s | 0.329s | **4.9x** | 0.410s | 0.146s | **2.8x** |
| Vector math | 1.639s | 0.380s | **4.3x** | 0.447s | 0.192s | **2.3x** |
| Noise + random | 1.663s | 0.432s | **3.8x** | 0.421s | 0.190s | **2.2x** |
| Mat + quat + transform | 1.706s | 0.536s | **3.2x** | 0.492s | 0.288s | **1.7x** |
| Volume sampling | 1.721s | 0.611s | **2.8x** | 0.775s | 0.614s | **1.3x** |
| Mesh queries | 1.815s | 0.824s | **2.2x** | 1.025s | 0.912s | **1.1x** |

Typical real-world kernels use 2–3 feature categories, landing in the **2x–4x CPU** and **1.5x–2.5x CUDA** speedup range. No regressions on any workload.

**Warp FEM examples — compile time (3 samples each):**

| Example | CPU main | CPU branch | CPU speedup | CUDA main | CUDA branch | CUDA speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fem.navier_stokes | 67.1s | 22.6s | **2.97x** | 23.7s | 15.7s | **1.51x** |
| fem.stokes | 55.6s | 17.8s | **3.12x** | 19.2s | 12.0s | **1.60x** |
| fem.deformed_geometry | 50.6s | 16.1s | **3.14x** | 19.9s | 13.4s | **1.49x** |
| fem.diffusion_3d | 42.7s | 13.1s | **3.26x** | 15.2s | 9.2s | **1.65x** |
| fem.convection_diffusion | 31.6s | 9.4s | **3.36x** | 11.0s | 6.8s | **1.62x** |

FEM examples see **3.0x–3.4x CPU** and **1.45x–1.65x CUDA** compile speedup.

<details>
<summary>Reproduce benchmarks</summary>

Isolated kernel benchmark (included in this branch):
```bash
uv run bench_compile_time.py --device cpu --samples 5
uv run bench_compile_time.py --device cuda:0 --samples 5
```

FEM example compile times — clear the kernel cache, then run any FEM example headless and sum the `took N ms (compiled)` lines:
```bash
uv run python -c "import warp as wp; wp.init(); wp.clear_kernel_cache()"
uv run python -m warp.examples.fem.example_navier_stokes \
    --num-frames 1 --resolution 10 --tri-mesh --headless --device cuda:0
```
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
