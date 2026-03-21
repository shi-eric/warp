# Compile Guard Benchmark Methodology

Reference document for reproducing and extending the compile-time benchmarks
used to evaluate the robust compile guards branch (`ershi/robust-compile-guards`)
against `main`.

## Environment

- **GPU**: NVIDIA L40 (47 GiB, sm_89)
- **OS**: Linux 5.15.0-113-generic
- **CUDA**: Toolkit 12.8, Driver 12.8
- **Python**: 3.12 (via uv)
- **Warp**: 1.13.0.dev0

### Worktree Layout

| Path | Branch | Purpose |
| --- | --- | --- |
| `../warp` | `main` | Baseline Warp (no compile guards) |
| `warp-worktree-3` | `ershi/robust-compile-guards` | This branch |
| `../newton` | N/A | Newton with `../warp` installed in `.venv` |
| `../newton-benchmark-expt` | N/A | Newton with `warp-worktree-3` installed in `.venv` |

### Key Rule: Use `python`, Not `uv run`

In Newton worktrees, always use `.venv/bin/python <file.py>`, never `uv run`.
`uv run` resolves dependencies from `uv.lock` and will reinstall the locked
Warp version, overwriting the manually installed development copy.

## Scripts

All scripts live in the `warp-worktree-3` repo root (not committed).

| Script | What it measures |
| --- | --- |
| `bench_compile_time.py` | Single trivial kernel cold-compile time (committed) |
| `_bench_comparison.py` | Isolated Warp kernels by feature category (7 kernels, each in own module) |
| `_bench_newton.py` | Newton module-level compile times via `wp.load_module(recursive=True)` |
| `_bench_newton_examples.py` | Newton examples end-to-end wall time (`--viewer null --num-frames 1`) |
| `_bench_newton_robots.py` | Newton robot examples specifically (same approach as above) |

### Running Scripts

**Isolated Warp kernels** (run from Warp repo root):
```bash
uv run _bench_comparison.py --device cpu --samples 5
uv run _bench_comparison.py --device cuda:0 --samples 5
```

**Newton modules** (run from Newton worktree):
```bash
.venv/bin/python /path/to/_bench_newton.py --device cpu --samples 3
.venv/bin/python /path/to/_bench_newton.py --device cuda:0 --samples 3
```

**Newton examples** (run from Newton worktree):
```bash
.venv/bin/python /path/to/_bench_newton_examples.py --device cuda:0 --samples 3
.venv/bin/python /path/to/_bench_newton_robots.py --device cuda:0 --samples 3
```

## Methodology

### Cold-Compile Measurement

Every benchmark clears the Warp kernel cache (`wp.clear_kernel_cache()`) and
unloads modules between samples. This forces full recompilation through the
JIT compiler (Clang for CPU, NVRTC for CUDA). The CUDA compute cache is also
disabled via `CUDA_CACHE_DISABLE=1`.

### Statistical Approach

- **Samples**: 3–5 per measurement (3 for Newton examples due to long runtimes)
- **Metric**: Median (robust to outliers)
- **Reported**: Median, mean, stdev, CV where applicable
- **Acceptable CV**: < 5% for isolated kernels, < 10% for end-to-end examples

### Isolated Kernel Design

Each kernel is defined in a separate Python file under `bench_kernels_src/` so
it gets its own Warp module. This ensures compile guards apply independently —
a scalar-only kernel shouldn't be penalized by a mesh kernel in the same module.

Kernels are designed to exercise specific feature categories:

| Kernel | Features used | Guards that fire |
| --- | --- | --- |
| Scalar only | float math | All 15 guards |
| Vector math | vec.h | 14 guards (not WP_NO_VEC) |
| Mat + quat | mat.h, quat.h, vec.h | 12 guards |
| Noise + random | noise.h, rand.h, vec.h | 12 guards |
| Volume sampling | volume.h, vec.h, mat.h | 11 guards |
| Mesh queries | mesh.h, bvh.h, tile.h, intersect.h, vec.h, mat.h | 8 guards |

### Newton Example Measurement

Newton examples are run as subprocesses via `python -m newton.examples <name>
--viewer null --num-frames 1 --device <device>`. Two metrics are captured:

1. **Wall time**: `time.perf_counter()` around the subprocess call (includes
   Python startup, imports, compilation, and 1 simulation frame)
2. **Compile time**: Sum of all `took N ms (compiled)` lines from Warp output

## Lessons Learned

### CRITICAL: Never Run Benchmarks in Parallel

During this session, running two benchmark scripts simultaneously (one for
`../newton` and one for `../newton-benchmark-expt`) caused:

1. **Kernel cache corruption**: Both processes call `wp.clear_kernel_cache()`
   and write to the same default cache directory (`~/.cache/warp/`). Concurrent
   writes caused `LLVM ERROR: IO failure on output stream: Bad file descriptor`.

2. **CPU contention**: Both Clang and NVRTC compilations are CPU-intensive.
   Parallel runs inflate times unpredictably.

**All benchmarks must run sequentially.** The robot example data was collected
sequentially after discovering this issue.

### Data Reliability Notes

All data in `_mr_benchmark_data.md` was collected sequentially with the
following exceptions:

- **No flagged data**: All final results were collected in sequential runs.
  Earlier parallel runs that produced errors were discarded and re-collected.

### Newton Example Caveats

- Examples include Python import overhead (~2–5s) in wall time. The "compile
  time" column isolates JIT time by parsing Warp's output.
- Some examples have high wall-time stdev (e.g., `robot_panda_hydro` at 20.5s)
  due to variable NVRTC warm-up and complex initialization.
- `basic_pendulum` errored on the experiment branch during one run but not
  others — likely a transient subprocess issue, not a Warp bug.
- `basic_shapes` and `basic_joints` errored on main during the first run but
  succeeded in the second — same transient issue.

### Bugs Found During Benchmarking

1. **`eig3`/`qr3` wrong compile guard**: These builtins were annotated
   `WP_NO_MAT` but live in `svd.h` (guarded by `WP_NO_SVD`). Newton's inertia
   validation kernel calls `eig3`, causing CUDA compilation failure. Fixed in
   commit `72d845a1`.

2. **FEM custom matrix types**: FEM geometry creates `mat_t<27,3>` inside
   `@wp.func` bodies. No builtin annotation covers this. Fixed by adding
   safety-net source scan (`scan_source_for_guards`).

3. **CUDA `tile_shared_storage_t`**: The CUDA kernel templates unconditionally
   referenced this type, but it's defined in `tile.h` which is excluded by
   `WP_NO_TILE`. Fixed by wrapping with `#ifndef WP_NO_TILE`.

## Future Work

- **More Warp examples**: Benchmark `warp/examples/` (core, optim, fem) in the
  same manner. Most are under 2s compile so the improvement will be larger
  percentage-wise.
- **CPU Newton examples**: Only CUDA was measured for Newton. CPU compilation
  is slower per-module, so the aggregate improvement should be larger.
- **Per-module breakdown for robot examples**: The robot examples compile many
  MuJoCo Warp modules. A per-module breakdown would show which modules benefit
  most.
- **Warm-compile (cache hit) verification**: Verify that cached modules are
  not affected by the compile guard changes (they shouldn't be — guards only
  affect the generated source, which is hashed for caching).
