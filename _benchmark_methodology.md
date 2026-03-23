# Compile Guard Benchmark Methodology

How to reproduce and extend the compile-time benchmarks in this branch.

## Environment

- NVIDIA L40 GPU, CUDA Toolkit 12.8, driver 570.158.01
- Warp installed as editable (`uv pip install -e .`) in each repo's `.venv`
- Newton's `.venv` has an editable install of Warp pointing at `../warp`

## Key Pitfalls

### Newton `--quiet` suppresses compile-time output

Newton's `--quiet` flag sets `wp.config.quiet = True`, which suppresses
Warp's `took N ms (compiled)` log lines.  **Do not use `--quiet`** when
measuring compile times — the benchmark scripts parse those lines to sum
total compile time.

### Editable installs and PYTHONPATH

Each repo (warp, warp-worktree-3, newton) has its own `.venv` with an
editable Warp install.  The editable install resolves based on **cwd**,
not the venv path:

```
# From ../warp → imports ../warp/warp
cd /path/to/warp && .venv/bin/python3 -c "import warp; print(warp.__file__)"

# From ../warp-worktree-3 → imports ../warp-worktree-3/warp
cd /path/to/warp-worktree-3 && .venv/bin/python3 -c "import warp; print(warp.__file__)"
```

For Newton benchmarks on the **branch**, override the import with:
```
PYTHONPATH=/path/to/warp-worktree-3 python3 -m newton.examples ...
```
PYTHONPATH takes precedence over the editable install.  Always verify
with `import warp; print(warp.__file__)` before trusting results.

### Cache isolation

Use `WARP_CACHE_PATH` (not `WARP_CACHE_ROOT`) to redirect the kernel
cache.  Wipe the cache directory with `shutil.rmtree()` between samples
for true cold compiles.  `wp.clear_kernel_cache()` is **not sufficient**
for FEM/Newton benchmarks because they generate dynamic modules during
`fem.integrate()` that persist across in-process cache clears.

### CUDA driver cache

Set `CUDA_CACHE_DISABLE=1` in the environment to prevent the NVIDIA
driver from caching compiled PTX/cubin across runs.  Without this,
second and subsequent samples may hit the driver cache and report
artificially low times.

### Precompiled headers (PCH)

`warp.config.use_precompiled_headers` defaults to `True`.  PCH can
actually **slow down** NVRTC compilation for modules that use few
features, because NVRTC must deserialize the full precompiled header
even when most of it is unused.  Benchmarks in this branch use the
default (PCH on) unless explicitly testing PCH impact.

To test PCH off, set `wp.config.use_precompiled_headers = False` before
any module compilation.  This must be done before `wp.init()` or in a
subprocess wrapper.

## Running Benchmarks

### Isolated kernels (CPU + CUDA)

```bash
# From the branch worktree:
uv run bench_compile_time.py --device cpu --samples 5
uv run bench_compile_time.py --device cuda:0 --samples 5
```

### FEM examples

Use `_bench_comprehensive.py` which handles cache wipes and subprocess
isolation:

```bash
# Baseline (uses ../warp):
python3 _bench_comprehensive.py --label baseline --suite fem --samples 3

# Branch (uses ../warp-worktree-3):
python3 _bench_comprehensive.py --label branch --suite fem --samples 3
```

### Newton examples

```bash
# Baseline:
python3 _bench_comprehensive.py --label baseline --suite newton --samples 3

# Branch:
python3 _bench_comprehensive.py --label branch --suite newton --samples 3
```

The `--label` flag controls which Warp is used: `baseline` uses
`../warp` (main), `branch` uses `../warp-worktree-3` and sets
`PYTHONPATH` for Newton.

### Comparing results

```bash
python3 _bench_compare.py _benchmark_baseline_all.json _benchmark_branch_all.json -o comparison.md
```

### Generating charts

```bash
uv run --with matplotlib _plot_compile_times.py
```

Reads from `_benchmark_baseline_all.json` and `_benchmark_branch_all.json`.

## Measurement Quality

- **Samples**: 3 minimum for FEM/Newton (each takes 10–170s), 5 for
  isolated kernels (each takes < 2s)
- **CV threshold**: Coefficient of variation above 10% indicates noisy
  measurements — increase samples or reduce system load
- **Sequential execution**: Never run benchmarks in parallel — they
  compete for CPU/GPU and inflate variance
- **Regression threshold**: >5% slowdown AND exceeding 2σ combined noise
  is flagged as a regression by `_bench_compare.py`
