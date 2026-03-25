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
for true cold compiles.  **Always use subprocess-per-sample isolation**
— do not collect multiple samples in a single process, even for isolated
kernels.  In-process `wp.clear_kernel_cache()` + `module.unload()` does
not reset all state:

- **PCH reuse**: Precompiled headers are created once per process.
  Subsequent samples reuse the PCH file, making CUDA times artificially
  low.  This disproportionately benefits the baseline (more header
  content) and inflates apparent speedup ratios.
- **FEM dynamic modules**: `fem.integrate()` generates dynamic modules
  that persist across in-process cache clears.

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

## Session Survival for Long Runs

A full recollection (baseline + branch, FEM + Newton, CPU + CUDA) takes
several hours.  Background processes launched from a CLI session die when
the session drops.  Use **tmux** to keep runs alive:

```bash
# Launch in a detached tmux session:
tmux new-session -d -s bench "cd /path/to/warp-worktree-2 && ./_run_remaining_benchmarks.sh"

# Monitor:
tmux list-sessions
tail -f _bench_full_run.log

# Check completion:
cat _bench_run_status.txt   # contains "DONE" when finished
```

**Do not use `nohup`/`setsid`/`disown`** — these are unreliable for
fully detaching from a parent shell; child processes may still be killed
on session teardown.  `tmux` (or `screen`) creates an independent
session that survives disconnects.

## Process Isolation

**Never run two benchmark processes concurrently**, even accidentally.
They compete for CPU/GPU and corrupt each other's measurements.  Before
starting a new run, always verify no stale processes remain:

```bash
pgrep -af "_bench_comprehensive|_run_remaining" || echo "all clean"
```

If a run is interrupted mid-flight (e.g., by killing a duplicate), the
first sample of the next example may fail with an LLVM error due to
residual state.  The script marks that example as failed and continues.
Re-run just that example afterward if the data point is needed.

## Env Var Reference

| Variable | Purpose | Notes |
| --- | --- | --- |
| `WARP_CACHE_PATH` | Redirect kernel cache to a temp dir | Per-subprocess; wiped between samples |
| `WARP_CACHE_ROOT` | Only used by `unittest_parallel` worker processes | **Do not use** for benchmarks |
| `CUDA_CACHE_DISABLE=1` | Prevent NVIDIA driver from caching PTX/cubin | Must be set or second sample hits driver cache |
| `PYTHONPATH` | Override Warp import for Newton branch runs | Takes precedence over editable install |

## Baseline Sensitivity to Upstream Changes

When recollecting data, the **baseline** (main branch) numbers can shift
significantly if upstream changes affect compilation.  For example, the
`ershi/fix-cu12-pch` fix (merged 2026-03-24) reduced CUDA baseline times
by ~40–50% for FEM examples.  Always record the main branch commit hash
alongside benchmark data so shifts can be attributed.
