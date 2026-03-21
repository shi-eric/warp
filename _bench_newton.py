#!/usr/bin/env python3
"""Benchmark Newton module compile times.

Imports Newton, clears cache, compiles all modules, parses compile times from output.

Usage (from a Newton worktree with the right .venv):
    .venv/bin/python /path/to/_bench_newton.py --device cpu --samples 3
    .venv/bin/python /path/to/_bench_newton.py --device cuda:0 --samples 3
"""

import argparse
import os
import re
import statistics
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_newton_compile(python: str, device: str) -> dict[str, float]:
    """Run a subprocess that imports Newton, clears cache, compiles, returns module times."""
    script = f"""
import os
os.environ.setdefault("CUDA_CACHE_DISABLE", "1")
import warp as wp
wp.init()
wp.clear_kernel_cache()

import newton
import newton.solvers
import newton.geometry

# Unload all newton modules
from warp._src.context import user_modules
for name, mod in list(user_modules.items()):
    if "newton" in name:
        mod.unload()

# Recompile
wp.load_module(module="newton", recursive=True, device="{device}")
"""

    env = os.environ.copy()
    env["CUDA_CACHE_DISABLE"] = "1"

    result = subprocess.run(
        [python, "-c", script],
        capture_output=True, text=True, env=env, timeout=600,
    )

    # Parse "Module <name> <hash> load on device '<dev>' took <N> ms  (compiled)"
    times = {}
    pattern = re.compile(r"Module (\S+) \S+ load on device '\S+' took ([\d.]+) ms\s+\(compiled\)")
    for line in (result.stdout + result.stderr).split("\n"):
        m = pattern.search(line)
        if m and "newton" in m.group(1):
            times[m.group(1)] = float(m.group(2)) / 1000.0

    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    python = sys.executable

    # Verify warp version
    import warp as wp
    wp.init()
    print(f"device: {args.device}, samples: {args.samples}")
    print(f"warp: {wp.__version__}, path: {wp.__file__}")
    print()

    # Collect samples
    all_times: dict[str, list[float]] = {}
    for s in range(args.samples):
        print(f"  sample {s + 1}/{args.samples}...", flush=True)
        times = run_newton_compile(python, args.device)
        for name, t in times.items():
            all_times.setdefault(name, []).append(t)

    # Report
    results = []
    for name, timings in all_times.items():
        med = statistics.median(timings)
        results.append((name, med, timings))

    results.sort(key=lambda x: -x[1])
    total_median = sum(r[1] for r in results)

    print()
    print(f"{'Module':<55} {'Median':>8} {'Stdev':>7}")
    print("-" * 75)
    for name, med, timings in results[:20]:
        std = statistics.stdev(timings) if len(timings) > 1 else 0.0
        print(f"{name:<55} {med:>7.3f}s {std:>6.3f}s")
    if len(results) > 20:
        rest_med = sum(r[1] for r in results[20:])
        print(f"{'... ({} more modules)'.format(len(results) - 20):<55} {rest_med:>7.3f}s")
    print("-" * 75)
    print(f"{'TOTAL (all newton modules)':<55} {total_median:>7.3f}s")


if __name__ == "__main__":
    main()
