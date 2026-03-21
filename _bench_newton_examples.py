#!/usr/bin/env python3
"""Benchmark Newton example end-to-end startup times.

Usage (from a Newton worktree with the right .venv):
    .venv/bin/python /path/to/_bench_newton_examples.py --device cuda:0 --samples 3
"""

import argparse
import os
import re
import statistics
import subprocess
import sys
import time


EXAMPLES = [
    "basic_shapes",
    "basic_joints",
    "cloth_hanging",
    "diffsim_ball",
    "robot_cartpole",
    "softbody_hanging",
    "mpm_granular",
]


def run_example(python: str, example: str, device: str) -> tuple[float, float]:
    """Run a Newton example, return (wall_time, total_compile_time)."""
    env = os.environ.copy()
    env["CUDA_CACHE_DISABLE"] = "1"

    # Clear warp kernel cache before each run
    clear_script = "import warp as wp; wp.init(); wp.clear_kernel_cache()"
    subprocess.run([python, "-c", clear_script], capture_output=True, env=env, timeout=30)

    cmd = [
        python, "-m", "newton.examples", example,
        "--viewer", "null",
        "--num-frames", "1",
        "--device", device,
    ]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    wall = time.perf_counter() - start

    # Sum up compile times from output
    pattern = re.compile(r"took ([\d.]+) ms\s+\(compiled\)")
    compile_total = 0.0
    for m in pattern.finditer(result.stdout + result.stderr):
        compile_total += float(m.group(1)) / 1000.0

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "Error" in stderr or "Traceback" in stderr:
            # Print last few lines for debugging
            for line in stderr.split("\n")[-3:]:
                print(f"    {line}", file=sys.stderr)
            return wall, -1.0

    return wall, compile_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    python = sys.executable

    import warp as wp
    wp.init()
    print(f"device: {args.device}, samples: {args.samples}")
    print(f"warp: {wp.__version__}, path: {wp.__file__}")
    print()

    print(f"{'Example':<25} {'Wall (med)':>10} {'Compile (med)':>13} {'Wall stdev':>10}")
    print("-" * 63)

    for ex in EXAMPLES:
        walls = []
        compiles = []
        skip = False
        for s in range(args.samples):
            w, c = run_example(python, ex, args.device)
            if c < 0:
                print(f"{ex:<25} {'ERROR':>10}")
                skip = True
                break
            walls.append(w)
            compiles.append(c)

        if skip:
            continue

        wmed = statistics.median(walls)
        cmed = statistics.median(compiles)
        wstd = statistics.stdev(walls) if len(walls) > 1 else 0.0
        print(f"{ex:<25} {wmed:>9.1f}s {cmed:>12.1f}s {wstd:>9.1f}s")


if __name__ == "__main__":
    main()
