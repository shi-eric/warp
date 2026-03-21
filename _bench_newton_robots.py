#!/usr/bin/env python3
"""Benchmark Newton robot examples end-to-end."""

import argparse
import os
import re
import statistics
import subprocess
import sys
import time

EXAMPLES = [
    "robot_g1",
    "robot_h1",
    "robot_anymal_d",
    "robot_ur10",
    "robot_allegro_hand",
    "robot_panda_hydro",
]


def run_example(python, example, device):
    env = os.environ.copy()
    env["CUDA_CACHE_DISABLE"] = "1"
    subprocess.run([python, "-c", "import warp as wp; wp.init(); wp.clear_kernel_cache()"],
                   capture_output=True, env=env, timeout=30)

    cmd = [python, "-m", "newton.examples", example,
           "--viewer", "null", "--num-frames", "1", "--device", device]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    wall = time.perf_counter() - start

    compile_total = 0.0
    for m in re.finditer(r"took ([\d.]+) ms\s+\(compiled\)", result.stdout + result.stderr):
        compile_total += float(m.group(1)) / 1000.0

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "Error" in stderr or "Traceback" in stderr:
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
    print(f"{'Example':<30} {'Wall (med)':>10} {'Compile (med)':>13} {'Wall stdev':>10}")
    print("-" * 68)

    for ex in EXAMPLES:
        walls, compiles = [], []
        skip = False
        for _ in range(args.samples):
            w, c = run_example(python, ex, args.device)
            if c < 0:
                print(f"{ex:<30} {'ERROR':>10}")
                skip = True
                break
            walls.append(w)
            compiles.append(c)
        if skip:
            continue
        wmed = statistics.median(walls)
        cmed = statistics.median(compiles)
        wstd = statistics.stdev(walls) if len(walls) > 1 else 0.0
        print(f"{ex:<30} {wmed:>9.1f}s {cmed:>12.1f}s {wstd:>9.1f}s")


if __name__ == "__main__":
    main()
