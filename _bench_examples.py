#!/usr/bin/env python3
"""Benchmark compile times for Warp examples.

Runs each example via `python -m warp.examples.<name>` with headless args
from test_examples.py. Clears kernel cache between runs, parses compile
times from output.

Usage:
    uv run _bench_examples.py --device cuda:0 --samples 3
    uv run _bench_examples.py --device cpu --samples 3
"""

import argparse
import os
import re
import statistics
import subprocess
import sys

# (display_name, module_name, extra_args)
# Args taken from test_examples.py commented-out FEM tests + active tests
EXAMPLES = [
    # FEM — use headless + reduced resolution for fast sim, heavy compile
    ("fem.apic_fluid", "fem.example_apic_fluid", ["--num-frames", "1", "--voxel-size", "2.0"]),
    ("fem.diffusion_3d", "fem.example_diffusion_3d", ["--headless"]),
    ("fem.navier_stokes", "fem.example_navier_stokes", ["--num-frames", "1", "--resolution", "10", "--tri-mesh", "--headless"]),
    ("fem.stokes", "fem.example_stokes", ["--resolution", "10", "--nonconforming-pressures", "--headless"]),
    ("fem.deformed_geometry", "fem.example_deformed_geometry", ["--resolution", "10", "--mesh", "tri", "--headless"]),
    ("fem.convection_diffusion", "fem.example_convection_diffusion", ["--resolution", "20", "--headless"]),
    ("fem.mixed_elasticity", "fem.example_mixed_elasticity", ["--nonconforming-stresses", "--mesh", "quad", "--headless"]),
    # Core
    ("core.fluid", "core.example_fluid", ["--num-frames", "1", "--headless"]),
    ("core.sph", "core.example_sph", ["--num-frames", "1"]),
    ("core.dem", "core.example_dem", ["--num-frames", "1"]),
    ("core.raymarch", "core.example_raymarch", ["--height", "512", "--width", "1024", "--headless"]),
    ("core.raycast", "core.example_raycast", ["--headless"]),
    ("core.wave", "core.example_wave", ["--num-frames", "1"]),
    # Optim
    ("optim.diffray", "optim.example_diffray", ["--headless", "--train-iters", "1"]),
    ("optim.particle_repulsion", "optim.example_particle_repulsion", ["--headless", "--num-frames", "1"]),
]


def run_example(python, module_name, device, extra_args):
    env = os.environ.copy()
    env["CUDA_CACHE_DISABLE"] = "1"

    # Clear kernel cache
    subprocess.run(
        [python, "-c", "import warp as wp; wp.init(); wp.clear_kernel_cache()"],
        capture_output=True, env=env, timeout=30,
    )

    cmd = [python, "-m", f"warp.examples.{module_name}", "--device", device] + extra_args

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    except subprocess.TimeoutExpired:
        return -1.0

    # Sum compile times from "took N ms (compiled)" lines
    total = 0.0
    for m in re.finditer(r"took ([\d.]+) ms\s+\(compiled\)", result.stdout + result.stderr):
        total += float(m.group(1)) / 1000.0

    if result.returncode != 0 and total == 0:
        # Print last error line for debugging
        stderr_lines = result.stderr.strip().split("\n")
        for line in stderr_lines[-3:]:
            print(f"    {line}", file=sys.stderr, flush=True)
        return -1.0

    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    python = sys.executable

    import warp as wp
    wp.config.quiet = True
    wp.init()

    print(f"device: {args.device}, samples: {args.samples}", flush=True)
    print(f"warp: {wp.__version__}, path: {wp.__file__}", flush=True)
    print(flush=True)
    print(f"{'Example':<30} {'Median':>8} {'Stdev':>7}", flush=True)
    print("-" * 50, flush=True)

    for display, mod_name, extra in EXAMPLES:
        timings = []
        skip = False
        for _ in range(args.samples):
            t = run_example(python, mod_name, args.device, extra)
            if t < 0:
                print(f"{display:<30} {'ERROR/TIMEOUT':>15}", flush=True)
                skip = True
                break
            timings.append(t)

        if skip:
            continue

        med = statistics.median(timings)
        std = statistics.stdev(timings) if len(timings) > 1 else 0.0
        print(f"{display:<30} {med:>7.1f}s {std:>6.1f}s", flush=True)


if __name__ == "__main__":
    main()
