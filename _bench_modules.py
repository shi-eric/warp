#!/usr/bin/env python3
"""Benchmark compile times for specific Warp modules.

Compiles each module cold (cache cleared) multiple times and reports stats.
Run from the repo root of the branch you want to measure.

Usage:
    uv run _bench_modules.py --device cpu --samples 3
    uv run _bench_modules.py --device cuda:0 --samples 3
"""

import argparse
import importlib
import os
import statistics
import time

os.environ.setdefault("CUDA_CACHE_DISABLE", "1")

import warp as wp


def cold_compile_module(module_name: str, device: str) -> float:
    """Clear cache, unload, recompile a module. Return seconds."""
    mod = importlib.import_module(module_name)

    # Find the warp Module object(s) for this Python module
    warp_modules = []
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, wp.context.Kernel):
            if obj.module not in warp_modules:
                warp_modules.append(obj.module)

    if not warp_modules:
        # Try loading the module by name directly
        warp_mod = wp.context.user_modules.get(module_name)
        if warp_mod:
            warp_modules = [warp_mod]

    if not warp_modules:
        raise RuntimeError(f"No Warp kernels found in {module_name}")

    # Unload all modules
    for m in warp_modules:
        m.unload()

    wp.clear_kernel_cache()

    start = time.perf_counter()
    for m in warp_modules:
        m.load(device)
    elapsed = time.perf_counter() - start
    return elapsed


# Modules to benchmark.  Each entry is (display_name, python_module_name).
MODULES = [
    # FEM geometry — real shipped code, creates custom matrix types
    ("FEM hexmesh", "warp.fem.geometry.hexmesh"),
    ("FEM tetmesh", "warp.fem.geometry.tetmesh"),
    # Examples — representative user workloads
    ("example: SPH fluid", "warp.examples.core.example_sph"),
    ("example: mesh deformation", "warp.examples.core.example_mesh_deform"),
    ("example: diffray (optim)", "warp.examples.optim.example_diffray"),
    # Standalone trivial kernel — best-case for guards
    ("trivial kernel (bench_compile_time)", "__main__"),
]


@wp.kernel
def _trivial_kernel(
    x: wp.array2d(dtype=float),
    y: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    x[i, j] += y[i, j]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    cache_dir = os.path.join(os.getcwd(), "bench_kernels")
    wp.config.kernel_cache_dir = cache_dir
    wp.init()
    wp.clear_kernel_cache()

    # Warm up — compile everything once
    for display_name, mod_name in MODULES:
        if mod_name == "__main__":
            wp.load_module(device=args.device)
        else:
            try:
                importlib.import_module(mod_name)
            except Exception as e:
                print(f"SKIP {display_name}: {e}")

    print(f"\ndevice: {args.device}, samples: {args.samples}\n")
    print(f"{'Module':<40} {'Median':>8} {'Mean':>8} {'Stdev':>8} {'Min':>8} {'Max':>8}")
    print("-" * 90)

    for display_name, mod_name in MODULES:
        if mod_name == "__main__":
            # Use the trivial kernel approach
            timings = []
            for _ in range(args.samples):
                wp.clear_kernel_cache()
                _trivial_kernel.module.unload()
                start = time.perf_counter()
                wp.load_module(device=args.device)
                timings.append(time.perf_counter() - start)
        else:
            try:
                timings = []
                for _ in range(args.samples):
                    t = cold_compile_module(mod_name, args.device)
                    timings.append(t)
            except Exception as e:
                print(f"{display_name:<40} ERROR: {e}")
                continue

        med = statistics.median(timings)
        mean = statistics.mean(timings)
        std = statistics.stdev(timings) if len(timings) > 1 else 0.0
        print(f"{display_name:<40} {med:>7.2f}s {mean:>7.2f}s {std:>7.2f}s {min(timings):>7.2f}s {max(timings):>7.2f}s")


if __name__ == "__main__":
    main()
