#!/usr/bin/env python3
"""Run all compile-time benchmarks sequentially on baseline and branch.

Produces JSON output for comparison. Runs each benchmark one at a time
to avoid measurement interference.
"""

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

BASELINE_DIR = "/home/horde/code-projects/warp"
BRANCH_DIR = "/home/horde/code-projects/warp-worktree-3"

# Isolated cache dirs to avoid interference
CACHE_BASELINE = "/tmp/bench_cache_baseline"
CACHE_BRANCH = "/tmp/bench_cache_branch"

RESULTS_FILE = "/tmp/benchmark_results.json"

# --- Kernel definitions (same as _bench_comparison.py) ---
KERNELS = {
    "scalar_only": '''
import warp as wp

@wp.kernel
def kernel(x: wp.array2d(dtype=float), y: wp.array2d(dtype=float)):
    i, j = wp.tid()
    x[i, j] += y[i, j]
''',
    "vec_math": '''
import warp as wp

@wp.kernel
def kernel(
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    result: wp.array(dtype=float),
):
    tid = wp.tid()
    p = points[tid]
    n = normals[tid]
    result[tid] = wp.dot(p, n) / (wp.length(p) + 1e-6)
''',
    "mat_quat_transform": '''
import warp as wp

@wp.kernel
def kernel(
    positions: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.quatf),
    transforms: wp.array(dtype=wp.mat44),
    output: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    p = positions[tid]
    q = rotations[tid]
    m = transforms[tid]
    rotated = wp.quat_rotate(q, p)
    output[tid] = wp.transform_point(m, rotated)
''',
    "volume_sample": '''
import warp as wp

@wp.kernel
def kernel(
    volume: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    output: wp.array(dtype=float),
):
    tid = wp.tid()
    p = points[tid]
    output[tid] = wp.volume_sample_f(volume, p, wp.Volume.LINEAR)
''',
    "mesh_query": '''
import warp as wp

@wp.kernel
def kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=float),
):
    tid = wp.tid()
    p = points[tid]
    sign = float(0.0)
    f = int(0)
    u = float(0.0)
    v = float(0.0)
    has_point = wp.mesh_query_point(mesh_id, p, 1000.0, sign, f, u, v)
    if has_point:
        closest = wp.mesh_eval_position(mesh_id, f, u, v)
        distances[tid] = wp.length(p - closest) * sign
''',
}

DISPLAY_NAMES = {
    "scalar_only": "Scalar only (trivial)",
    "vec_math": "Vector math",
    "mat_quat_transform": "Mat + quat + transform",
    "volume_sample": "Volume sampling",
    "mesh_query": "Mesh queries",
}

# FEM examples
FEM_EXAMPLES = [
    ("fem.navier_stokes", "fem.example_navier_stokes", ["--num-frames", "1", "--resolution", "10", "--tri-mesh", "--headless"]),
    ("fem.stokes", "fem.example_stokes", ["--resolution", "10", "--nonconforming-pressures", "--headless"]),
    ("fem.deformed_geometry", "fem.example_deformed_geometry", ["--resolution", "10", "--mesh", "tri", "--headless"]),
    ("fem.diffusion_3d", "fem.example_diffusion_3d", ["--headless"]),
    ("fem.convection_diffusion", "fem.example_convection_diffusion", ["--resolution", "20", "--headless"]),
]


def run_kernel_benchmark(warp_dir, cache_dir, device, samples=5):
    """Run isolated kernel benchmarks. Returns {kernel_name: median_seconds}."""
    # Write kernel files to a temp dir
    kern_src_dir = os.path.join(cache_dir, "kern_src")
    os.makedirs(kern_src_dir, exist_ok=True)
    with open(os.path.join(kern_src_dir, "__init__.py"), "w") as f:
        f.write("")
    for name, source in KERNELS.items():
        with open(os.path.join(kern_src_dir, f"{name}.py"), "w") as f:
            f.write(source)

    # Worker script that imports warp from the right dir and benchmarks one kernel
    worker = f'''
import os, sys, time, statistics
os.environ["CUDA_CACHE_DISABLE"] = "1"
sys.path.insert(0, "{warp_dir}")
sys.path.insert(0, "{cache_dir}")

import warp as wp
wp.config.kernel_cache_dir = "{cache_dir}/wp_cache"
wp.config.quiet = True
wp.init()
wp.clear_kernel_cache()

import importlib
kernel_name = sys.argv[1]
device = sys.argv[2]
num_samples = int(sys.argv[3])

mod = importlib.import_module(f"kern_src.{{kernel_name}}")
warp_mod = None
for attr_name in dir(mod):
    obj = getattr(mod, attr_name)
    if isinstance(obj, wp.Kernel):
        warp_mod = obj.module
        break

# Warm up
warp_mod.load(device)

timings = []
for i in range(num_samples):
    wp.clear_kernel_cache()
    warp_mod.unload()
    start = time.perf_counter()
    warp_mod.load(device)
    elapsed = time.perf_counter() - start
    timings.append(elapsed)

med = statistics.median(timings)
mean = statistics.mean(timings)
std = statistics.stdev(timings) if len(timings) > 1 else 0.0
cv = (std / mean * 100) if mean > 0 else 0.0
print(f"RESULT {{kernel_name}} {{med:.4f}} {{mean:.4f}} {{std:.4f}} {{cv:.1f}}")
'''

    results = {}
    for name in KERNELS:
        # Clean cache between kernels
        wp_cache = os.path.join(cache_dir, "wp_cache")
        if os.path.isdir(wp_cache):
            shutil.rmtree(wp_cache)

        print(f"  {DISPLAY_NAMES[name]}...", flush=True)
        result = subprocess.run(
            [sys.executable, "-c", worker, name, device, str(samples)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"    ERROR: {result.stderr[-200:]}", flush=True)
            results[name] = {"median": -1, "mean": -1, "stdev": -1, "cv": -1}
            continue

        for line in result.stdout.splitlines():
            if line.startswith("RESULT"):
                parts = line.split()
                results[name] = {
                    "median": float(parts[2]),
                    "mean": float(parts[3]),
                    "stdev": float(parts[4]),
                    "cv": float(parts[5]),
                }
                print(f"    median={parts[2]}s cv={parts[5]}%", flush=True)

    return results


def run_fem_benchmark(warp_dir, cache_dir, device, samples=3):
    """Run FEM example benchmarks. Returns {example_name: median_compile_seconds}."""
    results = {}
    python = sys.executable

    for display, mod_name, extra in FEM_EXAMPLES:
        print(f"  {display}...", flush=True)

        timings = []
        skip = False
        for s in range(samples):
            # Clear cache
            wp_cache = os.path.join(cache_dir, "wp_cache")
            if os.path.isdir(wp_cache):
                shutil.rmtree(wp_cache)

            env = os.environ.copy()
            env["CUDA_CACHE_DISABLE"] = "1"
            env["PYTHONPATH"] = warp_dir
            env["WARP_CACHE_ROOT"] = wp_cache

            cmd = [python, "-m", f"warp.examples.{mod_name}", "--device", device] + extra
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300, cwd=warp_dir)
            except subprocess.TimeoutExpired:
                print(f"    TIMEOUT on sample {s+1}", flush=True)
                skip = True
                break

            total_compile = 0.0
            for m in re.finditer(r"took ([\d.]+) ms\s+\(compiled\)", result.stdout + result.stderr):
                total_compile += float(m.group(1)) / 1000.0

            if result.returncode != 0 and total_compile == 0:
                stderr_lines = result.stderr.strip().split("\n")
                for line in stderr_lines[-3:]:
                    print(f"    {line}", flush=True)
                skip = True
                break

            timings.append(total_compile)
            print(f"    sample {s+1}/{samples}: {total_compile:.1f}s", flush=True)

        if skip or not timings:
            results[display] = {"median": -1, "stdev": -1}
            continue

        med = statistics.median(timings)
        std = statistics.stdev(timings) if len(timings) > 1 else 0.0
        results[display] = {"median": med, "stdev": std}
        print(f"    median={med:.1f}s", flush=True)

    return results


def main():
    all_results = {}

    for label, warp_dir, cache_dir in [
        ("baseline", BASELINE_DIR, CACHE_BASELINE),
        ("branch", BRANCH_DIR, CACHE_BRANCH),
    ]:
        print(f"\n{'='*60}", flush=True)
        print(f"Running benchmarks on: {label} ({warp_dir})", flush=True)
        print(f"{'='*60}", flush=True)

        # Clean cache dir
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        all_results[label] = {}

        for device in ["cpu", "cuda:0"]:
            print(f"\n--- Isolated kernels ({device}) ---", flush=True)
            all_results[label][f"kernels_{device}"] = run_kernel_benchmark(
                warp_dir, cache_dir, device, samples=5
            )

            print(f"\n--- FEM examples ({device}) ---", flush=True)
            all_results[label][f"fem_{device}"] = run_fem_benchmark(
                warp_dir, cache_dir, device, samples=3
            )

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)

    # Print comparison
    print(f"\n{'='*80}", flush=True)
    print("COMPARISON", flush=True)
    print(f"{'='*80}", flush=True)

    for device in ["cpu", "cuda:0"]:
        key = f"kernels_{device}"
        print(f"\n### Isolated Kernels ({device})", flush=True)
        print(f"{'Kernel':<30} {'Main':>8} {'Branch':>8} {'Speedup':>8} {'Delta':>8}", flush=True)
        print("-" * 70, flush=True)

        baseline = all_results["baseline"].get(key, {})
        branch = all_results["branch"].get(key, {})
        for name in KERNELS:
            b = baseline.get(name, {}).get("median", -1)
            br = branch.get(name, {}).get("median", -1)
            if b > 0 and br > 0:
                speedup = b / br
                delta_pct = (br - b) / b * 100
                flag = " *** REGRESSION" if delta_pct > 5 else ""
                print(f"{DISPLAY_NAMES[name]:<30} {b:>7.3f}s {br:>7.3f}s {speedup:>7.2f}x {delta_pct:>+7.1f}%{flag}", flush=True)
            else:
                print(f"{DISPLAY_NAMES[name]:<30} {'ERR':>8} {'ERR':>8}", flush=True)

    for device in ["cpu", "cuda:0"]:
        key = f"fem_{device}"
        print(f"\n### FEM Examples ({device})", flush=True)
        print(f"{'Example':<30} {'Main':>8} {'Branch':>8} {'Speedup':>8} {'Delta':>8}", flush=True)
        print("-" * 70, flush=True)

        baseline = all_results["baseline"].get(key, {})
        branch = all_results["branch"].get(key, {})
        for display, _, _ in FEM_EXAMPLES:
            b = baseline.get(display, {}).get("median", -1)
            br = branch.get(display, {}).get("median", -1)
            if b > 0 and br > 0:
                speedup = b / br
                delta_pct = (br - b) / b * 100
                flag = " *** REGRESSION" if delta_pct > 5 else ""
                print(f"{display:<30} {b:>7.1f}s {br:>7.1f}s {speedup:>7.2f}x {delta_pct:>+7.1f}%{flag}", flush=True)
            else:
                print(f"{display:<30} {'ERR/N/A':>8} {'ERR/N/A':>8}", flush=True)


if __name__ == "__main__":
    main()
