#!/usr/bin/env python3
"""Run isolated kernel compile-time benchmarks for baseline and branch.

Each sample spawns a fresh subprocess with a wiped cache directory,
ensuring true cold-compile (no PCH reuse, no residual state).
"""

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

BASELINE_DIR = "/home/horde/code-projects/warp"
BRANCH_DIR = "/home/horde/code-projects/warp-worktree-3"

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

NUM_SAMPLES = 5

# Worker script template — each invocation is a fresh process
WORKER_TEMPLATE = '''
import os, sys, time
os.environ["CUDA_CACHE_DISABLE"] = "1"

import warp as wp
wp.config.kernel_cache_dir = sys.argv[2]
wp.config.quiet = True
wp.init()

device = sys.argv[1]

# The kernel source is in the file we're running — import it
# Actually we import the kernel module from the path given
import importlib.util
spec = importlib.util.spec_from_file_location("bench_kernel", sys.argv[3])
mod = spec.loader.load_module()

warp_mod = None
for attr_name in dir(mod):
    obj = getattr(mod, attr_name)
    if isinstance(obj, wp.Kernel):
        warp_mod = obj.module
        break

start = time.perf_counter()
warp_mod.load(device)
elapsed = time.perf_counter() - start
print(f"ELAPSED {elapsed:.6f}")
'''


def run_one_sample(python, warp_dir, kernel_file, device, cache_dir):
    """Run one cold-compile sample in a fresh subprocess. Returns seconds or None."""
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_CACHE_DISABLE"] = "1"
    env["WARP_CACHE_PATH"] = cache_dir
    env.pop("VIRTUAL_ENV", None)

    worker_file = os.path.join(cache_dir, "_worker.py")
    with open(worker_file, "w") as f:
        f.write(WORKER_TEMPLATE)

    try:
        r = subprocess.run(
            [python, worker_file, device, cache_dir, kernel_file],
            capture_output=True, text=True, timeout=120,
            cwd=warp_dir, env=env,
        )
    except subprocess.TimeoutExpired:
        return None

    if r.returncode != 0:
        print(f"    ERROR: {r.stderr[-300:]}", flush=True)
        return None

    for line in r.stdout.splitlines():
        if line.startswith("ELAPSED"):
            return float(line.split()[1])
    return None


def run_kernel_benchmarks(warp_dir, device):
    """Run all kernels for one branch+device combo. Returns {name: {median, samples, cv}}."""
    python = os.path.join(warp_dir, ".venv", "bin", "python3")
    if not os.path.exists(python):
        print(f"WARNING: {python} not found, using system python", flush=True)
        python = sys.executable

    # Write kernel source files to a stable temp location
    kern_dir = tempfile.mkdtemp(prefix="bench_kern_src_")
    kernel_files = {}
    for name, source in KERNELS.items():
        path = os.path.join(kern_dir, f"{name}.py")
        with open(path, "w") as f:
            f.write(source)
        kernel_files[name] = path

    results = {}
    for name in KERNELS:
        print(f"  {DISPLAY_NAMES[name]} ({device})...", flush=True)
        cache_dir = f"/tmp/bench_kern_cache_{name}"

        timings = []
        failed = False
        for s in range(NUM_SAMPLES):
            t = run_one_sample(python, warp_dir, kernel_files[name], device, cache_dir)
            if t is None:
                print(f"    sample {s+1}/{NUM_SAMPLES}: FAILED", flush=True)
                failed = True
                break
            timings.append(t)
            print(f"    sample {s+1}/{NUM_SAMPLES}: {t:.4f}s", flush=True)

        if failed or not timings:
            results[name] = {"median": -1, "samples": [], "cv": -1}
            continue

        med = statistics.median(timings)
        std = statistics.stdev(timings) if len(timings) > 1 else 0.0
        cv = (std / statistics.mean(timings) * 100) if statistics.mean(timings) > 0 else 0.0
        results[name] = {
            "median": round(med, 4),
            "cv": round(cv, 1),
            "samples": [round(t, 4) for t in timings],
        }
        print(f"    → median={med:.4f}s cv={cv:.1f}%", flush=True)

    shutil.rmtree(kern_dir, ignore_errors=True)
    return results


def main():
    all_results = {}

    for label, warp_dir in [("baseline", BASELINE_DIR), ("branch", BRANCH_DIR)]:
        print(f"\n{'='*60}", flush=True)
        print(f"Isolated kernels — {label} ({warp_dir})", flush=True)
        print(f"{'='*60}", flush=True)

        for device in ["cpu", "cuda:0"]:
            key = f"{label}_{device.replace(':', '_')}"
            all_results[key] = run_kernel_benchmarks(warp_dir, device)

    # Save
    out_path = "/home/horde/code-projects/warp-worktree-2/_benchmark_results_kernels_new.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # Print comparison table
    print(f"\n{'='*80}", flush=True)
    print("COMPARISON", flush=True)
    print(f"{'='*80}", flush=True)
    for device in ["cpu", "cuda:0"]:
        dev_key = device.replace(":", "_")
        base = all_results.get(f"baseline_{dev_key}", {})
        branch = all_results.get(f"branch_{dev_key}", {})
        print(f"\n### {device}", flush=True)
        print(f"{'Kernel':<30} {'Main':>8} {'Branch':>8} {'Speedup':>8}", flush=True)
        print("-" * 60, flush=True)
        for name in KERNELS:
            b = base.get(name, {}).get("median", -1)
            br = branch.get(name, {}).get("median", -1)
            if b > 0 and br > 0:
                speedup = b / br
                print(f"{DISPLAY_NAMES[name]:<30} {b:>7.3f}s {br:>7.3f}s {speedup:>7.2f}x", flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
