#!/usr/bin/env python3
"""Benchmark compile times for representative kernels across feature categories.

Each kernel is in a separate module so compile guards apply independently.
Run on both branches to produce a before/after comparison.

Usage:
    uv run _bench_comparison.py --device cpu --samples 5
    uv run _bench_comparison.py --device cuda:0 --samples 5
"""

import argparse
import importlib
import os
import statistics
import sys
import time

os.environ.setdefault("CUDA_CACHE_DISABLE", "1")

# Create isolated kernel files so each gets its own Warp module.
KERNEL_DIR = os.path.join(os.path.dirname(__file__), "bench_kernels_src")
os.makedirs(KERNEL_DIR, exist_ok=True)

KERNELS = {
    "scalar_only": '''
import warp as wp

@wp.kernel
def kernel(x: wp.array2d(dtype=float), y: wp.array2d(dtype=float)):
    """Scalar math only — best case for guards."""
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
    """Vector math — uses vec.h."""
    tid = wp.tid()
    p = points[tid]
    n = normals[tid]
    result[tid] = wp.dot(p, n) / (wp.length(p) + 1e-6)
''',
    "mat_quat": '''
import warp as wp

@wp.kernel
def kernel(
    positions: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.quatf),
    transforms: wp.array(dtype=wp.mat44),
    output: wp.array(dtype=wp.vec3),
):
    """Matrix + quaternion — uses mat.h, quat.h, vec.h."""
    tid = wp.tid()
    p = positions[tid]
    q = rotations[tid]
    m = transforms[tid]
    rotated = wp.quat_rotate(q, p)
    output[tid] = wp.transform_point(m, rotated)
''',
    "mesh_query": '''
import warp as wp

@wp.kernel
def kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=float),
):
    """Mesh queries — uses mesh.h, bvh.h, intersect.h, tile.h, vec.h, mat.h."""
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
    "noise_rand": '''
import warp as wp

@wp.kernel
def kernel(
    seed: int,
    positions: wp.array(dtype=wp.vec3),
    output: wp.array(dtype=float),
):
    """Noise + random — uses noise.h, rand.h, vec.h."""
    tid = wp.tid()
    state = wp.rand_init(seed, tid)
    p = positions[tid] + wp.vec3(wp.randf(state), wp.randf(state), wp.randf(state))
    output[tid] = wp.noise(wp.uint32(seed), p[0]) + wp.noise(wp.uint32(seed), p[1])
''',
    "volume_sample": '''
import warp as wp

@wp.kernel
def kernel(
    volume: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    output: wp.array(dtype=float),
):
    """Volume sampling — uses volume.h, vec.h, mat.h."""
    tid = wp.tid()
    p = points[tid]
    output[tid] = wp.volume_sample_f(volume, p, wp.Volume.LINEAR)
''',
    "kitchen_sink": '''
import warp as wp

@wp.kernel
def kernel(
    mesh_id: wp.uint64,
    volume: wp.uint64,
    seed: int,
    points: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.quatf),
    matrices: wp.array(dtype=wp.mat44),
    output: wp.array(dtype=float),
):
    """All features — mesh, volume, noise, rand, vec, mat, quat."""
    tid = wp.tid()
    p = points[tid]
    q = rotations[tid]
    m = matrices[tid]
    p2 = wp.transform_point(m, wp.quat_rotate(q, p))
    sign = float(0.0)
    f = int(0)
    u = float(0.0)
    v = float(0.0)
    wp.mesh_query_point(mesh_id, p2, 1000.0, sign, f, u, v)
    vol_val = wp.volume_sample_f(volume, p2, wp.Volume.LINEAR)
    state = wp.rand_init(seed, tid)
    n = wp.noise(wp.uint32(seed), p2[0] + wp.randf(state))
    output[tid] = vol_val + n + sign
''',
}

DISPLAY_NAMES = {
    "scalar_only": "Scalar only (trivial)",
    "vec_math": "Vector math",
    "mat_quat": "Mat + quat + transform",
    "mesh_query": "Mesh queries",
    "noise_rand": "Noise + random",
    "volume_sample": "Volume sampling",
    "kitchen_sink": "All features combined",
}


def setup_kernel_files():
    """Write each kernel to a separate file."""
    init_path = os.path.join(KERNEL_DIR, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("")

    for name, source in KERNELS.items():
        path = os.path.join(KERNEL_DIR, f"{name}.py")
        with open(path, "w") as f:
            f.write(source)


def cold_compile(mod_name: str, device: str) -> float:
    """Clear cache, unload, recompile. Return seconds."""
    import warp as wp

    # Reload the Python module to get fresh Warp module
    mod = sys.modules.get(mod_name)
    if mod is None:
        mod = importlib.import_module(mod_name)

    # Find the Warp module
    warp_mod = None
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name)
        if isinstance(obj, wp.Kernel):
            warp_mod = obj.module
            break

    if warp_mod is None:
        raise RuntimeError(f"No kernel in {mod_name}")

    wp.clear_kernel_cache()
    warp_mod.unload()

    start = time.perf_counter()
    warp_mod.load(device)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    setup_kernel_files()

    # Add kernel dir to path
    parent = os.path.dirname(KERNEL_DIR)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    import warp as wp

    wp.config.kernel_cache_dir = os.path.join(os.getcwd(), "bench_kernels")
    wp.config.quiet = True
    wp.init()
    wp.clear_kernel_cache()

    # Import all modules first (warm up)
    for name in KERNELS:
        importlib.import_module(f"bench_kernels_src.{name}")

    # Warm up compile
    for name in KERNELS:
        mod_name = f"bench_kernels_src.{name}"
        mod = sys.modules[mod_name]
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if isinstance(obj, wp.Kernel):
                obj.module.load(args.device)

    print(f"device: {args.device}, samples: {args.samples}")
    print(f"warp: {wp.__version__}")
    print()
    print(f"{'Kernel':<30} {'Median':>8} {'Mean':>8} {'Stdev':>7} {'CV':>6}")
    print("-" * 65)

    for name in KERNELS:
        mod_name = f"bench_kernels_src.{name}"
        display = DISPLAY_NAMES[name]
        timings = []
        for _ in range(args.samples):
            t = cold_compile(mod_name, args.device)
            timings.append(t)

        med = statistics.median(timings)
        mean = statistics.mean(timings)
        std = statistics.stdev(timings) if len(timings) > 1 else 0.0
        cv = (std / mean * 100) if mean > 0 else 0.0
        print(f"{display:<30} {med:>7.3f}s {mean:>7.3f}s {std:>6.3f}s {cv:>5.1f}%")


if __name__ == "__main__":
    main()
