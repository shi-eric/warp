#!/usr/bin/env python3
"""Measure the compile-time impact of each compile guard individually.

For each suspect guard, disables it (forces the header to be included)
and measures the compile-time delta vs the baseline where all guards
are active.

Usage:
    uv run bench_guard_impact.py
"""

import os
import statistics
import time

os.environ.setdefault("CUDA_CACHE_DISABLE", "1")

import warp as wp
import warp._src.codegen as codegen

KERNEL_CACHE_DIR = os.path.join(os.getcwd(), "bench_kernels")
NUM_SAMPLES = 5


@wp.kernel
def array2d_augassign_kernel(
    x: wp.array2d(dtype=float),
    y: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    x[i, j] += y[i, j]


def cold_compile_once(device: str) -> float:
    wp.clear_kernel_cache()
    array2d_augassign_kernel.module.unload()
    start = time.perf_counter()
    wp.load_module(device=device)
    elapsed = time.perf_counter() - start
    return elapsed


def measure(label: str) -> float:
    """Warm up once, then collect NUM_SAMPLES and return median."""
    wp.clear_kernel_cache()
    wp.load_module(device="cpu")  # warm-up

    timings = []
    for _ in range(NUM_SAMPLES):
        t = cold_compile_once("cpu")
        timings.append(t)
    median = statistics.median(timings)
    print(f"  {label}: {median:.4f}s")
    return median


def main():
    wp.config.kernel_cache_dir = KERNEL_CACHE_DIR
    wp.init()
    wp.clear_kernel_cache()

    # Save original markers
    original_markers = dict(codegen._COMPILE_GUARD_MARKERS)

    # Baseline with all guards active
    print("Measuring baseline (all guards active)...")
    baseline = measure("baseline")

    # Guards to test — some must be disabled together due to header deps.
    # Each entry is (label, list of guards to disable).
    suspect_guards = [
        ("WP_NO_VEC", ["WP_NO_VEC"]),
        ("WP_NO_MAT", ["WP_NO_MAT"]),
        ("WP_NO_QUAT", ["WP_NO_QUAT"]),
        ("WP_NO_RAND (needs vec)", ["WP_NO_RAND", "WP_NO_VEC"]),
        ("WP_NO_MATNN", ["WP_NO_MATNN"]),
        ("WP_NO_FABRIC", ["WP_NO_FABRIC"]),
        ("WP_NO_HASHGRID (needs vec)", ["WP_NO_HASHGRID", "WP_NO_VEC"]),
    ]

    high_value_guards = [
        ("WP_NO_FLOAT16_OPS", ["WP_NO_FLOAT16_OPS"]),
        ("WP_NO_FLOAT64_OPS", ["WP_NO_FLOAT64_OPS"]),
        ("WP_NO_MESH", ["WP_NO_MESH"]),
        ("WP_NO_TILE", ["WP_NO_TILE"]),
        ("WP_NO_SVD (needs vec+mat)", ["WP_NO_SVD", "WP_NO_VEC", "WP_NO_MAT"]),
        ("WP_NO_NOISE", ["WP_NO_NOISE"]),
        ("WP_NO_TEXTURE", ["WP_NO_TEXTURE"]),
        ("WP_NO_INTERSECT", ["WP_NO_INTERSECT"]),
        ("WP_NO_BVH", ["WP_NO_BVH"]),
        ("WP_NO_VOLUME (needs vec+mat)", ["WP_NO_VOLUME", "WP_NO_VEC", "WP_NO_MAT"]),
    ]

    results = {}

    def test_group(label, guard_list):
        print(f"\n--- {label} ---")
        group_results = {}
        for name, guards_to_disable in guard_list:
            codegen._COMPILE_GUARD_MARKERS = {
                k: v for k, v in original_markers.items() if k not in guards_to_disable
            }
            try:
                median = measure(f"without {name}")
                delta = median - baseline
                group_results[name] = (median, delta)
            except Exception as e:
                print(f"  without {name}: FAILED ({e})")
                group_results[name] = (None, None)
            codegen._COMPILE_GUARD_MARKERS = dict(original_markers)
        return group_results

    suspect_results = test_group("Suspected low-value guards", suspect_guards)
    high_results = test_group("High-value guards (comparison)", high_value_guards)

    # Also test disabling ALL suspect guards at once
    print("\n--- All suspected low-value guards disabled at once ---")
    all_suspect_keys = set()
    for _, keys in suspect_guards:
        all_suspect_keys.update(keys)
    codegen._COMPILE_GUARD_MARKERS = {
        k: v for k, v in original_markers.items() if k not in all_suspect_keys
    }
    try:
        all_median = measure("without ALL suspects")
        all_suspects_delta = all_median - baseline
    except Exception:
        all_median = None
        all_suspects_delta = None
    codegen._COMPILE_GUARD_MARKERS = dict(original_markers)

    # Summary
    print("\n" + "=" * 65)
    print(f"{'Guard':<30} {'Median':>8} {'Delta':>10} {'Impact':>8}")
    print("-" * 65)
    print(f"{'baseline':<30} {baseline:>8.4f}s {'':>10} {'':>8}")

    for label, group_results in [("Suspected low-value:", suspect_results),
                                  ("High-value (comparison):", high_results)]:
        print(f"\n{label}")
        for name in [n for n, _ in (suspect_guards if "low" in label else high_value_guards)]:
            median, delta = group_results.get(name, (None, None))
            if median is None:
                print(f"  {name:<28} {'FAILED':>8}")
            else:
                pct = delta / baseline * 100
                print(f"  {name:<28} {median:>8.4f}s {delta:>+10.4f}s {pct:>+7.1f}%")

    if all_median is not None:
        pct = all_suspects_delta / baseline * 100
        print(f"\n  {'ALL suspects combined':<28} {all_median:>8.4f}s {all_suspects_delta:>+10.4f}s {pct:>+7.1f}%")


if __name__ == "__main__":
    main()
