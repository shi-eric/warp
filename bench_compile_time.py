#!/usr/bin/env python3
"""Standalone benchmark for Warp module compile times.

Collects multiple samples of cold-compile time and reports median, mean,
std-dev, min, max, and coefficient of variation (CV).  A CV below ~5 %
indicates stable sampling.

IMPORTANT: If you modify native code under ``warp/native/``, you must rebuild
Warp with ``uv run build_lib.py --quick`` before running this benchmark,
otherwise the results will not reflect your changes.

Usage:
    uv run bench_compile_time.py              # defaults: cpu, 10 samples
    uv run bench_compile_time.py --device cuda:0 --samples 20
"""

import argparse
import os
import statistics
import time

# Disable the NVIDIA CUDA driver compute cache so that each cold-compile
# sample truly starts from scratch.  See docs/deep_dive/profiling.rst for
# details on the two-layer cache system.
os.environ.setdefault("CUDA_CACHE_DISABLE", "1")

import warp as wp

# ---------------------------------------------------------------------------
# Configuration — change DEVICE here (or via CLI) to switch CPU / CUDA
# ---------------------------------------------------------------------------
DEVICE = "cpu"  # "cpu" or "cuda:0"
NUM_SAMPLES = 10
KERNEL_CACHE_DIR = os.path.join(os.getcwd(), "bench_kernels")

# ---------------------------------------------------------------------------
# Kernel under test
# ---------------------------------------------------------------------------


@wp.kernel
def array2d_augassign_kernel(
    x: wp.array2d(dtype=float),
    y: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    x[i, j] += y[i, j]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cold_compile_once(device: str) -> float:
    """Clear cache, compile, return wall-clock seconds."""
    wp.clear_kernel_cache()
    array2d_augassign_kernel.module.unload()

    start = time.perf_counter()
    wp.load_module(device=device)
    elapsed = time.perf_counter() - start
    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Warp module compile time")
    parser.add_argument(
        "--device",
        default=DEVICE,
        help='Target device, e.g. "cpu" or "cuda:0" (default: %(default)s)',
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=NUM_SAMPLES,
        help="Number of cold-compile samples (default: %(default)s)",
    )
    args = parser.parse_args()

    device: str = args.device
    num_samples: int = args.samples

    wp.config.kernel_cache_dir = KERNEL_CACHE_DIR
    wp.init()

    # Clear stale cached objects so every run starts fresh
    wp.clear_kernel_cache()

    # Warm-up: compile once so that any first-time init cost is excluded
    wp.load_module(device=device)

    timings: list[float] = []
    for i in range(num_samples):
        t = cold_compile_once(device)
        timings.append(t)
        print(f"  sample {i + 1:>{len(str(num_samples))}}/{num_samples}: {t:.4f}s")

    # --- Statistics ---
    median = statistics.median(timings)
    mean = statistics.mean(timings)
    stdev = statistics.stdev(timings) if num_samples > 1 else 0.0
    cv = (stdev / mean * 100) if mean > 0 else 0.0

    print()
    print(f"device:  {device}")
    print(f"samples: {num_samples}")
    print(f"median:  {median:.4f}s")
    print(f"mean:    {mean:.4f}s")
    print(f"stdev:   {stdev:.4f}s")
    print(f"min:     {min(timings):.4f}s")
    print(f"max:     {max(timings):.4f}s")
    print(f"CV:      {cv:.1f}%")
    if cv > 10:
        print("WARNING: CV > 10% — results may be noisy, consider more samples or reducing system load")
    print()
    print(f"RESULT: {median:.4f}")


if __name__ == "__main__":
    main()
