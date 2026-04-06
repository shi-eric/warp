#!/usr/bin/env python3
"""Benchmark total cold-compile time for the FEM Navier-Stokes example.

Spawns a fresh subprocess per sample to get true cold compiles — the FEM
system generates unique dynamic modules during fem.integrate(), so in-process
cache clearing is insufficient.

Uses enable_backward=False and CPU-only, matching the example's __main__ block.

IMPORTANT: If you modify native code under ``warp/native/``, you must rebuild
Warp with ``uv run build_lib.py --quick`` before running this benchmark,
otherwise the results will not reflect your changes.

Usage:
    uv run bench_fem_compile_time.py                    # 3 samples, cpu
    uv run bench_fem_compile_time.py --samples 5
    uv run bench_fem_compile_time.py --device cuda:0
"""

import argparse
import os
import shutil
import statistics
import subprocess
import sys
import time

CACHE_DIR = os.path.join(os.getcwd(), "bench_kernels_fem")

# This script is executed in a subprocess to compile the example once.
_WORKER_SCRIPT = """
import os, sys, time
os.environ["CUDA_CACHE_DISABLE"] = "1"

import warp as wp
wp.config.kernel_cache_dir = sys.argv[1]
wp.config.enable_backward = False
wp.init()

with wp.ScopedDevice(sys.argv[2]):
    start = time.perf_counter()
    from warp.examples.fem.example_navier_stokes import Example
    Example(quiet=True, resolution=10, degree=2)
    elapsed = time.perf_counter() - start

print("RESULT: {:.4f}".format(elapsed))
"""


def cold_compile_once(device: str) -> float:
    """Spawn a subprocess, compile the example, return wall-clock seconds."""
    # Wipe kernel cache so every compile is cold
    if os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    result = subprocess.run(
        [sys.executable, "-c", _WORKER_SCRIPT, CACHE_DIR, device],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr, file=sys.stderr)
        raise RuntimeError(f"Worker failed with exit code {result.returncode}")

    for line in result.stdout.splitlines():
        if line.startswith("RESULT:"):
            return float(line.split()[1])

    raise RuntimeError(f"No RESULT line in worker output:\n{result.stdout[-300:]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FEM Navier-Stokes compile time")
    parser.add_argument("--device", default="cpu", help='Target device (default: cpu)')
    parser.add_argument("--samples", type=int, default=3, help="Number of cold-compile samples (default: 3)")
    args = parser.parse_args()

    device: str = args.device
    num_samples: int = args.samples

    # Warm-up: first run may include one-time Warp init overhead
    print("Warm-up run...")
    try:
        t0 = cold_compile_once(device)
        print(f"  warm-up: {t0:.4f}s")
    except RuntimeError as e:
        print(f"Warm-up failed: {e}", file=sys.stderr)
        sys.exit(1)

    timings: list[float] = []
    for i in range(num_samples):
        t = cold_compile_once(device)
        timings.append(t)
        print(f"  sample {i + 1:>{len(str(num_samples))}}/{num_samples}: {t:.4f}s")

    median = statistics.median(timings)
    mean = statistics.mean(timings)
    stdev = statistics.stdev(timings) if num_samples > 1 else 0.0
    cv = (stdev / mean * 100) if mean > 0 else 0.0

    print()
    print(f"example: fem/example_navier_stokes.py")
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
