#!/usr/bin/env python3
"""Compile-time benchmarks for low-complexity Warp examples (non-FEM).

These examples compile 1–6 modules per process, filling the gap between
isolated kernel benchmarks (1 module) and FEM/Newton (dozens of modules).
They show how PCH amortization affects compile-guard speedups at
different workload scales.
"""

import json
import os
import re
import shutil
import statistics
import subprocess
import sys

BASELINE_DIR = "/home/horde/code-projects/warp"
BRANCH_DIR = "/home/horde/code-projects/warp-worktree-3"

NUM_SAMPLES = 3

_COMPILE_RE = re.compile(r"took ([\d.]+) ms\s+\(compiled\)")

# (display_name, module, extra_args, devices)
# "both" = cpu + cuda, "cuda" = cuda-only
CORE_EXAMPLES = [
    # 6 kernels
    ("core.fluid", "core.example_fluid",
     ["--headless", "--num-frames", "1"], "both"),
    ("core.sph", "core.example_sph",
     ["--num-frames", "1"], "both"),
    # 3 kernels
    ("core.wave", "core.example_wave",
     ["--num-frames", "1"], "both"),
    # 2 kernels
    ("core.dem", "core.example_dem",
     ["--num-frames", "1"], "both"),
    ("core.mesh", "core.example_mesh",
     [], "both"),
    # 1 kernel
    ("core.raycast", "core.example_raycast",
     ["--headless"], "both"),
    ("core.marching_cubes", "core.example_marching_cubes",
     [], "cuda"),
    ("core.nvdb", "core.example_nvdb",
     [], "both"),
    # optim (6 kernels)
    ("optim.diffray", "optim.example_diffray",
     ["--headless", "--num-frames", "1"], "both"),
    ("optim.particle_repulsion", "optim.example_particle_repulsion",
     ["--headless", "--num-frames", "1"], "both"),
]


def sum_compile_times(output):
    return sum(float(m.group(1)) / 1000.0 for m in _COMPILE_RE.finditer(output))


def run_example(python, warp_dir, module, device, extra_args, cache_dir):
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_CACHE_DISABLE"] = "1"
    env["WARP_CACHE_PATH"] = cache_dir
    env.pop("VIRTUAL_ENV", None)

    cmd = [python, "-m", f"warp.examples.{module}", "--device", device] + extra_args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                           cwd=warp_dir, env=env)
    except subprocess.TimeoutExpired:
        return None

    total = sum_compile_times(r.stdout + r.stderr)
    if r.returncode != 0 and total == 0:
        lines = r.stderr.strip().split("\n")
        for l in lines[-3:]:
            print(f"      ERR: {l}", file=sys.stderr, flush=True)
        return None
    return total


def dev_skip(device, devs):
    return device == "cpu" and devs == "cuda"


def main():
    all_results = {}

    for label, warp_dir in [("baseline", BASELINE_DIR), ("branch", BRANCH_DIR)]:
        python = os.path.join(warp_dir, ".venv", "bin", "python3")
        if not os.path.exists(python):
            print(f"WARNING: {python} not found", flush=True)
            python = sys.executable

        # Verify warp import
        r = subprocess.run([python, "-c", "import warp; print(warp.__file__)"],
                           capture_output=True, text=True, cwd=warp_dir)
        print(f"\nlabel={label}, warp={r.stdout.strip()}", flush=True)

        for device in ["cpu", "cuda:0"]:
            key = f"core_{device}"
            print(f"\n{'='*60}", flush=True)
            print(f"Core examples — {label} — {device}", flush=True)
            print(f"{'='*60}", flush=True)

            results = {}
            for display, mod, extra, devs in CORE_EXAMPLES:
                if dev_skip(device, devs):
                    continue
                print(f"  {display} ({device})...", flush=True)
                timings = []
                failed = False
                for s in range(NUM_SAMPLES):
                    cache = f"/tmp/bench_cache_{label}_{device.replace(':', '_')}_{display.replace('.', '_')}"
                    t = run_example(python, warp_dir, mod, device, extra, cache)
                    if t is None:
                        print(f"    sample {s+1}/{NUM_SAMPLES}: FAILED", flush=True)
                        failed = True
                        break
                    timings.append(t)
                    print(f"    sample {s+1}/{NUM_SAMPLES}: {t:.1f}s", flush=True)

                if failed or not timings:
                    results[display] = {"median": -1, "stdev": -1, "samples": [], "status": "failed"}
                    continue

                med = statistics.median(timings)
                std = statistics.stdev(timings) if len(timings) > 1 else 0.0
                cv = (std / statistics.mean(timings) * 100) if statistics.mean(timings) > 0 else 0.0
                results[display] = {
                    "median": round(med, 2),
                    "stdev": round(std, 2),
                    "cv": round(cv, 1),
                    "samples": [round(t, 2) for t in timings],
                    "status": "ok",
                }
                print(f"    → median={med:.1f}s stdev={std:.1f}s cv={cv:.1f}%", flush=True)

            all_results[f"{label}_{key}"] = results

    # Save
    out_path = "/home/horde/code-projects/warp-worktree-2/_benchmark_results_core.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # Print comparison
    print(f"\n{'='*80}", flush=True)
    print("COMPARISON", flush=True)
    print(f"{'='*80}", flush=True)
    for device in ["cpu", "cuda:0"]:
        key = f"core_{device}"
        base = all_results.get(f"baseline_{key}", {})
        branch = all_results.get(f"branch_{key}", {})
        print(f"\n### {device}", flush=True)
        print(f"{'Example':<35} {'Main':>8} {'Branch':>8} {'Speedup':>8}", flush=True)
        print("-" * 65, flush=True)
        for display, _, _, _ in CORE_EXAMPLES:
            b = base.get(display, {}).get("median", -1)
            br = branch.get(display, {}).get("median", -1)
            if b > 0 and br > 0:
                speedup = b / br
                print(f"{display:<35} {b:>7.1f}s {br:>7.1f}s {speedup:>7.2f}x", flush=True)
            elif b == -1 and br == -1:
                pass  # skip if both failed or skipped
            else:
                print(f"{display:<35} {'N/A':>8} {'N/A':>8}", flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
