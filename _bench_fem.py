#!/usr/bin/env python3
"""Run FEM compile-time benchmarks sequentially on baseline and branch.

Uses each repo's own venv and cwd to ensure correct warp imports.
Clears WARP_CACHE_PATH between samples for true cold compiles.
"""

import json
import os
import re
import shutil
import statistics
import subprocess
import sys

REPOS = {
    "baseline": {
        "dir": "/home/horde/code-projects/warp",
        "python": "/home/horde/code-projects/warp/.venv/bin/python3",
    },
    "branch": {
        "dir": "/home/horde/code-projects/warp-worktree-3",
        "python": "/home/horde/code-projects/warp-worktree-3/.venv/bin/python3",
    },
}

RESULTS_FILE = "/tmp/fem_benchmark_results.json"

FEM_EXAMPLES = [
    ("fem.navier_stokes", "fem.example_navier_stokes", ["--num-frames", "1", "--resolution", "10", "--tri-mesh", "--headless"]),
    ("fem.stokes", "fem.example_stokes", ["--resolution", "10", "--nonconforming-pressures", "--headless"]),
    ("fem.deformed_geometry", "fem.example_deformed_geometry", ["--resolution", "10", "--mesh", "tri", "--headless"]),
    ("fem.diffusion_3d", "fem.example_diffusion_3d", ["--headless"]),
    ("fem.convection_diffusion", "fem.example_convection_diffusion", ["--resolution", "20", "--headless"]),
]


def run_fem_benchmark(python, warp_dir, device, samples=3):
    """Run FEM example benchmarks using repo's own venv. Returns {example_name: {median, stdev}}."""
    results = {}
    cache_base = f"/tmp/fem_cache_{device.replace(':', '_')}_{os.path.basename(warp_dir)}"

    for display, mod_name, extra in FEM_EXAMPLES:
        print(f"  {display}...", flush=True)

        timings = []
        skip = False
        for s in range(samples):
            # Completely wipe the cache dir so it's a true cold compile
            if os.path.isdir(cache_base):
                shutil.rmtree(cache_base)
            os.makedirs(cache_base, exist_ok=True)

            env = os.environ.copy()
            env["CUDA_CACHE_DISABLE"] = "1"
            env["WARP_CACHE_PATH"] = cache_base
            # Remove VIRTUAL_ENV to not confuse subprocess
            env.pop("VIRTUAL_ENV", None)

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
                    print(f"    ERR: {line}", flush=True)
                skip = True
                break

            if total_compile == 0:
                # Example ran but no compile lines found — likely cached
                print(f"    WARNING: 0s compile on sample {s+1} — may be cached", flush=True)

            timings.append(total_compile)
            print(f"    sample {s+1}/{samples}: {total_compile:.1f}s", flush=True)

        if skip or not timings:
            results[display] = {"median": -1, "stdev": -1}
            continue

        med = statistics.median(timings)
        std = statistics.stdev(timings) if len(timings) > 1 else 0.0
        results[display] = {"median": round(med, 1), "stdev": round(std, 1)}
        print(f"    median={med:.1f}s stdev={std:.1f}s", flush=True)

    return results


def main():
    all_results = {}

    for label, repo in REPOS.items():
        python = repo["python"]
        warp_dir = repo["dir"]
        all_results[label] = {}

        # Verify correct warp is imported
        check = subprocess.run(
            [python, "-c", "import warp; print(warp.__file__)"],
            capture_output=True, text=True, cwd=warp_dir,
        )
        print(f"\n{label}: warp from {check.stdout.strip()}", flush=True)

        for device in ["cpu", "cuda:0"]:
            print(f"\n{'='*60}", flush=True)
            print(f"FEM benchmarks: {label} — {device}", flush=True)
            print(f"{'='*60}", flush=True)

            all_results[label][device] = run_fem_benchmark(python, warp_dir, device, samples=3)

    # Save
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print comparison
    print(f"\n{'='*80}", flush=True)
    print("FEM COMPARISON", flush=True)
    print(f"{'='*80}", flush=True)

    for device in ["cpu", "cuda:0"]:
        print(f"\n### FEM Examples ({device})", flush=True)
        print(f"{'Example':<30} {'Main':>8} {'Branch':>8} {'Speedup':>8} {'Delta':>8}", flush=True)
        print("-" * 70, flush=True)

        for display, _, _ in FEM_EXAMPLES:
            b = all_results["baseline"][device].get(display, {}).get("median", -1)
            br = all_results["branch"][device].get(display, {}).get("median", -1)
            if b > 0 and br > 0:
                speedup = b / br
                delta_pct = (br - b) / b * 100
                flag = " *** REGRESSION" if delta_pct > 5 else ""
                print(f"{display:<30} {b:>7.1f}s {br:>7.1f}s {speedup:>7.2f}x {delta_pct:>+7.1f}%{flag}", flush=True)
            else:
                print(f"{display:<30} {'N/A':>8} {'N/A':>8}", flush=True)


if __name__ == "__main__":
    main()
