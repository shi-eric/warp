#!/usr/bin/env python3
"""Additional core example benchmarks — raymarch and graph_capture."""

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

EXTRA_EXAMPLES = [
    ("core.raymarch", "core.example_raymarch",
     ["--headless", "--num-frames", "1"], "both"),
    ("core.graph_capture", "core.example_graph_capture",
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


def main():
    all_results = {}

    for label, warp_dir in [("baseline", BASELINE_DIR), ("branch", BRANCH_DIR)]:
        python = os.path.join(warp_dir, ".venv", "bin", "python3")

        for device in ["cpu", "cuda:0"]:
            key = f"{label}_{device.replace(':', '_')}"
            print(f"\n{'='*60}", flush=True)
            print(f"Extra core — {label} — {device}", flush=True)
            print(f"{'='*60}", flush=True)

            results = {}
            for display, mod, extra, devs in EXTRA_EXAMPLES:
                if device == "cpu" and devs == "cuda":
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
                    results[display] = {"median": -1, "status": "failed"}
                    continue

                med = statistics.median(timings)
                std = statistics.stdev(timings) if len(timings) > 1 else 0.0
                cv = (std / statistics.mean(timings) * 100) if statistics.mean(timings) > 0 else 0.0
                results[display] = {
                    "median": round(med, 2),
                    "cv": round(cv, 1),
                    "samples": [round(t, 2) for t in timings],
                    "status": "ok",
                }
                print(f"    → median={med:.1f}s cv={cv:.1f}%", flush=True)

            all_results[key] = results

    # Print comparison
    print(f"\n{'='*80}", flush=True)
    for device in ["cpu", "cuda:0"]:
        dev_key = device.replace(":", "_")
        base = all_results.get(f"baseline_{dev_key}", {})
        branch = all_results.get(f"branch_{dev_key}", {})
        print(f"\n### {device}", flush=True)
        for display, _, _, _ in EXTRA_EXAMPLES:
            b = base.get(display, {}).get("median", -1)
            br = branch.get(display, {}).get("median", -1)
            if b > 0 and br > 0:
                print(f"  {display:<30} {b:.1f}s → {br:.1f}s  {b/br:.2f}x", flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
