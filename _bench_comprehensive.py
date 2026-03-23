#!/usr/bin/env python3
"""Comprehensive compile-time benchmark: warp.fem + Newton examples.

Runs each example as a subprocess with WARP_CACHE_PATH wiped between
samples for true cold compiles.  Sums ``took N ms (compiled)`` lines
from Warp output to get total compile time per run.

Usage:
    python _bench_comprehensive.py                          # all, default 3 samples
    python _bench_comprehensive.py --suite fem --device cpu  # just FEM on CPU
    python _bench_comprehensive.py --suite newton --samples 5
    python _bench_comprehensive.py --label baseline          # tag in output filenames
"""

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

# ── repo paths ──────────────────────────────────────────────────────
BASELINE_WARP = "/home/horde/code-projects/warp"
BRANCH_WARP = "/home/horde/code-projects/warp-worktree-3"
NEWTON_DIR = "/home/horde/code-projects/newton"
NEWTON_PYTHON = "/home/horde/code-projects/newton/.venv/bin/python3"

# ── FEM examples ────────────────────────────────────────────────────
# (display_name, module, extra_args, devices)
# devices: "both" = cpu + cuda, "cuda" = cuda-only
FEM_EXAMPLES = [
    # --- single-shot solves (no --num-frames needed) ---
    ("fem.diffusion", "fem.example_diffusion",
     ["--resolution", "10", "--mesh", "tri", "--headless"], "both"),
    ("fem.diffusion_3d", "fem.example_diffusion_3d",
     ["--headless"], "both"),
    ("fem.deformed_geometry", "fem.example_deformed_geometry",
     ["--resolution", "10", "--mesh", "tri", "--headless"], "both"),
    ("fem.stokes", "fem.example_stokes",
     ["--resolution", "10", "--nonconforming-pressures", "--headless"], "both"),
    ("fem.stokes_transfer", "fem.example_stokes_transfer",
     ["--headless"], "both"),
    ("fem.mixed_elasticity", "fem.example_mixed_elasticity",
     ["--nonconforming-stresses", "--mesh", "quad", "--headless"], "both"),
    ("fem.magnetostatics", "fem.example_magnetostatics",
     ["--resolution", "16", "--headless"], "both"),
    ("fem.streamlines", "fem.example_streamlines",
     ["--headless"], "both"),
    ("fem.adaptive_grid", "fem.example_adaptive_grid",
     ["--headless", "--div-conforming"], "both"),
    ("fem.distortion_energy", "fem.example_distortion_energy",
     ["--resolution", "16", "--headless"], "both"),
    ("fem.nonconforming_contact", "fem.example_nonconforming_contact",
     ["--resolution", "16", "--num-steps", "2", "--headless"], "both"),

    # --- time-stepping (use minimal frames) ---
    ("fem.convection_diffusion", "fem.example_convection_diffusion",
     ["--resolution", "20", "--headless"], "both"),
    ("fem.navier_stokes", "fem.example_navier_stokes",
     ["--num-frames", "1", "--resolution", "10", "--tri-mesh", "--headless"], "both"),
    ("fem.burgers", "fem.example_burgers",
     ["--resolution", "20", "--num-frames", "25", "--degree", "1", "--headless"], "both"),
    ("fem.convection_diffusion_dg", "fem.example_convection_diffusion_dg",
     ["--resolution", "20", "--num-frames", "25", "--headless"], "both"),
    ("fem.taylor_green", "fem.example_taylor_green",
     ["--num-frames", "10", "--resolution", "10", "--headless"], "both"),
    ("fem.shallow_water", "fem.example_shallow_water",
     ["--num-frames", "10", "--resolution", "10", "--headless"], "both"),
    ("fem.kelvin_helmholtz", "fem.example_kelvin_helmholtz",
     ["--num-frames", "25", "--resolution", "20", "--headless"], "both"),
    ("fem.apic_fluid", "fem.example_apic_fluid",
     ["--num-frames", "1", "--voxel-size", "2.0"], "cuda"),

    # --- optimization (minimal iters) ---
    ("fem.elastic_shape_optimization", "fem.example_elastic_shape_optimization",
     ["--num-iters", "5", "--headless"], "both"),
    ("fem.darcy_ls_optimization", "fem.example_darcy_ls_optimization",
     ["--num-iters", "5", "--resolution", "25", "--headless"], "both"),
]

# ── Newton examples ─────────────────────────────────────────────────
# (display_name, newton_example_name, extra_args, devices)
# We skip: replay_viewer (interactive), basic_viewer (interactive),
#   basic_plotting (needs display), recording (needs file output),
#   robot_anymal_c_walk (torch), mpm_anymal (torch), robot_policy (torch),
#   diffsim_bear (USD asset), sensor_tiled_camera (rendering),
#   mpm_grain_rendering (rendering)
NEWTON_EXAMPLES = [
    # basic
    ("newton.basic_pendulum", "basic_pendulum", [], "both"),
    ("newton.basic_urdf", "basic_urdf", ["--world-count", "4"], "both"),
    ("newton.basic_joints", "basic_joints", [], "both"),
    ("newton.basic_shapes", "basic_shapes", [], "both"),
    ("newton.basic_conveyor", "basic_conveyor", [], "both"),
    ("newton.basic_heightfield", "basic_heightfield", [], "both"),

    # cable
    ("newton.cable_twist", "cable_twist", [], "both"),
    ("newton.cable_y_junction", "cable_y_junction", [], "both"),
    ("newton.cable_bundle_hysteresis", "cable_bundle_hysteresis", [], "both"),
    ("newton.cable_pile", "cable_pile", [], "both"),

    # cloth
    ("newton.cloth_bending", "cloth_bending", [], "both"),
    ("newton.cloth_hanging", "cloth_hanging", [], "both"),
    ("newton.cloth_style3d", "cloth_style3d", [], "cuda"),
    ("newton.cloth_twist", "cloth_twist", [], "cuda"),
    ("newton.cloth_rollers", "cloth_rollers", [], "cuda"),
    ("newton.cloth_poker_cards", "cloth_poker_cards", [], "cuda"),
    ("newton.cloth_franka", "cloth_franka", [], "cuda"),
    ("newton.cloth_h1", "cloth_h1", [], "cuda"),

    # contacts
    ("newton.brick_stacking", "brick_stacking", [], "cuda"),
    ("newton.nut_bolt_sdf", "nut_bolt_sdf", [], "cuda"),
    ("newton.nut_bolt_hydro", "nut_bolt_hydro", [], "cuda"),
    ("newton.pyramid", "pyramid", [], "cuda"),

    # diffsim
    ("newton.diffsim_ball", "diffsim_ball", [], "both"),
    ("newton.diffsim_cloth", "diffsim_cloth", [], "both"),
    ("newton.diffsim_drone", "diffsim_drone", [], "both"),
    ("newton.diffsim_spring_cage", "diffsim_spring_cage", [], "both"),
    ("newton.diffsim_soft_body", "diffsim_soft_body", [], "both"),

    # ik
    ("newton.ik_franka", "ik_franka", [], "both"),
    ("newton.ik_h1", "ik_h1", [], "both"),
    ("newton.ik_custom", "ik_custom", [], "cuda"),
    ("newton.ik_cube_stacking", "ik_cube_stacking", [], "both"),

    # mpm
    ("newton.mpm_granular", "mpm_granular", [], "cuda"),
    ("newton.mpm_multi_material", "mpm_multi_material", [], "cuda"),
    ("newton.mpm_twoway_coupling", "mpm_twoway_coupling", [], "cuda"),
    ("newton.mpm_beam_twist", "mpm_beam_twist", [], "cuda"),
    ("newton.mpm_snow_ball", "mpm_snow_ball", [], "cuda"),
    ("newton.mpm_viscous", "mpm_viscous", [], "cuda"),

    # robot
    ("newton.robot_cartpole", "robot_cartpole", [], "both"),
    ("newton.robot_anymal_d", "robot_anymal_d", [], "both"),
    ("newton.robot_ur10", "robot_ur10", [], "both"),
    ("newton.robot_allegro_hand", "robot_allegro_hand", [], "cuda"),
    ("newton.robot_g1", "robot_g1", [], "cuda"),
    ("newton.robot_h1", "robot_h1", [], "cuda"),
    ("newton.robot_panda_hydro", "robot_panda_hydro", [], "cuda"),

    # selection
    ("newton.selection_articulations", "selection_articulations", [], "both"),
    ("newton.selection_cartpole", "selection_cartpole", [], "both"),
    ("newton.selection_materials", "selection_materials", [], "both"),
    ("newton.selection_multiple", "selection_multiple", [], "both"),

    # sensors
    ("newton.sensor_contact", "sensor_contact", [], "both"),
    ("newton.sensor_imu", "sensor_imu", [], "both"),

    # softbody + multiphysics
    ("newton.softbody_hanging", "softbody_hanging", [], "cuda"),
    ("newton.softbody_franka", "softbody_franka", [], "cuda"),
    ("newton.softbody_gift", "softbody_gift", [], "cuda"),
    ("newton.softbody_dropping_to_cloth", "softbody_dropping_to_cloth", [], "cuda"),
]


# ── helpers ─────────────────────────────────────────────────────────

_COMPILE_RE = re.compile(r"took ([\d.]+) ms\s+\(compiled\)")


def sum_compile_times(output: str) -> float:
    """Sum all ``took N ms (compiled)`` from combined stdout+stderr."""
    return sum(float(m.group(1)) / 1000.0 for m in _COMPILE_RE.finditer(output))


def run_fem_example(python: str, warp_dir: str, module: str, device: str,
                    extra_args: list[str], cache_dir: str) -> float | None:
    """Run one FEM example, return total compile seconds or None on error."""
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
        return None
    return total


def run_newton_example(example_name: str, device: str, extra_args: list[str],
                       cache_dir: str, warp_dir: str | None = None) -> float | None:
    """Run one Newton example, return total compile seconds or None on error."""
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_CACHE_DISABLE"] = "1"
    env["WARP_CACHE_PATH"] = cache_dir
    env.pop("VIRTUAL_ENV", None)
    if warp_dir:
        env["PYTHONPATH"] = warp_dir

    cmd = [NEWTON_PYTHON, "-m", "newton.examples", example_name,
           "--viewer", "null", "--num-frames", "1",
           "--device", device] + extra_args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                           cwd=NEWTON_DIR, env=env)
    except subprocess.TimeoutExpired:
        return None

    total = sum_compile_times(r.stdout + r.stderr)
    if r.returncode != 0 and total == 0:
        # Print last error for debugging
        lines = r.stderr.strip().split("\n")
        for l in lines[-3:]:
            print(f"      ERR: {l}", file=sys.stderr, flush=True)
        return None
    return total


def benchmark_examples(run_fn, examples, device, num_samples, label):
    """Run benchmarks for a list of examples. Returns dict of results."""
    results = {}
    for display, *args in examples:
        # Check device compatibility
        devices_flag = args[-1] if isinstance(args[-1], str) and args[-1] in ("both", "cuda") else "both"
        if device == "cpu" and devices_flag == "cuda":
            continue

        print(f"  {display} ({device})...", flush=True)
        timings = []
        failed = False

        for s in range(num_samples):
            cache = f"/tmp/bench_cache_{label}_{device.replace(':', '_')}_{display.replace('.', '_')}"
            t = run_fn(display, device, cache, s)
            if t is None:
                print(f"    sample {s+1}/{num_samples}: FAILED", flush=True)
                failed = True
                break
            timings.append(t)
            print(f"    sample {s+1}/{num_samples}: {t:.1f}s", flush=True)

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

    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive compile-time benchmarks")
    parser.add_argument("--suite", choices=["all", "fem", "newton"], default="all")
    parser.add_argument("--device", choices=["cpu", "cuda:0", "both"], default="both")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--label", default="baseline",
                        help="Tag for output: 'baseline' uses main warp, 'branch' uses worktree-3")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: _benchmark_{label}.json)")
    args = parser.parse_args()

    if args.label == "branch":
        warp_dir = BRANCH_WARP
    else:
        warp_dir = BASELINE_WARP

    # Determine python for FEM (uses warp's own venv)
    fem_python = os.path.join(warp_dir, ".venv", "bin", "python3")
    if not os.path.exists(fem_python):
        print(f"WARNING: {fem_python} not found, FEM benchmarks may fail", flush=True)
        fem_python = sys.executable

    devices = ["cpu", "cuda:0"] if args.device == "both" else [args.device]
    output_path = args.output or f"_benchmark_{args.label}.json"

    # Verify warp import
    r = subprocess.run([fem_python, "-c", "import warp; print(warp.__file__)"],
                       capture_output=True, text=True, cwd=warp_dir)
    print(f"label={args.label}, warp={r.stdout.strip()}", flush=True)
    print(f"suite={args.suite}, devices={devices}, samples={args.samples}", flush=True)

    all_results = {}

    for device in devices:
        # ── FEM ──
        if args.suite in ("all", "fem"):
            print(f"\n{'='*60}", flush=True)
            print(f"FEM examples — {args.label} — {device}", flush=True)
            print(f"{'='*60}", flush=True)

            def run_fem(display, dev, cache, sample_idx):
                # Find the matching example tuple
                for d, mod, extra, devs in FEM_EXAMPLES:
                    if d == display:
                        return run_fem_example(fem_python, warp_dir, mod, dev, extra, cache)
                return None

            fem_results = {}
            for display, mod, extra, devs in FEM_EXAMPLES:
                if dev_skip(device, devs):
                    continue
                print(f"  {display} ({device})...", flush=True)
                timings = []
                failed = False
                for s in range(args.samples):
                    cache = f"/tmp/bench_cache_{args.label}_{device.replace(':', '_')}_{display.replace('.', '_')}"
                    t = run_fem_example(fem_python, warp_dir, mod, device, extra, cache)
                    if t is None:
                        print(f"    sample {s+1}/{args.samples}: FAILED", flush=True)
                        failed = True
                        break
                    timings.append(t)
                    print(f"    sample {s+1}/{args.samples}: {t:.1f}s", flush=True)

                if failed or not timings:
                    fem_results[display] = {"median": -1, "stdev": -1, "samples": [], "status": "failed"}
                    continue

                med = statistics.median(timings)
                std = statistics.stdev(timings) if len(timings) > 1 else 0.0
                cv = (std / statistics.mean(timings) * 100) if statistics.mean(timings) > 0 else 0.0
                fem_results[display] = {
                    "median": round(med, 2),
                    "stdev": round(std, 2),
                    "cv": round(cv, 1),
                    "samples": [round(t, 2) for t in timings],
                    "status": "ok",
                }
                print(f"    → median={med:.1f}s stdev={std:.1f}s cv={cv:.1f}%", flush=True)

            all_results[f"fem_{device}"] = fem_results

        # ── Newton ──
        if args.suite in ("all", "newton"):
            print(f"\n{'='*60}", flush=True)
            print(f"Newton examples — {args.label} — {device}", flush=True)
            print(f"{'='*60}", flush=True)

            newton_warp = warp_dir if args.label == "branch" else None

            newton_results = {}
            for display, example_name, extra, devs in NEWTON_EXAMPLES:
                if dev_skip(device, devs):
                    continue
                print(f"  {display} ({device})...", flush=True)
                timings = []
                failed = False
                for s in range(args.samples):
                    cache = f"/tmp/bench_cache_{args.label}_{device.replace(':', '_')}_{display.replace('.', '_')}"
                    t = run_newton_example(example_name, device, extra, cache, newton_warp)
                    if t is None:
                        print(f"    sample {s+1}/{args.samples}: FAILED", flush=True)
                        failed = True
                        break
                    timings.append(t)
                    print(f"    sample {s+1}/{args.samples}: {t:.1f}s", flush=True)

                if failed or not timings:
                    newton_results[display] = {"median": -1, "stdev": -1, "samples": [], "status": "failed"}
                    continue

                med = statistics.median(timings)
                std = statistics.stdev(timings) if len(timings) > 1 else 0.0
                cv = (std / statistics.mean(timings) * 100) if statistics.mean(timings) > 0 else 0.0
                newton_results[display] = {
                    "median": round(med, 2),
                    "stdev": round(std, 2),
                    "cv": round(cv, 1),
                    "samples": [round(t, 2) for t in timings],
                    "status": "ok",
                }
                print(f"    → median={med:.1f}s stdev={std:.1f}s cv={cv:.1f}%", flush=True)

            all_results[f"newton_{device}"] = newton_results

    # Save results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}", flush=True)


def dev_skip(device: str, devs: str) -> bool:
    return device == "cpu" and devs == "cuda"


if __name__ == "__main__":
    main()
