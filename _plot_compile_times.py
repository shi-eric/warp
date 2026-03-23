#!/usr/bin/env python3
"""Generate comparison bar charts from benchmark JSON files.

Reads _benchmark_baseline_all.json and _benchmark_branch_all.json,
produces PNG charts for isolated kernels, FEM examples, and Newton examples.

Usage:
    uv run --with matplotlib _plot_compile_times.py
"""

import json
import os
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

mpl.rcParams["figure.figsize"] = (14, 7)
mpl.rcParams["font.size"] = 16
mpl.rcParams["lines.linewidth"] = 2
mpl.rcParams["font.family"] = "sans-serif"

COLOR_MAIN = "#7f7f7f"  # gray — main branch (before)
COLOR_BRANCH = "#76B900"  # NVIDIA green — this branch (after)
BAR_WIDTH = 0.35

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATE_TAG = datetime.now().strftime("%Y_%m_%d")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def load_json(path):
    with open(os.path.join(SCRIPT_DIR, path)) as f:
        return json.load(f)


def get_medians(data, section, examples):
    """Extract median values for a list of example names from a section."""
    section_data = data.get(section, {})
    return [section_data.get(name, {}).get("median", 0) for name in examples]


# --------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------


def plot_comparison(ax, labels, main_vals, branch_vals, title, ylabel="Compile time (s)"):
    """Grouped bar chart: main vs branch."""
    x = np.arange(len(labels))

    ax.bar(x - BAR_WIDTH / 2, main_vals, BAR_WIDTH,
           label="main (before)", color=COLOR_MAIN)
    ax.bar(x + BAR_WIDTH / 2, branch_vals, BAR_WIDTH,
           label="This branch (after)", color=COLOR_BRANCH)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(bottom=0, top=1.25 * max(main_vals) if main_vals else 1)
    ax.grid(axis="y", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    # Speedup labels above "after" bars
    for i, (m, b) in enumerate(zip(main_vals, branch_vals)):
        if m > 0 and b > 0:
            speedup = m / b
            ax.annotate(
                f"{speedup:.1f}x",
                xy=(x[i] + BAR_WIDTH / 2, b),
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=COLOR_BRANCH,
            )

    ax.legend(loc="upper right", fontsize=11)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    baseline = load_json("_benchmark_baseline_all.json")
    branch = load_json("_benchmark_branch_all.json")

    # Also load isolated kernel data if available
    try:
        kern_baseline = load_json("_benchmark_results_kernels.json")
        kern_branch = load_json("_benchmark_results_kernels.json")
        has_kernels = True
    except FileNotFoundError:
        has_kernels = False

    # ==================================================================
    # Figure 1: FEM examples — CPU + CUDA side by side
    # ==================================================================
    # Pick the most representative FEM examples (ordered by CPU baseline time)
    fem_examples = [
        ("fem.stokes_transfer", "Stokes\nTransfer"),
        ("fem.taylor_green", "Taylor-\nGreen"),
        ("fem.navier_stokes", "Navier-\nStokes"),
        ("fem.convection_diffusion_dg", "Conv-Diff\nDG"),
        ("fem.stokes", "Stokes"),
        ("fem.streamlines", "Stream-\nlines"),
        ("fem.deformed_geometry", "Deformed\nGeom."),
        ("fem.diffusion", "Diffusion"),
        ("fem.diffusion_3d", "Diffusion\n3D"),
        ("fem.convection_diffusion", "Conv-\nDiffusion"),
        ("fem.burgers", "Burgers"),
    ]

    fem_keys = [k for k, _ in fem_examples]
    fem_labels = [l for _, l in fem_examples]

    fem_cpu_main = get_medians(baseline, "fem_cpu", fem_keys)
    fem_cpu_branch = get_medians(branch, "fem_cpu", fem_keys)
    fem_cuda_main = get_medians(baseline, "fem_cuda:0", fem_keys)
    fem_cuda_branch = get_medians(branch, "fem_cuda:0", fem_keys)

    # Sort by CPU baseline time descending
    order = sorted(range(len(fem_cpu_main)), key=lambda i: fem_cpu_main[i], reverse=True)
    fem_labels = [fem_labels[i] for i in order]
    fem_cpu_main = [fem_cpu_main[i] for i in order]
    fem_cpu_branch = [fem_cpu_branch[i] for i in order]
    fem_cuda_main = [fem_cuda_main[i] for i in order]
    fem_cuda_branch = [fem_cuda_branch[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    plot_comparison(ax1, fem_labels, fem_cpu_main, fem_cpu_branch,
                    "CPU (Clang JIT) — Warp FEM Examples")
    plot_comparison(ax2, fem_labels, fem_cuda_main, fem_cuda_branch,
                    "CUDA (NVRTC) — Warp FEM Examples")
    fig.suptitle("Cold-Compile Time for FEM Examples", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()

    fname1 = os.path.join(SCRIPT_DIR, f"compile_guards_fem_examples_{DATE_TAG}.png")
    print(f"Saving {fname1}")
    fig.savefig(fname1, facecolor="w", dpi=150, bbox_inches="tight")

    # ==================================================================
    # Figure 2: Newton examples — CPU + CUDA side by side
    # ==================================================================
    # Pick representative Newton examples across categories
    newton_examples = [
        ("newton.robot_anymal_d", "Robot\nANYmal D"),
        ("newton.robot_ur10", "Robot\nUR10"),
        ("newton.robot_cartpole", "Robot\nCartPole"),
        ("newton.selection_articulations", "Selection\nArticulations"),
        ("newton.sensor_contact", "Sensor\nContact"),
        ("newton.cloth_hanging", "Cloth\nHanging"),
        ("newton.cable_twist", "Cable\nTwist"),
        ("newton.basic_shapes", "Basic\nShapes"),
        ("newton.diffsim_drone", "Diffsim\nDrone"),
        ("newton.diffsim_ball", "Diffsim\nBall"),
    ]

    newton_keys = [k for k, _ in newton_examples]
    newton_labels = [l for _, l in newton_examples]

    newton_cpu_main = get_medians(baseline, "newton_cpu", newton_keys)
    newton_cpu_branch = get_medians(branch, "newton_cpu", newton_keys)
    newton_cuda_main = get_medians(baseline, "newton_cuda:0", newton_keys)
    newton_cuda_branch = get_medians(branch, "newton_cuda:0", newton_keys)

    # Sort by CPU baseline descending
    order = sorted(range(len(newton_cpu_main)), key=lambda i: newton_cpu_main[i], reverse=True)
    newton_labels = [newton_labels[i] for i in order]
    newton_cpu_main = [newton_cpu_main[i] for i in order]
    newton_cpu_branch = [newton_cpu_branch[i] for i in order]
    newton_cuda_main = [newton_cuda_main[i] for i in order]
    newton_cuda_branch = [newton_cuda_branch[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    plot_comparison(ax1, newton_labels, newton_cpu_main, newton_cpu_branch,
                    "CPU (Clang JIT) — Newton Examples")
    plot_comparison(ax2, newton_labels, newton_cuda_main, newton_cuda_branch,
                    "CUDA (NVRTC) — Newton Examples")
    fig.suptitle("Cold-Compile Time for Newton Examples", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()

    fname2 = os.path.join(SCRIPT_DIR, f"compile_guards_newton_examples_{DATE_TAG}.png")
    print(f"Saving {fname2}")
    fig.savefig(fname2, facecolor="w", dpi=150, bbox_inches="tight")

    # ==================================================================
    # Figure 3: Newton CUDA-only examples (MPM, contacts, softbody, robots)
    # ==================================================================
    newton_cuda_only = [
        ("newton.robot_panda_hydro", "Panda\nHydro"),
        ("newton.brick_stacking", "Brick\nStacking"),
        ("newton.mpm_twoway_coupling", "MPM\nCoupling"),
        ("newton.nut_bolt_hydro", "Nut-Bolt\nHydro"),
        ("newton.nut_bolt_sdf", "Nut-Bolt\nSDF"),
        ("newton.robot_g1", "Robot G1"),
        ("newton.robot_allegro_hand", "Allegro\nHand"),
        ("newton.robot_h1", "Robot H1"),
        ("newton.cloth_franka", "Cloth\nFranka"),
        ("newton.cloth_poker_cards", "Poker\nCards"),
        ("newton.mpm_granular", "MPM\nGranular"),
        ("newton.softbody_hanging", "Softbody\nHanging"),
    ]

    cuda_keys = [k for k, _ in newton_cuda_only]
    cuda_labels = [l for _, l in newton_cuda_only]

    cuda_main = get_medians(baseline, "newton_cuda:0", cuda_keys)
    cuda_branch = get_medians(branch, "newton_cuda:0", cuda_keys)

    # Sort by baseline descending
    order = sorted(range(len(cuda_main)), key=lambda i: cuda_main[i], reverse=True)
    cuda_labels = [cuda_labels[i] for i in order]
    cuda_main = [cuda_main[i] for i in order]
    cuda_branch = [cuda_branch[i] for i in order]

    fig, ax = plt.subplots(figsize=(16, 7))
    plot_comparison(ax, cuda_labels, cuda_main, cuda_branch,
                    "CUDA (NVRTC) — Newton Heavy Examples")
    fig.suptitle("Cold-Compile Time for Newton CUDA-Only Examples",
                 fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()

    fname3 = os.path.join(SCRIPT_DIR, f"compile_guards_newton_cuda_heavy_{DATE_TAG}.png")
    print(f"Saving {fname3}")
    fig.savefig(fname3, facecolor="w", dpi=150, bbox_inches="tight")

    # ==================================================================
    # Figure 4: FEM optimization examples + heavy solvers (CPU + CUDA)
    # ==================================================================
    fem_heavy = [
        ("fem.elastic_shape_optimization", "Elastic Shape\nOptim."),
        ("fem.darcy_ls_optimization", "Darcy LS\nOptim."),
        ("fem.nonconforming_contact", "Nonconforming\nContact"),
        ("fem.adaptive_grid", "Adaptive\nGrid"),
        ("fem.mixed_elasticity", "Mixed\nElasticity"),
        ("fem.apic_fluid", "APIC\nFluid"),
    ]

    heavy_keys = [k for k, _ in fem_heavy]
    heavy_labels = [l for _, l in fem_heavy]

    heavy_cuda_main = get_medians(baseline, "fem_cuda:0", heavy_keys)
    heavy_cuda_branch = get_medians(branch, "fem_cuda:0", heavy_keys)

    # Filter to only those with data
    valid = [(l, m, b) for l, m, b in zip(heavy_labels, heavy_cuda_main, heavy_cuda_branch) if m > 0 and b > 0]
    if valid:
        heavy_labels, heavy_cuda_main, heavy_cuda_branch = zip(*valid)
        heavy_labels = list(heavy_labels)
        heavy_cuda_main = list(heavy_cuda_main)
        heavy_cuda_branch = list(heavy_cuda_branch)

        # Sort descending
        order = sorted(range(len(heavy_cuda_main)), key=lambda i: heavy_cuda_main[i], reverse=True)
        heavy_labels = [heavy_labels[i] for i in order]
        heavy_cuda_main = [heavy_cuda_main[i] for i in order]
        heavy_cuda_branch = [heavy_cuda_branch[i] for i in order]

        fig, ax = plt.subplots(figsize=(14, 7))
        plot_comparison(ax, heavy_labels, heavy_cuda_main, heavy_cuda_branch,
                        "CUDA (NVRTC) — FEM Heavy / Optimization Examples")
        fig.suptitle("Cold-Compile Time for FEM Optimization Examples",
                     fontsize=18, fontweight="bold", y=1.02)
        fig.tight_layout()

        fname4 = os.path.join(SCRIPT_DIR, f"compile_guards_fem_heavy_{DATE_TAG}.png")
        print(f"Saving {fname4}")
        fig.savefig(fname4, facecolor="w", dpi=150, bbox_inches="tight")

    print("\nDone.")


if __name__ == "__main__":
    main()
