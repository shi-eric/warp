#!/usr/bin/env python3
"""Generate comparison bar charts for compile guard benchmark data.

Usage:
    python _plot_compile_times.py
"""

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


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

# Isolated kernels (5 samples each)
isolated_kernels = {
    "labels": [
<<<<<<< Updated upstream
        "Scalar\nonly",
        "Vector\nmath",
        "Noise +\nrandom",
        "Mat + quat\n+ transform",
        "Volume\nsampling",
        "Mesh\nqueries",
=======
        "Scalar",
        "Vec",
        "Noise + Rand",
        "Mat + Quat",
        "Volume",
        "Mesh",
>>>>>>> Stashed changes
    ],
    "cpu_main": [1.617, 1.639, 1.663, 1.706, 1.721, 1.815],
    "cpu_branch": [0.329, 0.380, 0.432, 0.536, 0.611, 0.824],
    "cuda_main": [0.410, 0.447, 0.421, 0.492, 0.775, 1.025],
    "cuda_branch": [0.146, 0.192, 0.190, 0.288, 0.614, 0.912],
}

# Warp FEM examples — compile time only (3 samples each)
fem_examples = {
    "labels": [
<<<<<<< Updated upstream
        "navier\nstokes",
        "stokes",
        "deformed\ngeometry",
        "diffusion\n3d",
        "convection\ndiffusion",
=======
        "Navier-Stokes",
        "Stokes Flow",
        "Deformed Geom.",
        "Diffusion 3D",
        "Conv.-Diffusion",
>>>>>>> Stashed changes
    ],
    "cpu_main": [67.1, 55.6, 50.6, 42.7, 31.6],
    "cpu_branch": [22.6, 17.8, 16.1, 13.1, 9.4],
    "cuda_main": [23.7, 19.2, 19.9, 15.2, 11.0],
    "cuda_branch": [15.7, 12.0, 13.4, 9.2, 6.8],
}


# --------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------


def plot_comparison(ax, labels, main_vals, branch_vals, title, ylabel="Compile time (s)"):
    """Grouped bar chart: main vs branch."""
    x = np.arange(len(labels))

    rects1 = ax.bar(
        x - BAR_WIDTH / 2,
        main_vals,
        BAR_WIDTH,
        label="main (before)",
        color=COLOR_MAIN,
    )
    rects2 = ax.bar(
        x + BAR_WIDTH / 2,
        branch_vals,
        BAR_WIDTH,
        label="This branch (after)",
        color=COLOR_BRANCH,
    )

<<<<<<< Updated upstream
    ax.set_xticks(x, labels)
=======
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=12)
>>>>>>> Stashed changes
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0, top=1.25 * max(main_vals))
    ax.grid(axis="y", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    # Add speedup labels above the "after" bars
    for i, (m, b) in enumerate(zip(main_vals, branch_vals)):
        speedup = m / b
        ax.annotate(
            f"{speedup:.1f}x",
            xy=(x[i] + BAR_WIDTH / 2, b),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=COLOR_BRANCH,
        )

    ax.legend(loc="upper right", fontsize=12)


# --------------------------------------------------------------------------
# Figure 1: Isolated kernels (CPU + CUDA side by side)
# --------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

plot_comparison(
    ax1,
    isolated_kernels["labels"],
    isolated_kernels["cpu_main"],
    isolated_kernels["cpu_branch"],
    "CPU (Clang JIT) — Isolated Kernels",
)

plot_comparison(
    ax2,
    isolated_kernels["labels"],
    isolated_kernels["cuda_main"],
    isolated_kernels["cuda_branch"],
    "CUDA (NVRTC) — Isolated Kernels",
)

fig.suptitle("Cold-Compile Time by Feature Category", fontsize=18, fontweight="bold", y=1.02)
fig.tight_layout()

fname1 = f"compile_guards_isolated_kernels_{datetime.now():%Y_%m_%d}.png"
print(f"Saving {fname1}")
fig.savefig(fname1, facecolor="w", dpi=150, bbox_inches="tight")

# --------------------------------------------------------------------------
# Figure 2: FEM examples (CPU + CUDA side by side)
# --------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

plot_comparison(
    ax1,
    fem_examples["labels"],
    fem_examples["cpu_main"],
    fem_examples["cpu_branch"],
    "CPU (Clang JIT) — Warp FEM Examples",
)

plot_comparison(
    ax2,
    fem_examples["labels"],
    fem_examples["cuda_main"],
    fem_examples["cuda_branch"],
    "CUDA (NVRTC) — Warp FEM Examples",
)

fig.suptitle("Cold-Compile Time for FEM Examples", fontsize=18, fontweight="bold", y=1.02)
fig.tight_layout()

fname2 = f"compile_guards_fem_examples_{datetime.now():%Y_%m_%d}.png"
print(f"Saving {fname2}")
fig.savefig(fname2, facecolor="w", dpi=150, bbox_inches="tight")

# --------------------------------------------------------------------------
<<<<<<< Updated upstream
# Figure 3: Combined overview — CPU speedups only (single chart)
# --------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 7))

all_labels = (
    [f"kern:\n{l}" for l in isolated_kernels["labels"]]
    + [""]  # spacer
    + [f"fem:\n{l}" for l in fem_examples["labels"]]
)
all_main = isolated_kernels["cpu_main"] + [0] + fem_examples["cpu_main"]
all_branch = isolated_kernels["cpu_branch"] + [0] + fem_examples["cpu_branch"]

x = np.arange(len(all_labels))

rects1 = ax.bar(x - BAR_WIDTH / 2, all_main, BAR_WIDTH, label="main (before)", color=COLOR_MAIN)
rects2 = ax.bar(x + BAR_WIDTH / 2, all_branch, BAR_WIDTH, label="This branch (after)", color=COLOR_BRANCH)

# Speedup annotations
for i, (m, b) in enumerate(zip(all_main, all_branch)):
    if m > 0 and b > 0:
        speedup = m / b
        ax.annotate(
            f"{speedup:.1f}x",
            xy=(x[i] + BAR_WIDTH / 2, b),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=COLOR_BRANCH,
        )

ax.set_xticks(x, all_labels, fontsize=10)
ax.set_ylabel("CPU compile time (s)")
ax.set_title("CPU Cold-Compile Time: main vs. Robust Compile Guards", fontsize=16, fontweight="bold")
ax.set_ylim(bottom=0, top=1.2 * max(all_main))
ax.grid(axis="y", linewidth=0.5, alpha=0.7)
ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=12)

fig.tight_layout()

fname3 = f"compile_guards_cpu_overview_{datetime.now():%Y_%m_%d}.png"
=======
# Figure 3: Newton examples — CPU + CUDA side by side
# --------------------------------------------------------------------------

newton_examples = {
    "labels": [
        "Robot UR10",
        "Robot Allegro",
        "Robot H1",
        "Robot G1",
        "Basic Shapes",
        "Cloth Hanging",
        "Softbody",
        "Diffsim Ball",
    ],
    # CPU compile time (median, 3 samples)
    "cpu_main":   [73.0, 96.3, 90.7, 87.8, 36.8, 27.8, 27.6, 13.6],
    "cpu_branch": [33.6, 47.5, 45.4, 44.5, 23.1, 18.1, 17.9, 7.8],
    # CUDA compile time (median, 3 samples)
    "cuda_main":   [51.1, 87.9, 88.6, 95.3, 52.7, 46.8, 37.4, 10.6],
    "cuda_branch": [43.1, 77.7, 78.2, 85.8, 49.4, 43.3, 34.4, 9.8],
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

plot_comparison(
    ax1,
    newton_examples["labels"],
    newton_examples["cpu_main"],
    newton_examples["cpu_branch"],
    "CPU (Clang JIT) — Newton Examples",
)

plot_comparison(
    ax2,
    newton_examples["labels"],
    newton_examples["cuda_main"],
    newton_examples["cuda_branch"],
    "CUDA (NVRTC) — Newton Examples",
)

fig.suptitle("Cold-Compile Time for Newton Examples", fontsize=18, fontweight="bold", y=1.02)
fig.tight_layout()

fname3 = f"compile_guards_newton_examples_{datetime.now():%Y_%m_%d}.png"
>>>>>>> Stashed changes
print(f"Saving {fname3}")
fig.savefig(fname3, facecolor="w", dpi=150, bbox_inches="tight")

print("\nDone. Generated 3 figures.")
