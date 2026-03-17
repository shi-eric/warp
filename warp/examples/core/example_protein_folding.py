# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Protein Folding (Coarse-Grained)
#
# Simulates a coarse-grained protein chain collapsing into a compact
# structure. Each bead represents an amino acid residue, classified as
# hydrophobic (H) or polar (P). The HP model drives folding:
# hydrophobic beads attract each other, causing the chain to collapse
# into a globular shape with a hydrophobic core.
#
# The chain uses:
#   - Bond springs (sequential beads)
#   - Angle springs (maintain chain stiffness)
#   - Lennard-Jones for non-bonded interactions
#   - Langevin thermostat for thermal fluctuations
#
# Demonstrates:
#   - 1D chain topology with spring constraints
#   - Mixed attractive/repulsive interactions (HP model)
#   - Langevin dynamics (stochastic ODE integration)
#   - Protein-like structure emergence from simple rules
#
###########################################################################

import numpy as np

import warp as wp
import warp.render


# Residue types
HYDROPHOBIC = int(0)
POLAR = int(1)


@wp.kernel
def compute_forces(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
    residue_type: wp.array[wp.int32],
    num_beads: int,
    bond_k: float,
    bond_r0: float,
    angle_k: float,
    lj_epsilon_hh: float,
    lj_epsilon_hp: float,
    lj_epsilon_pp: float,
    lj_sigma: float,
    lj_cutoff: float,
    collapse_strength: float,
    com_x: float,
    com_y: float,
    com_z: float,
    friction: float,
    temperature: float,
    seed: int,
):
    i = wp.tid()
    pi = positions[i]
    f = wp.vec3(0.0, 0.0, 0.0)

    state = wp.rand_init(seed, i)

    # Bond springs (i-1, i) and (i, i+1)
    if i > 0:
        d = pi - positions[i - 1]
        r = wp.length(d)
        if r > 1.0e-6:
            f = f - d / r * bond_k * (r - bond_r0)

    if i < num_beads - 1:
        d = pi - positions[i + 1]
        r = wp.length(d)
        if r > 1.0e-6:
            f = f - d / r * bond_k * (r - bond_r0)

    # Angle springs (i-1, i, i+1)
    if 0 < i < num_beads - 1:
        a = positions[i - 1] - pi
        b = positions[i + 1] - pi
        la = wp.length(a)
        lb = wp.length(b)
        if la > 1.0e-6 and lb > 1.0e-6:
            cos_theta = wp.dot(a, b) / (la * lb)
            cos_theta = wp.clamp(cos_theta, -0.99, 0.99)
            # Restoring torque toward 109.5 degrees (tetrahedral)
            target_cos = -0.33  # cos(109.5°)
            # Simple harmonic angle potential
            f = f + (a / la + b / lb) * angle_k * (cos_theta - target_cos)

    # Non-bonded Lennard-Jones (skip bonded neighbors)
    sigma6 = lj_sigma * lj_sigma * lj_sigma * lj_sigma * lj_sigma * lj_sigma
    cutoff_sq = lj_cutoff * lj_cutoff
    ti = residue_type[i]

    for j in range(num_beads):
        # Skip self and bonded neighbors
        if j == i or j == i - 1 or j == i + 1:
            continue

        d = pi - positions[j]
        r_sq = wp.dot(d, d)

        if r_sq < cutoff_sq and r_sq > 0.1:
            tj = residue_type[j]

            # Select epsilon based on residue pair
            eps = lj_epsilon_pp
            if ti == HYDROPHOBIC and tj == HYDROPHOBIC:
                eps = lj_epsilon_hh  # Strong attraction
            elif ti != tj:
                eps = lj_epsilon_hp  # Weak

            r2_inv = 1.0 / r_sq
            r6_inv = r2_inv * r2_inv * r2_inv
            s6_r6 = sigma6 * r6_inv

            force_mag = 24.0 * eps * (2.0 * s6_r6 * s6_r6 - s6_r6) * r2_inv
            f = f + d * force_mag

    # Langevin thermostat: friction + random force
    # Collapse force toward center of mass (mimics solvent pressure)
    com = wp.vec3(com_x, com_y, com_z)
    to_com = com - pi
    f = f + to_com * collapse_strength

    f = f - velocities[i] * friction

    # Gaussian random force (Box-Muller)
    u1 = wp.max(wp.randf(state), 1.0e-6)
    u2 = wp.randf(state)
    gauss = wp.sqrt(-2.0 * wp.log(u1)) * wp.cos(6.2832 * u2)
    noise_strength = wp.sqrt(2.0 * friction * temperature)
    f = f + wp.vec3(gauss, 0.0, 0.0) * noise_strength

    u1 = wp.max(wp.randf(state), 1.0e-6)
    u2 = wp.randf(state)
    gauss = wp.sqrt(-2.0 * wp.log(u1)) * wp.cos(6.2832 * u2)
    f = f + wp.vec3(0.0, gauss, 0.0) * noise_strength

    u1 = wp.max(wp.randf(state), 1.0e-6)
    u2 = wp.randf(state)
    gauss = wp.sqrt(-2.0 * wp.log(u1)) * wp.cos(6.2832 * u2)
    f = f + wp.vec3(0.0, 0.0, gauss) * noise_strength

    forces[i] = f


@wp.kernel
def integrate(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
    dt: float,
):
    tid = wp.tid()
    velocities[tid] = velocities[tid] + forces[tid] * dt
    positions[tid] = positions[tid] + velocities[tid] * dt


class Example:
    def __init__(self, stage_path="example_protein_folding.usd", num_beads=2000):
        self.num_beads = num_beads
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.substeps = 20
        self.seed = 0

        # Generate HP sequence (alternating blocks with some randomness)
        rng = np.random.default_rng(42)
        residue_types = np.zeros(num_beads, dtype=np.int32)
        # Create hydrophobic core tendency: middle residues are more hydrophobic
        for i in range(num_beads):
            # Sinusoidal pattern with noise
            p_hydrophobic = 0.5 + 0.3 * np.sin(2 * np.pi * i / 15.0)
            residue_types[i] = HYDROPHOBIC if rng.random() < p_hydrophobic else POLAR

        self.residue_type = wp.array(residue_types, dtype=wp.int32)

        # Initialize as a loose coiled coil (3D helix of helices)
        bond_length = 1.0
        positions = np.zeros((num_beads, 3), dtype=np.float32)
        # Inner helix: tight turns
        inner_r = 2.0
        inner_rise = 0.5
        inner_bpt = 6  # beads per turn
        # Outer supercoil
        outer_r = num_beads * inner_rise / (2.0 * np.pi * 4.0)  # ~4 supercoil turns
        outer_bpt = num_beads / 4.0

        for i in range(num_beads):
            # Inner helix angle
            a_inner = 2.0 * np.pi * i / inner_bpt
            # Outer supercoil angle
            a_outer = 2.0 * np.pi * i / outer_bpt

            # Position on supercoil + inner helix offset
            cx = outer_r * np.cos(a_outer)
            cy = outer_r * np.sin(a_outer)
            cz = i * inner_rise * 0.1  # Slight rise

            positions[i] = [
                cx + inner_r * np.cos(a_inner),
                cz,
                cy + inner_r * np.sin(a_inner),
            ]

        # Center at origin
        positions -= positions.mean(axis=0)

        self.positions = wp.array(positions, dtype=wp.vec3)
        self.velocities = wp.zeros(num_beads, dtype=wp.vec3)
        self.forces = wp.zeros(num_beads, dtype=wp.vec3)

        # Force field parameters
        self.bond_k = 200.0
        self.bond_r0 = bond_length
        self.angle_k = 3.0
        self.lj_epsilon_hh = 5.0   # Strong H-H attraction
        self.lj_epsilon_hp = 1.0   # Moderate H-P
        self.lj_epsilon_pp = 0.5   # Weak P-P
        self.lj_sigma = 1.2
        self.lj_cutoff = 8.0
        self.friction = 10.0
        self.temperature = 0.5
        self.collapse_strength = 2.0  # Mimics solvent pressure

        if stage_path and stage_path.endswith((".usd", ".usda", ".usdc")):
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = wp.render.NativeRenderer(512, 512)
            self.renderer.setup_camera(pos=(30, 15, 30), target=(0, 0, 0), fov=50)

    def step(self):
        with wp.ScopedTimer("step", active=False):
            dt = self.frame_dt / self.substeps

            for _ in range(self.substeps):
                self.seed += 1
                self.forces.zero_()

                # Compute center of mass for collapse force
                pos_np = self.positions.numpy()
                com = pos_np.mean(axis=0)

                wp.launch(
                    kernel=compute_forces,
                    dim=self.num_beads,
                    inputs=[
                        self.positions, self.velocities, self.forces,
                        self.residue_type, self.num_beads,
                        self.bond_k, self.bond_r0, self.angle_k,
                        self.lj_epsilon_hh, self.lj_epsilon_hp, self.lj_epsilon_pp,
                        self.lj_sigma, self.lj_cutoff,
                        self.collapse_strength,
                        float(com[0]), float(com[1]), float(com[2]),
                        self.friction, self.temperature, self.seed,
                    ],
                )

                wp.launch(
                    kernel=integrate,
                    dim=self.num_beads,
                    inputs=[self.positions, self.velocities, self.forces, dt],
                )

            self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            pos = self.positions.numpy()
            types = self.residue_type.numpy()
            n = len(pos)

            # Rainbow gradient along chain (N→C terminus)
            colors = np.zeros((n, 3), dtype=np.float32)
            for i in range(n):
                t = float(i) / max(n - 1, 1)
                # Smooth rainbow: blue → cyan → green → yellow → red
                if t < 0.25:
                    s = t / 0.25
                    colors[i] = [0.1, 0.2 + 0.6 * s, 0.9 - 0.2 * s]
                elif t < 0.5:
                    s = (t - 0.25) / 0.25
                    colors[i] = [0.1 + 0.5 * s, 0.8, 0.7 - 0.5 * s]
                elif t < 0.75:
                    s = (t - 0.5) / 0.25
                    colors[i] = [0.6 + 0.3 * s, 0.8 - 0.3 * s, 0.2]
                else:
                    s = (t - 0.75) / 0.25
                    colors[i] = [0.9, 0.5 - 0.3 * s, 0.2 - 0.1 * s]

                # Desaturate polar residues slightly
                if types[i] == POLAR:
                    colors[i] = colors[i] * 0.7 + 0.3

            # Hydrophobic beads slightly larger
            radii = np.where(types == HYDROPHOBIC, 0.55, 0.4).astype(np.float32)

            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_points(
                name="chain",
                points=pos,
                radius=radii,
                colors=colors,
            )
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default=None,
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=500, help="Total number of frames.")
    parser.add_argument("--num-beads", type=int, default=2000, help="Number of amino acid beads.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_beads=args.num_beads)

        for i in range(args.num_frames):
            example.step()
            example.render()

            if i % 100 == 0:
                pos = example.positions.numpy()
                rg = np.sqrt(np.mean(np.sum((pos - pos.mean(axis=0))**2, axis=1)))
                print(f"Frame {i}: Rg={rg:.2f}")

        if example.renderer:
            if hasattr(example.renderer, 'save'):
                example.renderer.save()
            if hasattr(example.renderer, 'save_image'):
                example.renderer.save_image("example_protein_folding.png")
                print("Saved example_protein_folding.png")
