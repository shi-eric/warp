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
# Example Lennard-Jones
#
# Simulates atoms interacting via the Lennard-Jones 6-12 potential,
# the standard model for noble gas molecular dynamics. Particles
# attract at long range and repel at short range, forming clusters,
# liquids, or gases depending on temperature.
#
# This implementation uses a simple O(N²) all-pairs force calculation
# suitable for small-to-medium systems. A cutoff distance reduces
# unnecessary computations.
#
# Demonstrates:
#   - Lennard-Jones 6-12 force calculation
#   - Velocity Verlet time integration
#   - Periodic boundary conditions
#   - Temperature control via velocity rescaling
#   - O(N²) pairwise interaction pattern
#
###########################################################################

import numpy as np

import warp as wp
import warp.render


@wp.kernel
def compute_forces(
    positions: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
    num_atoms: int,
    epsilon: float,
    sigma: float,
    cutoff: float,
    box_size: float,
):
    """Compute Lennard-Jones forces with cutoff and minimum image convention."""
    i = wp.tid()

    f = wp.vec3(0.0, 0.0, 0.0)
    pi = positions[i]

    cutoff_sq = cutoff * cutoff
    sigma6 = sigma * sigma * sigma * sigma * sigma * sigma

    for j in range(num_atoms):
        if i == j:
            continue

        # Minimum image convention for periodic boundaries
        dx = pi[0] - positions[j][0]
        dy = pi[1] - positions[j][1]
        dz = pi[2] - positions[j][2]

        # Wrap to nearest image
        half_box = box_size * 0.5
        if dx > half_box:
            dx = dx - box_size
        elif dx < -half_box:
            dx = dx + box_size
        if dy > half_box:
            dy = dy - box_size
        elif dy < -half_box:
            dy = dy + box_size
        if dz > half_box:
            dz = dz - box_size
        elif dz < -half_box:
            dz = dz + box_size

        r_sq = dx * dx + dy * dy + dz * dz

        if r_sq < cutoff_sq and r_sq > 0.01:
            r2_inv = 1.0 / r_sq
            r6_inv = r2_inv * r2_inv * r2_inv
            s6_r6 = sigma6 * r6_inv

            # F = 24 * epsilon * (2 * (sigma/r)^12 - (sigma/r)^6) / r^2
            force_mag = 24.0 * epsilon * (2.0 * s6_r6 * s6_r6 - s6_r6) * r2_inv

            f = f + wp.vec3(dx * force_mag, dy * force_mag, dz * force_mag)

    forces[i] = f


@wp.kernel
def integrate_verlet(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
    mass_inv: float,
    dt: float,
    box_size: float,
):
    """Velocity Verlet integration with periodic boundary wrapping."""
    tid = wp.tid()

    v = velocities[tid] + forces[tid] * mass_inv * dt * 0.5
    p = positions[tid] + v * dt

    # Periodic boundary conditions
    px = p[0]
    py = p[1]
    pz = p[2]
    if px < 0.0:
        px = px + box_size
    elif px >= box_size:
        px = px - box_size
    if py < 0.0:
        py = py + box_size
    elif py >= box_size:
        py = py - box_size
    if pz < 0.0:
        pz = pz + box_size
    elif pz >= box_size:
        pz = pz - box_size

    positions[tid] = wp.vec3(px, py, pz)
    velocities[tid] = v


@wp.kernel
def update_velocities(
    velocities: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
    mass_inv: float,
    dt: float,
):
    """Second half of velocity Verlet: update velocities with new forces."""
    tid = wp.tid()
    velocities[tid] = velocities[tid] + forces[tid] * mass_inv * dt * 0.5


@wp.kernel
def compute_kinetic_energy(
    velocities: wp.array[wp.vec3],
    ke: wp.array[wp.float32],
):
    tid = wp.tid()
    v = velocities[tid]
    wp.atomic_add(ke, 0, 0.5 * wp.dot(v, v))


class Example:
    def __init__(self, stage_path="example_lennard_jones.usd", num_atoms=1000, temperature=1.0):
        self.num_atoms = num_atoms
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0

        # LJ parameters (reduced units)
        self.epsilon = 1.0
        self.sigma = 1.0
        self.mass = 1.0
        self.mass_inv = 1.0 / self.mass
        self.dt = 0.001  # Reduced time units (smaller for energy conservation)
        self.substeps = 20

        # Box size for desired density (~0.5 in reduced units)
        density = 0.3
        self.box_size = (num_atoms / density) ** (1.0 / 3.0)
        self.cutoff = min(2.5 * self.sigma, self.box_size * 0.5)

        # Initialize on FCC-like lattice
        n_side = int(np.ceil(num_atoms ** (1.0 / 3.0)))
        spacing = self.box_size / n_side

        positions = []
        for ix in range(n_side):
            for iy in range(n_side):
                for iz in range(n_side):
                    if len(positions) >= num_atoms:
                        break
                    positions.append([
                        (ix + 0.5) * spacing,
                        (iy + 0.5) * spacing,
                        (iz + 0.5) * spacing,
                    ])

        positions = np.array(positions[:num_atoms], dtype=np.float32)

        # Maxwell-Boltzmann velocity distribution
        rng = np.random.default_rng(42)
        v_scale = np.sqrt(temperature / self.mass)
        velocities = rng.normal(0, v_scale, (num_atoms, 3)).astype(np.float32)

        # Remove center-of-mass velocity
        velocities -= velocities.mean(axis=0)

        self.positions = wp.array(positions, dtype=wp.vec3)
        self.velocities = wp.array(velocities, dtype=wp.vec3)
        self.forces = wp.zeros(num_atoms, dtype=wp.vec3)
        self.ke_buf = wp.zeros(1, dtype=wp.float32)

        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = None

    def step(self):
        with wp.ScopedTimer("step", active=False):
            n = self.num_atoms

            for _ in range(self.substeps):
                # Compute forces
                self.forces.zero_()
                wp.launch(
                    kernel=compute_forces,
                    dim=n,
                    inputs=[self.positions, self.forces, n, self.epsilon, self.sigma, self.cutoff, self.box_size],
                )

                # Velocity Verlet: half-step velocity + full-step position
                wp.launch(
                    kernel=integrate_verlet,
                    dim=n,
                    inputs=[self.positions, self.velocities, self.forces, self.mass_inv, self.dt, self.box_size],
                )

                # Recompute forces at new positions
                self.forces.zero_()
                wp.launch(
                    kernel=compute_forces,
                    dim=n,
                    inputs=[self.positions, self.forces, n, self.epsilon, self.sigma, self.cutoff, self.box_size],
                )

                # Second half-step velocity update
                wp.launch(
                    kernel=update_velocities,
                    dim=n,
                    inputs=[self.velocities, self.forces, self.mass_inv, self.dt],
                )

            self.sim_time += self.frame_dt

    def get_temperature(self):
        """Compute instantaneous kinetic temperature: T = 2*KE / (3*N*kB)."""
        self.ke_buf.zero_()
        wp.launch(
            kernel=compute_kinetic_energy,
            dim=self.num_atoms,
            inputs=[self.velocities, self.ke_buf],
        )
        ke = self.ke_buf.numpy()[0]
        return 2.0 * ke / (3.0 * self.num_atoms)

    def radial_distribution_function(self, num_bins=100, r_max=None):
        """Compute the radial distribution function g(r).

        g(r) is the probability of finding a particle at distance r,
        normalized by the ideal gas expectation. Peaks in g(r) reveal
        the liquid/solid structure (first peak at ~σ, second at ~2σ).
        """
        if r_max is None:
            r_max = self.box_size * 0.5

        pos = self.positions.numpy()
        n = len(pos)
        dr = r_max / num_bins
        hist = np.zeros(num_bins)

        # All-pairs distance histogram (with minimum image convention)
        for i in range(n):
            diff = pos - pos[i]
            # Minimum image
            diff = diff - self.box_size * np.round(diff / self.box_size)
            dists = np.sqrt(np.sum(diff**2, axis=1))
            dists[i] = r_max + 1.0  # Exclude self
            bins = (dists / dr).astype(int)
            mask = bins < num_bins
            np.add.at(hist, bins[mask], 1)

        # Normalize: g(r) = hist / (N * 4πr²dr * ρ)
        density = n / (self.box_size**3)
        r = (np.arange(num_bins) + 0.5) * dr
        shell_vol = 4.0 * np.pi * r**2 * dr
        g_r = hist / (n * shell_vol * density + 1e-10)

        return r, g_r

    def compute_total_energy(self):
        """Compute total energy (kinetic + potential) for conservation check.

        In a microcanonical (NVE) ensemble, total energy should be conserved.
        Drift indicates integrator error.
        """
        # Kinetic energy
        self.ke_buf.zero_()
        wp.launch(
            kernel=compute_kinetic_energy,
            dim=self.num_atoms,
            inputs=[self.velocities, self.ke_buf],
        )
        ke = self.ke_buf.numpy()[0]

        # Potential energy (computed on CPU for simplicity)
        pos = self.positions.numpy()
        n = len(pos)
        pe = 0.0
        sigma6 = self.sigma**6
        for i in range(n):
            diff = pos[i + 1:] - pos[i]
            diff = diff - self.box_size * np.round(diff / self.box_size)
            r2 = np.sum(diff**2, axis=1)
            mask = (r2 < self.cutoff**2) & (r2 > 0.01)
            r2_inv = 1.0 / r2[mask]
            r6_inv = r2_inv**3
            s6r6 = sigma6 * r6_inv
            pe += 4.0 * self.epsilon * np.sum(s6r6**2 - s6r6)

        return ke, pe, ke + pe

    def compute_pressure(self):
        """Compute pressure via the virial theorem.

        P = NkT/V + (1/3V) * Σ r_ij · F_ij

        In reduced LJ units with kB=1, this gives the equation of state.
        """
        pos = self.positions.numpy()
        n = len(pos)
        T = self.get_temperature()

        # Virial sum
        virial = 0.0
        sigma6 = self.sigma**6
        for i in range(n):
            diff = pos[i + 1:] - pos[i]
            diff = diff - self.box_size * np.round(diff / self.box_size)
            r2 = np.sum(diff**2, axis=1)
            mask = (r2 < self.cutoff**2) & (r2 > 0.01)
            r2_inv = 1.0 / r2[mask]
            r6_inv = r2_inv**3
            s6r6 = sigma6 * r6_inv
            # F·r = 24ε(2(σ/r)^12 - (σ/r)^6)
            virial += 24.0 * self.epsilon * np.sum(2.0 * s6r6**2 - s6r6)

        V = self.box_size**3
        P = n * T / V + virial / (3.0 * V)
        return P

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_points(
                points=self.positions.numpy(),
                radius=self.sigma * 0.5,
                name="atoms",
                colors=(0.2, 0.6, 0.9),
            )
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_lennard_jones.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument("--num-atoms", type=int, default=1000, help="Number of atoms.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Initial temperature (reduced units).")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            stage_path=args.stage_path,
            num_atoms=args.num_atoms,
            temperature=args.temperature,
        )

        for i in range(args.num_frames):
            example.step()
            example.render()

            if i % 50 == 0:
                T = example.get_temperature()
                print(f"Frame {i}: T={T:.3f}")

        if example.renderer:
            example.renderer.save()
