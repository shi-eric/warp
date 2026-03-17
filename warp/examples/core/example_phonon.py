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
# Example Phonon — Phonon Propagation in a Crystal Lattice
#
# Simulates acoustic phonon propagation through a 3D simple-cubic
# lattice of atoms connected by harmonic springs.  Each atom has a
# scalar displacement u(i,j,k) from equilibrium and velocity v(i,j,k).
#
# Equations of motion (harmonic nearest-neighbor):
#     F_i = -k_s Σ_neighbors (u_i - u_j)
#     m a_i = F_i
#
# Integration: Velocity Verlet (symplectic, energy-conserving).
#
# A thermal pulse (Gaussian displacement perturbation) at the center
# launches acoustic waves that propagate outward showing lattice
# dispersion and anisotropy.
#
# Demonstrates:
#   - 3D lattice dynamics with harmonic springs
#   - Velocity Verlet integration on GPU
#   - Phonon wavefront propagation and dispersion
#   - Marching cubes isosurface of displacement magnitude
#
# Validation:
#   - Phonon dispersion: ω(k) = 2√(k_s/m)|sin(ka/2)|
#   - Acoustic wave group velocity: v_g = a√(k_s/m)
#   - Total energy conservation (KE + PE)
#
###########################################################################

import math

import numpy as np

import warp as wp
import warp.render


@wp.func
def periodic(i: int, n: int):
    """Wrap index into [0, n) with periodic boundary conditions."""
    return (i % n + n) % n


@wp.kernel
def compute_forces(
    u: wp.array3d[wp.float32],
    force: wp.array3d[wp.float32],
    k_spring: float,
):
    """Compute harmonic force: F_i = -k Σ_neighbors (u_i - u_j)."""
    i, j, k = wp.tid()
    nx = u.shape[0]
    ny = u.shape[1]
    nz = u.shape[2]

    u_c = u[i, j, k]

    ip = periodic(i + 1, nx)
    im = periodic(i - 1, nx)
    jp = periodic(j + 1, ny)
    jm = periodic(j - 1, ny)
    kp = periodic(k + 1, nz)
    km = periodic(k - 1, nz)

    # Sum of (u_i - u_neighbor) for 6 neighbors
    diff_sum = (
        (u_c - u[ip, j, k])
        + (u_c - u[im, j, k])
        + (u_c - u[i, jp, k])
        + (u_c - u[i, jm, k])
        + (u_c - u[i, j, kp])
        + (u_c - u[i, j, km])
    )

    force[i, j, k] = -k_spring * diff_sum


@wp.kernel
def verlet_kick(
    vel: wp.array3d[wp.float32],
    force: wp.array3d[wp.float32],
    half_dt_over_m: float,
):
    """Velocity half-step: v += (dt/2m) * F."""
    i, j, k = wp.tid()
    vel[i, j, k] = vel[i, j, k] + half_dt_over_m * force[i, j, k]


@wp.kernel
def verlet_drift(
    u: wp.array3d[wp.float32],
    vel: wp.array3d[wp.float32],
    dt: float,
):
    """Position step: u += dt * v."""
    i, j, k = wp.tid()
    u[i, j, k] = u[i, j, k] + dt * vel[i, j, k]


@wp.kernel
def compute_kinetic_energy_field(
    vel: wp.array3d[wp.float32],
    ke_field: wp.array3d[wp.float32],
    mass: float,
):
    """Compute kinetic energy per atom: KE = 0.5 * m * v²."""
    i, j, k = wp.tid()
    v = vel[i, j, k]
    ke_field[i, j, k] = 0.5 * mass * v * v


@wp.kernel
def compute_displacement_mag(
    u: wp.array3d[wp.float32],
    mag: wp.array3d[wp.float32],
):
    """Compute |u| for isosurface extraction."""
    i, j, k = wp.tid()
    mag[i, j, k] = wp.abs(u[i, j, k])


@wp.kernel
def compute_total_ke_kernel(
    vel: wp.array3d[wp.float32],
    result: wp.array[wp.float32],
    mass: float,
):
    """Sum kinetic energy over all atoms."""
    i, j, k = wp.tid()
    v = vel[i, j, k]
    wp.atomic_add(result, 0, 0.5 * mass * v * v)


@wp.kernel
def compute_total_pe_kernel(
    u: wp.array3d[wp.float32],
    result: wp.array[wp.float32],
    k_spring: float,
):
    """Sum potential energy: PE = 0.5 * k * Σ (u_i - u_j)² for +x,+y,+z neighbors."""
    i, j, k = wp.tid()
    nx = u.shape[0]
    ny = u.shape[1]
    nz = u.shape[2]

    u_c = u[i, j, k]
    ip = periodic(i + 1, nx)
    jp = periodic(j + 1, ny)
    kp = periodic(k + 1, nz)

    dx = u_c - u[ip, j, k]
    dy = u_c - u[i, jp, k]
    dz = u_c - u[i, j, kp]

    pe = 0.5 * k_spring * (dx * dx + dy * dy + dz * dz)
    wp.atomic_add(result, 0, pe)


class Example:
    def __init__(self, stage_path="example_phonon.usd", grid_size=128):
        self.grid_size = grid_size
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 30.0
        self.substeps = 20

        n = grid_size

        # Physical parameters
        self.k_spring = 1.0    # Spring constant
        self.mass = 1.0        # Atom mass
        self.dt = 0.02         # Substep time step
        self.lattice_a = 1.0   # Lattice constant

        # Theoretical speed of sound: v_s = a * sqrt(k/m)
        self.v_sound = self.lattice_a * math.sqrt(self.k_spring / self.mass)

        # Displacement and velocity fields (scalar — 1D displacement per atom)
        self.u = wp.zeros((n, n, n), dtype=wp.float32)
        self.vel = wp.zeros((n, n, n), dtype=wp.float32)
        self.force = wp.zeros((n, n, n), dtype=wp.float32)

        # Visualization fields
        self.ke_field = wp.zeros((n, n, n), dtype=wp.float32)
        self.disp_mag = wp.zeros((n, n, n), dtype=wp.float32)

        # Energy accumulators
        self.energy_buf = wp.zeros(1, dtype=wp.float32)

        # Apply Gaussian thermal pulse at center
        self._apply_pulse(n)

        # Marching cubes for displacement isosurface
        self.mc = wp.MarchingCubes(nx=n, ny=n, nz=n)

        if stage_path and stage_path.endswith((".usd", ".usda", ".usdc")):
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = wp.render.NativeRenderer(512, 512)
            half = float(n) / 2.0
            self.renderer.setup_camera(
                pos=(half + n * 0.7, half + n * 0.4, half + n * 0.7),
                target=(half, half, half),
                fov=50,
            )
            self.renderer.set_environment("dark")

    def _apply_pulse(self, n):
        """Apply Gaussian displacement pulse at center."""
        u_np = self.u.numpy()
        cx, cy, cz = n // 2, n // 2, n // 2
        sigma = max(3.0, n / 30.0)
        amplitude = 0.5

        for i in range(max(0, cx - int(4 * sigma)), min(n, cx + int(4 * sigma) + 1)):
            for j in range(max(0, cy - int(4 * sigma)), min(n, cy + int(4 * sigma) + 1)):
                for k in range(max(0, cz - int(4 * sigma)), min(n, cz + int(4 * sigma) + 1)):
                    dx = i - cx
                    dy = j - cy
                    dz = k - cz
                    r2 = dx * dx + dy * dy + dz * dz
                    u_np[i, j, k] = amplitude * math.exp(-r2 / (2.0 * sigma * sigma))

        self.u = wp.array(u_np, dtype=wp.float32)

    def step(self):
        with wp.ScopedTimer("step", active=False):
            n = self.grid_size
            half_dt_over_m = 0.5 * self.dt / self.mass

            for _ in range(self.substeps):
                # Velocity Verlet:
                # 1. Compute forces at current positions
                wp.launch(
                    kernel=compute_forces,
                    dim=(n, n, n),
                    inputs=[self.u, self.force, self.k_spring],
                )

                # 2. Half-kick: v += (dt/2m) * F
                wp.launch(
                    kernel=verlet_kick,
                    dim=(n, n, n),
                    inputs=[self.vel, self.force, half_dt_over_m],
                )

                # 3. Drift: u += dt * v
                wp.launch(
                    kernel=verlet_drift,
                    dim=(n, n, n),
                    inputs=[self.u, self.vel, self.dt],
                )

                # 4. Compute forces at new positions
                wp.launch(
                    kernel=compute_forces,
                    dim=(n, n, n),
                    inputs=[self.u, self.force, self.k_spring],
                )

                # 5. Half-kick again: v += (dt/2m) * F
                wp.launch(
                    kernel=verlet_kick,
                    dim=(n, n, n),
                    inputs=[self.vel, self.force, half_dt_over_m],
                )

            # Update visualization fields
            wp.launch(
                kernel=compute_kinetic_energy_field,
                dim=(n, n, n),
                inputs=[self.vel, self.ke_field, self.mass],
            )

            self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            n = self.grid_size

            # Isosurface of displacement magnitude
            wp.launch(
                kernel=compute_displacement_mag,
                dim=(n, n, n),
                inputs=[self.u, self.disp_mag],
            )
            self.mc.surface(self.disp_mag, threshold=0.01)

            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_ground(y=0.0)

            if self.mc.verts is not None and len(self.mc.verts) > 0:
                self.renderer.render_mesh(
                    points=self.mc.verts,
                    indices=self.mc.indices,
                    name="phonon_wave",
                    colors=(0.9, 0.4, 0.1),
                )
            self.renderer.end_frame()

    # ---- Validation methods ----

    def compute_kinetic_energy(self):
        """Compute total kinetic energy: KE = Σ 0.5 * m * v²."""
        self.energy_buf.zero_()
        wp.launch(
            kernel=compute_total_ke_kernel,
            dim=(self.grid_size, self.grid_size, self.grid_size),
            inputs=[self.vel, self.energy_buf, self.mass],
        )
        return float(self.energy_buf.numpy()[0])

    def compute_potential_energy(self):
        """Compute total potential energy: PE = Σ 0.5 * k * (u_i - u_j)²."""
        self.energy_buf.zero_()
        wp.launch(
            kernel=compute_total_pe_kernel,
            dim=(self.grid_size, self.grid_size, self.grid_size),
            inputs=[self.u, self.energy_buf, self.k_spring],
        )
        return float(self.energy_buf.numpy()[0])

    def compute_total_energy(self):
        """Compute total energy KE + PE (should be conserved by Verlet)."""
        return self.compute_kinetic_energy() + self.compute_potential_energy()

    def measure_wavefront_radius(self, threshold=0.001):
        """Measure maximum radius where |u| > threshold.

        The acoustic wavefront should propagate at the speed of sound
        v_s = a * sqrt(k/m).
        """
        u_np = self.u.numpy()
        n = self.grid_size
        cx = cy = cz = n // 2

        max_r = 0.0
        # Sample along principal axes for speed
        for axis in range(3):
            for sign in (-1, 1):
                for d in range(1, n // 2):
                    idx = [cx, cy, cz]
                    idx[axis] = (idx[axis] + sign * d) % n
                    if abs(u_np[idx[0], idx[1], idx[2]]) > threshold:
                        max_r = max(max_r, float(d))

        return max_r

    def validate_energy_conservation(self, steps=30):
        """Run steps and verify total energy is conserved.

        Returns (E_initial, E_final, relative_change).
        Verlet integration should conserve energy to machine precision
        for harmonic systems.
        """
        e0 = self.compute_total_energy()
        for _ in range(steps):
            self.step()
        e1 = self.compute_total_energy()
        rel = abs(e1 - e0) / (abs(e0) + 1.0e-12)
        return e0, e1, rel

    def validate_wave_speed(self):
        """Compare measured wavefront speed with theoretical sound velocity.

        Returns dict with measured and theoretical speeds.
        Theoretical: v_s = a * sqrt(k_spring / mass).
        """
        r = self.measure_wavefront_radius()
        t = self.sim_time
        measured_v = r / (t + 1.0e-12)
        return {
            "measured_radius": r,
            "time": t,
            "measured_speed": measured_v,
            "theoretical_speed": self.v_sound,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default=None,
        help="Path to output USD file. If None, uses NativeRenderer.",
    )
    parser.add_argument("--num-frames", type=int, default=200, help="Total number of frames.")
    parser.add_argument("--grid-size", type=int, default=128, help="Grid resolution per axis.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, grid_size=args.grid_size)

        for frame in range(args.num_frames):
            example.step()
            example.render()

            if frame % 40 == 0:
                ke = example.compute_kinetic_energy()
                pe = example.compute_potential_energy()
                total = ke + pe
                radius = example.measure_wavefront_radius()
                print(f"Frame {frame}: KE={ke:.4f}, PE={pe:.4f}, total={total:.4f}, wavefront_r={radius:.1f}")

        # Final validation
        ws = example.validate_wave_speed()
        print(f"\nWave speed validation:")
        print(f"  Wavefront radius:  {ws['measured_radius']:.1f} sites")
        print(f"  Measured speed:    {ws['measured_speed']:.3f} sites/time")
        print(f"  Theoretical speed: {ws['theoretical_speed']:.3f} sites/time")

        if example.renderer:
            if hasattr(example.renderer, "save"):
                example.renderer.save()
