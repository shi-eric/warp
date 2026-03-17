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
# Example Particle-in-Cell (PIC) — Two-Stream Instability
#
# Implements a simplified 3D electrostatic Particle-in-Cell code.
# Two counter-streaming electron beams interact via self-consistent
# electric fields, triggering the classic two-stream instability.
# The beams decelerate, trap particles, and form phase-space vortices.
#
# The PIC cycle:
#   1. Deposit particle charge onto grid (scatter)
#   2. Solve Poisson equation for electric potential (Jacobi)
#   3. Compute electric field from potential gradient
#   4. Interpolate field to particle positions (gather)
#   5. Push particles with Boris-like integrator
#
# Demonstrates:
#   - Particle-grid scatter (charge deposition via atomics)
#   - Poisson solver (Jacobi iteration)
#   - Gradient computation from scalar potential
#   - Particle-grid gather (field interpolation)
#   - Leapfrog particle push
#   - Periodic boundary conditions
#
###########################################################################

import numpy as np

import warp as wp
import warp.render


@wp.kernel
def deposit_charge(
    positions: wp.array[wp.vec3],
    charge: wp.array[wp.float32],
    rho: wp.array3d[wp.float32],
    grid_size: int,
    cell_size: float,
    inv_cell_vol: float,
):
    """Nearest-grid-point charge deposition with volume normalization."""
    tid = wp.tid()

    p = positions[tid]
    q = charge[tid]

    ix = int(p[0] / cell_size) % grid_size
    iy = int(p[1] / cell_size) % grid_size
    iz = int(p[2] / cell_size) % grid_size

    wp.atomic_add(rho, ix, iy, iz, q * inv_cell_vol)


@wp.kernel
def add_background_charge(
    rho: wp.array3d[wp.float32],
    background: float,
):
    """Add uniform neutralizing ion background."""
    i, j, k = wp.tid()
    rho[i, j, k] = rho[i, j, k] + background


@wp.kernel
def poisson_jacobi(
    phi: wp.array3d[wp.float32],
    phi_out: wp.array3d[wp.float32],
    rho: wp.array3d[wp.float32],
    dx2: float,
):
    """One Jacobi iteration for Poisson equation: ∇²φ = -ρ/ε₀."""
    i, j, k = wp.tid()

    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]

    # Periodic neighbors
    ip = (i + 1) % nx
    im = (i - 1 + nx) % nx
    jp = (j + 1) % ny
    jm = (j - 1 + ny) % ny
    kp = (k + 1) % nz
    km = (k - 1 + nz) % nz

    phi_out[i, j, k] = (
        phi[ip, j, k] + phi[im, j, k]
        + phi[i, jp, k] + phi[i, jm, k]
        + phi[i, j, kp] + phi[i, j, km]
        + rho[i, j, k] * dx2
    ) / 6.0


@wp.kernel
def compute_efield(
    phi: wp.array3d[wp.float32],
    ex: wp.array3d[wp.float32],
    ey: wp.array3d[wp.float32],
    ez: wp.array3d[wp.float32],
    inv_2dx: float,
):
    """E = -∇φ via central differences with periodic BC."""
    i, j, k = wp.tid()

    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]

    ip = (i + 1) % nx
    im = (i - 1 + nx) % nx
    jp = (j + 1) % ny
    jm = (j - 1 + ny) % ny
    kp = (k + 1) % nz
    km = (k - 1 + nz) % nz

    ex[i, j, k] = -(phi[ip, j, k] - phi[im, j, k]) * inv_2dx
    ey[i, j, k] = -(phi[i, jp, k] - phi[i, jm, k]) * inv_2dx
    ez[i, j, k] = -(phi[i, j, kp] - phi[i, j, km]) * inv_2dx


@wp.kernel
def push_particles(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    charge: wp.array[wp.float32],
    mass: wp.array[wp.float32],
    ex: wp.array3d[wp.float32],
    ey: wp.array3d[wp.float32],
    ez: wp.array3d[wp.float32],
    grid_size: int,
    cell_size: float,
    domain_size: float,
    dt: float,
):
    """Leapfrog particle push with field interpolation."""
    tid = wp.tid()

    p = positions[tid]
    v = velocities[tid]
    q = charge[tid]
    m = mass[tid]

    # Nearest grid point field interpolation
    ix = int(p[0] / cell_size) % grid_size
    iy = int(p[1] / cell_size) % grid_size
    iz = int(p[2] / cell_size) % grid_size

    e = wp.vec3(ex[ix, iy, iz], ey[ix, iy, iz], ez[ix, iy, iz])

    # Leapfrog: v(n+1/2) = v(n-1/2) + q/m * E * dt
    acc = e * (q / m)
    v = v + acc * dt
    p = p + v * dt

    # Periodic boundaries
    px = p[0]
    py = p[1]
    pz = p[2]

    if px < 0.0:
        px = px + domain_size
    elif px >= domain_size:
        px = px - domain_size
    if py < 0.0:
        py = py + domain_size
    elif py >= domain_size:
        py = py - domain_size
    if pz < 0.0:
        pz = pz + domain_size
    elif pz >= domain_size:
        pz = pz - domain_size

    positions[tid] = wp.vec3(px, py, pz)
    velocities[tid] = v


class Example:
    def __init__(self, stage_path="example_pic.usd", num_particles=200000, grid_size=32):
        self.num_particles = num_particles
        self.grid_size = grid_size
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0

        # Domain
        self.domain_size = 10.0
        self.cell_size = self.domain_size / grid_size
        self.dx2 = self.cell_size * self.cell_size

        # PIC parameters
        self.dt = 0.05
        self.substeps = 3
        self.jacobi_iters = 100
        self.inv_cell_vol = 1.0 / (self.cell_size ** 3)

        n_half = num_particles // 2
        rng = np.random.default_rng(42)

        # Two counter-streaming beams
        positions = rng.uniform(0, self.domain_size, (num_particles, 3)).astype(np.float32)

        velocities = np.zeros((num_particles, 3), dtype=np.float32)
        beam_speed = 3.0
        thermal_speed = 0.1

        # Beam 1: +x direction
        velocities[:n_half, 0] = beam_speed + rng.normal(0, thermal_speed, n_half)
        velocities[:n_half, 1] = rng.normal(0, thermal_speed, n_half)
        velocities[:n_half, 2] = rng.normal(0, thermal_speed, n_half)

        # Beam 2: -x direction
        velocities[n_half:, 0] = -beam_speed + rng.normal(0, thermal_speed, num_particles - n_half)
        velocities[n_half:, 1] = rng.normal(0, thermal_speed, num_particles - n_half)
        velocities[n_half:, 2] = rng.normal(0, thermal_speed, num_particles - n_half)

        # All electrons: charge = -1, mass = 1
        charges = np.full(num_particles, -1.0, dtype=np.float32)
        masses = np.ones(num_particles, dtype=np.float32)

        # Background ion charge density (neutralizing)
        self.background_charge = float(num_particles) * self.inv_cell_vol / float(grid_size ** 3)

        self.positions = wp.array(positions, dtype=wp.vec3)
        self.velocities = wp.array(velocities, dtype=wp.vec3)
        self.charge = wp.array(charges, dtype=wp.float32)
        self.mass = wp.array(masses, dtype=wp.float32)

        # Grid fields
        n = grid_size
        self.rho = wp.zeros((n, n, n), dtype=wp.float32)
        self.phi = wp.zeros((n, n, n), dtype=wp.float32)
        self.phi_tmp = wp.zeros((n, n, n), dtype=wp.float32)
        self.ex = wp.zeros((n, n, n), dtype=wp.float32)
        self.ey = wp.zeros((n, n, n), dtype=wp.float32)
        self.ez = wp.zeros((n, n, n), dtype=wp.float32)

        if stage_path and stage_path.endswith((".usd", ".usda", ".usdc")):
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = wp.render.NativeRenderer(512, 512)
            self.renderer.setup_camera(pos=(12, 8, 12), target=(5, 5, 5), fov=50)
            self.renderer.bg_top = wp.vec3(0.03, 0.03, 0.08)
            self.renderer.bg_bottom = wp.vec3(0.01, 0.01, 0.03)
            self.renderer.shadows = False

    def step(self):
        with wp.ScopedTimer("step", active=False):
            n = self.grid_size

            for _ in range(self.substeps):
                # 1. Deposit charge
                self.rho.zero_()
                wp.launch(
                    kernel=deposit_charge,
                    dim=self.num_particles,
                    inputs=[self.positions, self.charge, self.rho, n, self.cell_size, self.inv_cell_vol],
                )

                # Add neutralizing background
                wp.launch(
                    kernel=add_background_charge,
                    dim=(n, n, n),
                    inputs=[self.rho, self.background_charge],
                )

                # 2. Solve Poisson equation
                self.phi.zero_()
                for _ in range(self.jacobi_iters):
                    wp.launch(
                        kernel=poisson_jacobi,
                        dim=(n, n, n),
                        inputs=[self.phi, self.phi_tmp, self.rho, self.dx2],
                    )
                    self.phi, self.phi_tmp = self.phi_tmp, self.phi

                # 3. Compute electric field
                wp.launch(
                    kernel=compute_efield,
                    dim=(n, n, n),
                    inputs=[self.phi, self.ex, self.ey, self.ez, 0.5 / self.cell_size],
                )

                # 4 & 5. Push particles
                wp.launch(
                    kernel=push_particles,
                    dim=self.num_particles,
                    inputs=[
                        self.positions, self.velocities,
                        self.charge, self.mass,
                        self.ex, self.ey, self.ez,
                        n, self.cell_size, self.domain_size, self.dt,
                    ],
                )

            self.sim_time += self.frame_dt

    def compute_field_energy(self):
        """Compute electrostatic field energy: E_field = ε₀/2 ∫ |E|² dV.

        For the two-stream instability, this energy should grow
        exponentially at the linear growth rate γ ≈ ωp * √3/2 * (v_b/v_th)
        before saturating. Plotting log(E_field) vs time should show
        a linear growth phase.
        """
        ex = self.ex.numpy()
        ey = self.ey.numpy()
        ez = self.ez.numpy()
        e2 = ex**2 + ey**2 + ez**2
        cell_vol = self.cell_size**3
        return 0.5 * np.sum(e2) * cell_vol

    def compute_beam_kinetic_energy(self):
        """Compute kinetic energy decomposed into beam (directed) and thermal.

        KE_beam = 0.5 * m * N * v_beam²  (organized motion)
        KE_thermal = 0.5 * m * Σ(v - v_beam)²  (random motion)

        During the two-stream instability, KE_beam decreases and
        KE_thermal + E_field increases (energy conservation).
        """
        v = self.velocities.numpy()
        n_half = self.num_particles // 2

        # Beam 1 mean velocity
        v1_mean = v[:n_half, 0].mean()
        # Beam 2 mean velocity
        v2_mean = v[n_half:, 0].mean()

        ke_beam = 0.5 * (n_half * v1_mean**2 + (self.num_particles - n_half) * v2_mean**2)
        ke_thermal = 0.5 * np.sum((v[:n_half, 0] - v1_mean)**2) + 0.5 * np.sum((v[n_half:, 0] - v2_mean)**2)
        ke_thermal += 0.5 * np.sum(v[:, 1]**2) + 0.5 * np.sum(v[:, 2]**2)

        return ke_beam, ke_thermal, v1_mean, v2_mean

    def phase_space_distribution(self, num_bins_x=64, num_bins_v=64):
        """Compute phase-space density f(x, vx) for the x-direction.

        For the two-stream instability, this should show:
        - Initially: two horizontal bands at vx = ±v_beam
        - During instability: bands bending into phase-space vortices
        - Saturated: trapped particle orbits (phase-space holes)
        """
        pos = self.positions
        vel = self.velocities.numpy()

        x = pos[:, 0]
        vx = vel[:, 0]

        x_edges = np.linspace(0, self.domain_size, num_bins_x + 1)
        v_max = max(abs(vx.min()), abs(vx.max())) * 1.1
        v_edges = np.linspace(-v_max, v_max, num_bins_v + 1)

        hist, _, _ = np.histogram2d(x, vx, bins=[x_edges, v_edges])
        return hist, x_edges, v_edges

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_points(
                points=self.positions,
                radius=0.02,
                name="electrons",
                colors=(0.3, 0.6, 1.0),
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
    parser.add_argument("--num-frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument("--num-particles", type=int, default=200000, help="Number of particles.")
    parser.add_argument("--grid-size", type=int, default=32, help="Grid resolution per axis.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            stage_path=args.stage_path,
            num_particles=args.num_particles,
            grid_size=args.grid_size,
        )

        for i in range(args.num_frames):
            example.step()
            example.render()

            if i % 50 == 0:
                v = example.velocities.numpy()
                ke = 0.5 * np.sum(v**2)
                print(f"Frame {i}: KE={ke:.1f}")

        if example.renderer:
            if hasattr(example.renderer, 'save'):
                example.renderer.save()
            if hasattr(example.renderer, 'save_image'):
                example.renderer.save_image("example_pic.png")
                print("Saved example_pic.png")
