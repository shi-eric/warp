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
# Example Crystal Growth — Dendritic Solidification (Phase-Field Model)
#
# Simulates crystal growth from a supercooled melt using a phase-field
# model.  A scalar order parameter φ (0 = liquid, 1 = solid) evolves
# via the Allen-Cahn equation coupled to a temperature field T:
#
#     τ ∂φ/∂t = ε²∇²φ + φ(1-φ)(φ - 0.5 + m)
#     ∂T/∂t   = D∇²T + L ∂φ/∂t
#
# Anisotropic surface energy (4-fold symmetry) produces beautiful
# dendritic branching patterns reminiscent of snowflakes.
#
# Demonstrates:
#   - Phase-field solidification on GPU
#   - Anisotropic interface energy with 4-fold crystal symmetry
#   - Coupled reaction-diffusion (order parameter + temperature)
#   - Marching cubes isosurface extraction of the solidification front
#
# Validation:
#   - Dendrite tip velocity (should reach steady state)
#   - Total energy conservation (latent heat balance)
#   - Solid fraction growth rate
#
###########################################################################

import math

import numpy as np

import warp as wp
import warp.render


@wp.kernel
def phase_field_step(
    phi: wp.array3d[wp.float32],
    temp: wp.array3d[wp.float32],
    phi_new: wp.array3d[wp.float32],
    temp_new: wp.array3d[wp.float32],
    tau: float,
    eps0: float,
    delta_aniso: float,
    D_thermal: float,
    latent: float,
    coupling: float,
    dt: float,
):
    """Coupled phase-field + temperature step with anisotropic surface energy."""
    i, j, k = wp.tid()

    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]

    # Boundary: no-flux (Neumann)
    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
        phi_new[i, j, k] = phi[i, j, k]
        temp_new[i, j, k] = temp[i, j, k]
        return

    p = phi[i, j, k]
    t = temp[i, j, k]

    # Laplacian of phi (7-point 3D stencil)
    lap_phi = (
        phi[i + 1, j, k] + phi[i - 1, j, k]
        + phi[i, j + 1, k] + phi[i, j - 1, k]
        + phi[i, j, k + 1] + phi[i, j, k - 1]
        - 6.0 * p
    )

    # Laplacian of temperature
    lap_t = (
        temp[i + 1, j, k] + temp[i - 1, j, k]
        + temp[i, j + 1, k] + temp[i, j - 1, k]
        + temp[i, j, k + 1] + temp[i, j, k - 1]
        - 6.0 * t
    )

    # Interface normal (gradient of phi) for anisotropy
    dpx = (phi[i + 1, j, k] - phi[i - 1, j, k]) * 0.5
    dpy = (phi[i, j + 1, k] - phi[i, j - 1, k]) * 0.5

    grad_mag = wp.sqrt(dpx * dpx + dpy * dpy + 1.0e-12)
    nx_dir = dpx / grad_mag
    ny_dir = dpy / grad_mag

    # 4-fold anisotropy: ε(θ) = ε0 * (1 + δ cos(4θ))
    # cos(4θ) = 8cos⁴θ - 8cos²θ + 1  for nx_dir = cos(θ)
    cos2 = nx_dir * nx_dir
    cos4_theta = 8.0 * cos2 * cos2 - 8.0 * cos2 + 1.0
    eps_val = eps0 * (1.0 + delta_aniso * cos4_theta)
    eps2 = eps_val * eps_val

    # Driving force: m = -coupling * T (negative T = undercooling favors solidification)
    m = -coupling * t

    # Double-well potential + driving
    # f'(φ) = φ(1-φ)(φ - 0.5 + m)
    reaction = p * (1.0 - p) * (p - 0.5 + m)

    # Phase-field update: τ dφ/dt = ε²∇²φ + f'(φ)
    dphi_dt = (eps2 * lap_phi + reaction) / tau
    p_new = p + dt * dphi_dt

    # Clamp to [0, 1]
    p_new = wp.clamp(p_new, 0.0, 1.0)

    # Temperature update: dT/dt = D∇²T + L dφ/dt
    dt_dt = D_thermal * lap_t + latent * dphi_dt
    t_new = t + dt * dt_dt

    phi_new[i, j, k] = p_new
    temp_new[i, j, k] = t_new


@wp.kernel
def compute_solid_fraction_kernel(
    phi: wp.array3d[wp.float32],
    result: wp.array[wp.float32],
):
    """Sum phi values (solid fraction ∝ Σφ / N)."""
    i, j, k = wp.tid()
    wp.atomic_add(result, 0, phi[i, j, k])


@wp.kernel
def compute_thermal_energy_kernel(
    temp: wp.array3d[wp.float32],
    phi: wp.array3d[wp.float32],
    result: wp.array[wp.float32],
    latent: float,
):
    """Sum total thermal energy: T + L*φ should be approximately conserved."""
    i, j, k = wp.tid()
    wp.atomic_add(result, 0, temp[i, j, k] + latent * phi[i, j, k])


class Example:
    def __init__(self, stage_path="example_crystal_growth.usd", grid_x=256, grid_y=256, grid_z=32):
        self.nx = grid_x
        self.ny = grid_y
        self.nz = grid_z
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 30.0
        self.substeps = 50

        # Phase-field parameters (tuned for visible dendritic growth)
        self.tau = 0.3          # Relaxation time
        self.eps0 = 0.5         # Interface width parameter
        self.delta_aniso = 0.05 # Anisotropy strength (4-fold)
        self.D_thermal = 1.0    # Thermal diffusivity
        self.latent = 0.5       # Latent heat coefficient
        self.coupling = 0.8     # Coupling: m = coupling * T
        self.dt = 0.005         # Substep dt

        self.undercooling = -0.65  # Initial supercooling (stronger)

        # Fields
        self.phi = wp.zeros((self.nx, self.ny, self.nz), dtype=wp.float32)
        self.temp = wp.full((self.nx, self.ny, self.nz), self.undercooling, dtype=wp.float32)
        self.phi_new = wp.zeros((self.nx, self.ny, self.nz), dtype=wp.float32)
        self.temp_new = wp.zeros((self.nx, self.ny, self.nz), dtype=wp.float32)

        # Seed: small solid nucleus at center
        self._seed_nucleus()

        # Accumulators
        self.sum_buf = wp.zeros(1, dtype=wp.float32)

        # Marching cubes for isosurface
        self.mc = wp.MarchingCubes(nx=self.nx, ny=self.ny, nz=self.nz)

        # Tip tracking
        self.tip_positions = []  # (time, max_x_of_solid)

        if stage_path and stage_path.endswith((".usd", ".usda", ".usdc")):
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = wp.render.NativeRenderer(512, 512)
            cx = float(self.nx) / 2.0
            cy = float(self.ny) / 2.0
            cz = float(self.nz) / 2.0
            dist = float(max(self.nx, self.ny)) * 0.8
            self.renderer.setup_camera(
                pos=(cx + dist * 0.3, cy + dist * 0.3, cz + dist * 1.2),
                target=(cx, cy, cz),
                fov=50,
            )
            self.renderer.set_environment("dark")

    def _seed_nucleus(self):
        """Place a small solid sphere at the center."""
        phi_np = self.phi.numpy()
        cx, cy, cz = self.nx // 2, self.ny // 2, self.nz // 2
        radius = 15

        for i in range(max(0, cx - radius), min(self.nx, cx + radius + 1)):
            for j in range(max(0, cy - radius), min(self.ny, cy + radius + 1)):
                for k in range(max(0, cz - radius), min(self.nz, cz + radius + 1)):
                    dx = i - cx
                    dy = j - cy
                    dz = k - cz
                    r = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if r < radius:
                        phi_np[i, j, k] = 1.0

        self.phi = wp.array(phi_np, dtype=wp.float32)

    def step(self):
        with wp.ScopedTimer("step", active=False):
            for _ in range(self.substeps):
                wp.launch(
                    kernel=phase_field_step,
                    dim=(self.nx, self.ny, self.nz),
                    inputs=[
                        self.phi, self.temp,
                        self.phi_new, self.temp_new,
                        self.tau, self.eps0, self.delta_aniso,
                        self.D_thermal, self.latent, self.coupling,
                        self.dt,
                    ],
                )

                # Swap buffers
                self.phi, self.phi_new = self.phi_new, self.phi
                self.temp, self.temp_new = self.temp_new, self.temp

            self.sim_time += self.frame_dt

            # Track dendrite tip
            self._track_tip()

    def _track_tip(self):
        """Record the furthest extent of solid along +x axis from center."""
        phi_np = self.phi.numpy()
        cx = self.nx // 2
        cy = self.ny // 2
        kz = self.nz // 2

        max_r = 0.0
        for i in range(self.nx):
            if phi_np[i, cy, kz] > 0.5:
                r = abs(i - cx)
                max_r = max(max_r, float(r))

        self.tip_positions.append((self.sim_time, max_r))

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            # Extract solidification front isosurface at φ = 0.5
            self.mc.surface(self.phi, threshold=0.5)

            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_ground(y=0.0)

            if self.mc.verts is not None and len(self.mc.verts) > 0:
                self.renderer.render_mesh(
                    points=self.mc.verts,
                    indices=self.mc.indices,
                    name="crystal",
                    colors=(0.7, 0.85, 1.0),
                )
            self.renderer.end_frame()

    # ---- Validation methods ----

    def compute_solid_fraction(self):
        """Compute fraction of domain that is solid (φ > 0.5 equivalent: mean(φ))."""
        self.sum_buf.zero_()
        wp.launch(
            kernel=compute_solid_fraction_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[self.phi, self.sum_buf],
        )
        total = float(self.sum_buf.numpy()[0])
        return total / float(self.nx * self.ny * self.nz)

    def compute_thermal_energy(self):
        """Compute total thermal energy: Σ(T + L*φ).

        In a closed system without boundary flux, this should be conserved
        (latent heat released by solidification heats the liquid).
        """
        self.sum_buf.zero_()
        wp.launch(
            kernel=compute_thermal_energy_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[self.temp, self.phi, self.sum_buf, self.latent],
        )
        return float(self.sum_buf.numpy()[0])

    def compute_tip_velocity(self):
        """Estimate dendrite tip velocity from tracked tip positions.

        Returns velocity in grid cells per unit time (averaged over
        recent measurements). Steady-state tip velocity is a key
        validation metric for phase-field models.
        """
        if len(self.tip_positions) < 5:
            return 0.0

        # Use last 5 measurements for slope
        recent = self.tip_positions[-5:]
        t0, r0 = recent[0]
        t1, r1 = recent[-1]
        dt = t1 - t0
        if dt < 1.0e-8:
            return 0.0
        return (r1 - r0) / dt

    def validate_energy_conservation(self):
        """Check that total thermal energy (T + L*φ) is conserved.

        Returns (energy, initial_undercooling_energy) for comparison.
        Initial: N * (T0 + L * 0) for liquid + N_seed * L for seed.
        """
        energy = self.compute_thermal_energy()
        n_total = self.nx * self.ny * self.nz
        # Approximate initial: most cells are liquid at T=undercooling, small seed at φ=1
        initial_approx = n_total * self.undercooling  # plus small seed correction
        return energy, initial_approx


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
    parser.add_argument("--num-frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument("--grid-x", type=int, default=256, help="Grid x resolution.")
    parser.add_argument("--grid-y", type=int, default=256, help="Grid y resolution.")
    parser.add_argument("--grid-z", type=int, default=32, help="Grid z resolution.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            stage_path=args.stage_path,
            grid_x=args.grid_x,
            grid_y=args.grid_y,
            grid_z=args.grid_z,
        )

        for frame in range(args.num_frames):
            example.step()
            example.render()

            if frame % 50 == 0:
                sf = example.compute_solid_fraction()
                tv = example.compute_tip_velocity()
                te = example.compute_thermal_energy()
                print(f"Frame {frame}: solid_frac={sf:.4f}, tip_vel={tv:.2f}, thermal_energy={te:.2f}")

        # Final validation
        sf = example.compute_solid_fraction()
        tv = example.compute_tip_velocity()
        te, te_init = example.validate_energy_conservation()
        print(f"\nFinal validation:")
        print(f"  Solid fraction:  {sf:.4f}")
        print(f"  Tip velocity:    {tv:.3f} cells/time")
        print(f"  Thermal energy:  {te:.2f} (initial ≈ {te_init:.2f})")

        if example.renderer:
            if hasattr(example.renderer, "save"):
                example.renderer.save()
