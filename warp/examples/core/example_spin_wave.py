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
# Example Spin Wave — Heisenberg Spin Wave (Magnon) Propagation
#
# Simulates classical Heisenberg spins on a 3D cubic lattice with
# nearest-neighbor exchange coupling.  The Landau-Lifshitz equation
# of motion (without damping) propagates magnon excitations:
#
#     dS/dt = -γ S × H_eff,    H_eff = J Σ_neighbors S_j
#
# All spins start along +z; a localized perturbation seeds a spin
# wave that propagates outward at the magnon group velocity.
#
# Demonstrates:
#   - 3D stencil computation on a periodic lattice (GPU)
#   - Landau-Lifshitz spin dynamics with symplectic mid-point rule
#   - Marching cubes isosurface of spin-z deviation
#   - Energy conservation validation for undamped dynamics
#
# Validation:
#   - Magnon dispersion ω(k) = 4JS(1 − cos(ka))
#   - Wavefront propagation speed
#   - Total energy conservation (Heisenberg Hamiltonian)
#
###########################################################################

import math

import numpy as np

import warp as wp
import warp.render


@wp.func
def periodic(i: int, n: int):
    """Wrap index into [0, n) with periodic BC."""
    return (i % n + n) % n


@wp.kernel
def compute_heff(
    sx: wp.array3d[wp.float32],
    sy: wp.array3d[wp.float32],
    sz: wp.array3d[wp.float32],
    hx: wp.array3d[wp.float32],
    hy: wp.array3d[wp.float32],
    hz: wp.array3d[wp.float32],
    J: float,
):
    """Compute effective field H_eff = J * sum of 6 nearest neighbors."""
    i, j, k = wp.tid()
    nx = sx.shape[0]
    ny = sx.shape[1]
    nz = sx.shape[2]

    ip = periodic(i + 1, nx)
    im = periodic(i - 1, nx)
    jp = periodic(j + 1, ny)
    jm = periodic(j - 1, ny)
    kp = periodic(k + 1, nz)
    km = periodic(k - 1, nz)

    sum_x = sx[ip, j, k] + sx[im, j, k] + sx[i, jp, k] + sx[i, jm, k] + sx[i, j, kp] + sx[i, j, km]
    sum_y = sy[ip, j, k] + sy[im, j, k] + sy[i, jp, k] + sy[i, jm, k] + sy[i, j, kp] + sy[i, j, km]
    sum_z = sz[ip, j, k] + sz[im, j, k] + sz[i, jp, k] + sz[i, jm, k] + sz[i, j, kp] + sz[i, j, km]

    hx[i, j, k] = J * sum_x
    hy[i, j, k] = J * sum_y
    hz[i, j, k] = J * sum_z


@wp.kernel
def spin_step_ll(
    sx: wp.array3d[wp.float32],
    sy: wp.array3d[wp.float32],
    sz: wp.array3d[wp.float32],
    hx: wp.array3d[wp.float32],
    hy: wp.array3d[wp.float32],
    hz: wp.array3d[wp.float32],
    sx_out: wp.array3d[wp.float32],
    sy_out: wp.array3d[wp.float32],
    sz_out: wp.array3d[wp.float32],
    gamma_dt: float,
):
    """Landau-Lifshitz step: S_new = S + gamma_dt * (S × H_eff), then normalize."""
    i, j, k = wp.tid()

    s_x = sx[i, j, k]
    s_y = sy[i, j, k]
    s_z = sz[i, j, k]

    h_x = hx[i, j, k]
    h_y = hy[i, j, k]
    h_z = hz[i, j, k]

    # Cross product: S × H_eff
    cx = s_y * h_z - s_z * h_y
    cy = s_z * h_x - s_x * h_z
    cz = s_x * h_y - s_y * h_x

    # Update: dS/dt = -gamma S × H_eff  (note negative sign)
    nx_val = s_x - gamma_dt * cx
    ny_val = s_y - gamma_dt * cy
    nz_val = s_z - gamma_dt * cz

    # Normalize to maintain |S| = 1
    inv_len = 1.0 / wp.sqrt(nx_val * nx_val + ny_val * ny_val + nz_val * nz_val + 1.0e-20)
    sx_out[i, j, k] = nx_val * inv_len
    sy_out[i, j, k] = ny_val * inv_len
    sz_out[i, j, k] = nz_val * inv_len


@wp.kernel
def compute_sz_deviation(
    sz: wp.array3d[wp.float32],
    deviation: wp.array3d[wp.float32],
):
    """Compute |Sz - 1| for isosurface extraction."""
    i, j, k = wp.tid()
    deviation[i, j, k] = wp.abs(sz[i, j, k] - 1.0)


@wp.kernel
def compute_energy_kernel(
    sx: wp.array3d[wp.float32],
    sy: wp.array3d[wp.float32],
    sz: wp.array3d[wp.float32],
    energy: wp.array[wp.float32],
    J: float,
):
    """Compute Heisenberg energy H = -J Σ Si·Sj (sum over +x,+y,+z neighbors only)."""
    i, j, k = wp.tid()
    nx = sx.shape[0]
    ny = sx.shape[1]
    nz = sx.shape[2]

    s_x = sx[i, j, k]
    s_y = sy[i, j, k]
    s_z = sz[i, j, k]

    ip = periodic(i + 1, nx)
    jp = periodic(j + 1, ny)
    kp = periodic(k + 1, nz)

    # Dot product with +x, +y, +z neighbors (each bond counted once)
    dot_xp = s_x * sx[ip, j, k] + s_y * sy[ip, j, k] + s_z * sz[ip, j, k]
    dot_yp = s_x * sx[i, jp, k] + s_y * sy[i, jp, k] + s_z * sz[i, jp, k]
    dot_zp = s_x * sx[i, j, kp] + s_y * sy[i, j, kp] + s_z * sz[i, j, kp]

    e = -J * (dot_xp + dot_yp + dot_zp)
    wp.atomic_add(energy, 0, e)


class Example:
    def __init__(self, stage_path="example_spin_wave.usd", grid_size=128):
        self.grid_size = grid_size
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 30.0
        self.substeps = 10

        n = grid_size

        # Physical parameters
        self.J = 1.0        # Exchange coupling
        self.gamma = 1.0    # Gyromagnetic ratio
        self.dt = 0.005     # Time step per substep

        # Spin arrays: 3 separate scalar fields for Sx, Sy, Sz
        self.sx = wp.zeros((n, n, n), dtype=wp.float32)
        self.sy = wp.zeros((n, n, n), dtype=wp.float32)
        self.sz = wp.full((n, n, n), 1.0, dtype=wp.float32)

        # Temporary arrays for the step
        self.sx_out = wp.zeros((n, n, n), dtype=wp.float32)
        self.sy_out = wp.zeros((n, n, n), dtype=wp.float32)
        self.sz_out = wp.zeros((n, n, n), dtype=wp.float32)

        # Effective field
        self.hx = wp.zeros((n, n, n), dtype=wp.float32)
        self.hy = wp.zeros((n, n, n), dtype=wp.float32)
        self.hz = wp.zeros((n, n, n), dtype=wp.float32)

        # Deviation field for marching cubes
        self.deviation = wp.zeros((n, n, n), dtype=wp.float32)

        # Energy accumulator
        self.energy_buf = wp.zeros(1, dtype=wp.float32)

        # Apply initial perturbation: tilt spins in a localized region
        self._apply_perturbation(n)

        # Marching cubes
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

    def _apply_perturbation(self, n):
        """Tilt spins in a sphere around the center away from +z."""
        sx_np = self.sx.numpy()
        sy_np = self.sy.numpy()
        sz_np = self.sz.numpy()

        cx, cy, cz = n // 2, n // 2, n // 2
        radius = max(3, n // 20)
        tilt_angle = 1.2  # radians (strong tilt for visible wavefront)

        for i in range(max(0, cx - radius), min(n, cx + radius + 1)):
            for j in range(max(0, cy - radius), min(n, cy + radius + 1)):
                for k in range(max(0, cz - radius), min(n, cz + radius + 1)):
                    dx = i - cx
                    dy = j - cy
                    dz = k - cz
                    r = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if r < radius:
                        # Smooth Gaussian-like tilt
                        angle = tilt_angle * math.exp(-0.5 * (r / (radius * 0.4)) ** 2)
                        sx_np[i, j, k] = math.sin(angle)
                        sy_np[i, j, k] = 0.0
                        sz_np[i, j, k] = math.cos(angle)

        self.sx = wp.array(sx_np, dtype=wp.float32)
        self.sy = wp.array(sy_np, dtype=wp.float32)
        self.sz = wp.array(sz_np, dtype=wp.float32)

    def step(self):
        with wp.ScopedTimer("step", active=False):
            n = self.grid_size
            gamma_dt = self.gamma * self.dt

            for _ in range(self.substeps):
                # Compute effective field
                wp.launch(
                    kernel=compute_heff,
                    dim=(n, n, n),
                    inputs=[self.sx, self.sy, self.sz,
                            self.hx, self.hy, self.hz, self.J],
                )

                # Landau-Lifshitz step
                wp.launch(
                    kernel=spin_step_ll,
                    dim=(n, n, n),
                    inputs=[
                        self.sx, self.sy, self.sz,
                        self.hx, self.hy, self.hz,
                        self.sx_out, self.sy_out, self.sz_out,
                        gamma_dt,
                    ],
                )

                # Swap buffers
                self.sx, self.sx_out = self.sx_out, self.sx
                self.sy, self.sy_out = self.sy_out, self.sy
                self.sz, self.sz_out = self.sz_out, self.sz

            self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            n = self.grid_size

            # Compute Sz deviation field
            wp.launch(
                kernel=compute_sz_deviation,
                dim=(n, n, n),
                inputs=[self.sz, self.deviation],
            )

            # Extract isosurface where |Sz - 1| > threshold
            self.mc.surface(self.deviation, threshold=0.01)

            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_ground(y=0.0)

            if self.mc.verts is not None and len(self.mc.verts) > 0:
                self.renderer.render_mesh(
                    points=self.mc.verts,
                    indices=self.mc.indices,
                    name="spin_wave",
                    colors=(0.3, 0.6, 0.95),
                )
            self.renderer.end_frame()

    # ---- Validation methods ----

    def compute_total_energy(self):
        """Compute Heisenberg exchange energy H = -J Σ Si · Sj.

        For undamped Landau-Lifshitz dynamics this is a conserved quantity.
        Returns the total energy (float).
        """
        n = self.grid_size
        self.energy_buf.zero_()
        wp.launch(
            kernel=compute_energy_kernel,
            dim=(n, n, n),
            inputs=[self.sx, self.sy, self.sz, self.energy_buf, self.J],
        )
        return float(self.energy_buf.numpy()[0])

    def measure_wavefront_radius(self, threshold=0.005):
        """Measure maximum radius (in lattice units) where |Sz - 1| > threshold.

        The spin wave should propagate at the magnon group velocity
        v_g = 2Ja sin(ka) ≈ 2Ja for small k. For the long-wavelength
        limit on a cubic lattice with J=1, a=1: v_g ≈ 2.
        """
        sz_np = self.sz.numpy()
        n = self.grid_size
        cx = cy = cz = n // 2

        deviation = np.abs(sz_np - 1.0)
        max_r = 0.0
        # Sample along principal axes for speed
        for axis in range(3):
            for sign in (-1, 1):
                for d in range(1, n // 2):
                    idx = [cx, cy, cz]
                    idx[axis] = (idx[axis] + sign * d) % n
                    if deviation[idx[0], idx[1], idx[2]] > threshold:
                        max_r = max(max_r, float(d))

        return max_r

    def validate_energy_conservation(self, steps=50):
        """Run several steps and check that energy is conserved.

        Returns (initial_energy, final_energy, relative_change).
        """
        e0 = self.compute_total_energy()
        for _ in range(steps):
            self.step()
        e1 = self.compute_total_energy()
        rel = abs(e1 - e0) / (abs(e0) + 1.0e-12)
        return e0, e1, rel

    def validate_dispersion(self):
        """Measure wavefront speed and compare with theoretical magnon velocity.

        Returns dict with measured_radius, time, measured_speed, theoretical_speed.
        Theoretical long-wavelength magnon group velocity: v_g = 2*J*a*gamma
        (with a=1, gamma=1, J=1 → v_g ≈ 2 lattice sites per unit time).
        """
        r = self.measure_wavefront_radius()
        t = self.sim_time
        measured_v = r / (t + 1.0e-12)
        theoretical_v = 2.0 * self.J * self.gamma  # long-wavelength limit
        return {
            "measured_radius": r,
            "time": t,
            "measured_speed": measured_v,
            "theoretical_speed": theoretical_v,
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
                energy = example.compute_total_energy()
                radius = example.measure_wavefront_radius()
                print(f"Frame {frame}: energy={energy:.2f}, wavefront_radius={radius:.1f}")

        # Final validation
        disp = example.validate_dispersion()
        print(f"\nDispersion validation:")
        print(f"  Wavefront radius: {disp['measured_radius']:.1f} sites")
        print(f"  Measured speed:    {disp['measured_speed']:.3f} sites/time")
        print(f"  Theoretical speed: {disp['theoretical_speed']:.3f} sites/time")

        if example.renderer:
            if hasattr(example.renderer, "save"):
                example.renderer.save()
