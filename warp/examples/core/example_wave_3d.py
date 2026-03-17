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
# Example Wave 3D
#
# Extends the 2D wave equation to a 3D domain, simulating pressure
# waves propagating through a cubic volume. A pulsing source at the
# center emits spherical wavefronts that reflect off boundaries.
# The wavefield is visualized by extracting an isosurface with
# marching cubes.
#
# Demonstrates:
#   - 3D finite-difference wave equation
#   - 7-point Laplacian stencil on a 3D grid
#   - Time-stepping with Verlet integration (current + previous)
#   - Marching cubes isosurface of a time-varying field
#
###########################################################################

import math

import numpy as np

import warp as wp
import warp.render


@wp.kernel
def wave_step(
    current: wp.array3d[wp.float32],
    previous: wp.array3d[wp.float32],
    output: wp.array3d[wp.float32],
    c2_dt2: float,
    damping: float,
):
    i, j, k = wp.tid()

    nx = current.shape[0]
    ny = current.shape[1]
    nz = current.shape[2]

    # Skip boundaries (reflective)
    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
        output[i, j, k] = 0.0
        return

    u = current[i, j, k]

    # 7-point Laplacian
    lap = (
        current[i + 1, j, k]
        + current[i - 1, j, k]
        + current[i, j + 1, k]
        + current[i, j - 1, k]
        + current[i, j, k + 1]
        + current[i, j, k - 1]
        - 6.0 * u
    )

    # Verlet: u_new = 2*u - u_prev + c^2 * dt^2 * laplacian
    output[i, j, k] = (2.0 * u - previous[i, j, k] + c2_dt2 * lap) * damping


@wp.kernel
def add_source(
    field: wp.array3d[wp.float32],
    cx: int,
    cy: int,
    cz: int,
    radius: int,
    amplitude: float,
):
    i, j, k = wp.tid()

    nx = field.shape[0]
    ny = field.shape[1]
    nz = field.shape[2]

    dx = i - cx
    dy = j - cy
    dz = k - cz
    dist_sq = dx * dx + dy * dy + dz * dz

    if dist_sq < radius * radius:
        # Smooth falloff
        r = wp.sqrt(float(dist_sq)) / float(radius)
        field[i, j, k] = field[i, j, k] + amplitude * (1.0 - r)


class Example:
    def __init__(self, stage_path="example_wave_3d.usd", grid_size=128):
        self.grid_size = grid_size
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 30.0

        n = grid_size

        # Wave parameters
        self.wave_speed = 15.0
        self.dx = 1.0
        self.sim_dt = 0.03
        self.substeps = 5
        self.damping = 0.998
        self.source_interval = 0.5  # Seconds between pulses

        c2_dt2 = (self.wave_speed * self.sim_dt / self.dx) ** 2
        self.c2_dt2 = min(c2_dt2, 0.3)  # CFL limit

        # Fields
        self.current = wp.zeros((n, n, n), dtype=wp.float32)
        self.previous = wp.zeros((n, n, n), dtype=wp.float32)
        self.output = wp.zeros((n, n, n), dtype=wp.float32)

        # Source position (center)
        self.source_x = n // 2
        self.source_y = n // 2
        self.source_z = n // 2

        # Marching cubes
        self.mc = wp.MarchingCubes(nx=n, ny=n, nz=n)

        if stage_path and stage_path.endswith((".usd", ".usda", ".usdc")):
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = wp.render.NativeRenderer(512, 512)
            self.renderer.setup_camera(pos=(100, 60, 100), target=(40, 40, 40), fov=50)

    def step(self):
        with wp.ScopedTimer("step", active=False):
            n = self.grid_size

            for _ in range(self.substeps):
                # Add pulsing source
                pulse = math.sin(self.sim_time * 2.0 * math.pi / self.source_interval)
                if abs(pulse) > 0.8:
                    amplitude = 0.3 * pulse
                    wp.launch(
                        kernel=add_source,
                        dim=(n, n, n),
                        inputs=[self.current, self.source_x, self.source_y, self.source_z, 3, amplitude],
                    )

                # Wave equation step
                wp.launch(
                    kernel=wave_step,
                    dim=(n, n, n),
                    inputs=[self.current, self.previous, self.output, self.c2_dt2, self.damping],
                )

                # Rotate buffers
                self.previous, self.current, self.output = self.current, self.output, self.previous

                self.sim_time += self.sim_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            # Extract positive and negative wavefronts
            self.mc.surface(self.current, threshold=0.05)

            self.renderer.begin_frame(self.sim_time)
            if self.mc.verts is not None and len(self.mc.verts) > 0:
                self.renderer.render_mesh(
                    points=self.mc.verts,
                    indices=self.mc.indices,
                    name="wavefront",
                    colors=(0.2, 0.5, 0.9),
                )
            self.renderer.end_frame()

    def compute_wave_energy(self):
        """Compute total wave energy: E = 0.5 * ∫ [(∂u/∂t)² + c²|∇u|²] dV.

        Without damping or sources, total energy should be exactly conserved
        by the Verlet integrator. With damping, energy should decay
        exponentially. This is the primary validation metric.
        """
        u = self.current.numpy()
        u_prev = self.previous.numpy()

        # Kinetic energy: (∂u/∂t)² ≈ ((u - u_prev) / dt)²
        dudt = (u - u_prev) / self.sim_dt
        ke = 0.5 * np.sum(dudt**2)

        # Potential energy: c²|∇u|² via finite differences
        dudx = np.diff(u, axis=0, append=u[:1, :, :])
        dudy = np.diff(u, axis=1, append=u[:, :1, :])
        dudz = np.diff(u, axis=2, append=u[:, :, :1])
        pe = 0.5 * self.wave_speed**2 * np.sum(dudx**2 + dudy**2 + dudz**2)

        return ke, pe, ke + pe

    def measure_wavefront_speed(self):
        """Measure wavefront propagation speed and compare to theoretical c.

        Finds the maximum radius at which |u| > threshold, which should
        advance at speed c = wave_speed per unit time. Comparing measured
        vs theoretical validates the discretization.
        """
        u = self.current.numpy()
        n = self.grid_size
        cx, cy, cz = n // 2, n // 2, n // 2

        # Find maximum radius where |u| > threshold
        threshold = 0.01
        max_r = 0.0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if abs(u[i, j, k]) > threshold:
                        r = math.sqrt((i - cx)**2 + (j - cy)**2 + (k - cz)**2)
                        max_r = max(max_r, r)

        # Theoretical: r = c * t (in grid units, c_grid = wave_speed * dt / dx)
        theoretical_r = self.wave_speed * self.sim_time / self.dx
        return max_r, theoretical_r


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
    parser.add_argument("--grid-size", type=int, default=128, help="Grid resolution per axis.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, grid_size=args.grid_size)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            if hasattr(example.renderer, 'save'):
                example.renderer.save()
            if hasattr(example.renderer, 'save_image'):
                example.renderer.save_image("example_wave_3d.png")
                print("Saved example_wave_3d.png")
