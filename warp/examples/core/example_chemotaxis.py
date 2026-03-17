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
# Example Chemotaxis
#
# Simulates bacterial chemotaxis: thousands of agents swim through a
# 3D chemical concentration field, biasing their movement toward
# higher nutrient concentrations. Bacteria secrete a chemoattractant
# that recruits more cells, producing emergent clustering, streaming,
# and colony-like aggregation patterns.
#
# The model couples:
#   - Agent-based motility (run-and-tumble with gradient bias)
#   - Diffusing chemical field (nutrient + secreted attractant)
#   - Consumption/secretion at agent positions
#
# Demonstrates:
#   - Hybrid particle-grid simulation
#   - Gradient-biased random walks
#   - Chemical field diffusion on a 3D grid
#   - Scatter/gather between particles and grid
#
###########################################################################

import numpy as np

import warp as wp
import warp.render


@wp.kernel
def diffuse_field(
    current: wp.array3d[wp.float32],
    output: wp.array3d[wp.float32],
    diff_rate: float,
    decay: float,
    dt: float,
):
    """Diffuse and decay a 3D chemical field."""
    i, j, k = wp.tid()

    nx = current.shape[0]
    ny = current.shape[1]
    nz = current.shape[2]

    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
        output[i, j, k] = 0.0
        return

    u = current[i, j, k]
    lap = (
        current[i + 1, j, k] + current[i - 1, j, k]
        + current[i, j + 1, k] + current[i, j - 1, k]
        + current[i, j, k + 1] + current[i, j, k - 1]
        - 6.0 * u
    )

    output[i, j, k] = u + (diff_rate * lap - decay * u) * dt


@wp.kernel
def move_bacteria(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    chem_field: wp.array3d[wp.float32],
    speed: float,
    tumble_rate: float,
    gradient_bias: float,
    domain_size: float,
    grid_size: int,
    seed: int,
    dt: float,
):
    """Run-and-tumble motility with chemotactic bias."""
    tid = wp.tid()

    p = positions[tid]
    v = velocities[tid]
    state = wp.rand_init(seed, tid)

    # Sample chemical gradient at current position
    scale = float(grid_size) / domain_size
    gx = p[0] * scale
    gy = p[1] * scale
    gz = p[2] * scale

    ix = wp.clamp(int(gx), 1, grid_size - 2)
    iy = wp.clamp(int(gy), 1, grid_size - 2)
    iz = wp.clamp(int(gz), 1, grid_size - 2)

    # Central-difference gradient
    grad_x = (chem_field[ix + 1, iy, iz] - chem_field[ix - 1, iy, iz]) * 0.5
    grad_y = (chem_field[ix, iy + 1, iz] - chem_field[ix, iy - 1, iz]) * 0.5
    grad_z = (chem_field[ix, iy, iz + 1] - chem_field[ix, iy, iz - 1]) * 0.5

    grad = wp.vec3(grad_x, grad_y, grad_z)
    grad_mag = wp.length(grad)

    # Tumble decision: less likely to tumble when moving up-gradient
    alignment = float(0.0)
    if grad_mag > 1.0e-6:
        alignment = wp.dot(v, grad) / (wp.length(v) * grad_mag + 1.0e-8)

    # Reduce tumble rate when aligned with gradient
    effective_tumble = tumble_rate * (1.0 - gradient_bias * wp.clamp(alignment, 0.0, 1.0))

    if wp.randf(state) < effective_tumble * dt:
        # Tumble: pick new random direction, biased toward gradient
        rx = wp.randf(state) * 2.0 - 1.0
        ry = wp.randf(state) * 2.0 - 1.0
        rz = wp.randf(state) * 2.0 - 1.0
        rand_dir = wp.normalize(wp.vec3(rx, ry, rz))

        # Bias toward gradient
        if grad_mag > 1.0e-6:
            grad_dir = grad / grad_mag
            new_dir = wp.normalize(rand_dir + grad_dir * gradient_bias * 2.0)
        else:
            new_dir = rand_dir

        v = new_dir * speed
    
    # Move
    p = p + v * dt

    # Reflective boundaries
    for dim in range(3):
        pd = p[dim]
        vd = v[dim]
        if pd < 0.0:
            pd = -pd
            vd = -vd
        elif pd >= domain_size:
            pd = 2.0 * domain_size - pd
            vd = -vd
        if dim == 0:
            p = wp.vec3(pd, p[1], p[2])
            v = wp.vec3(vd, v[1], v[2])
        elif dim == 1:
            p = wp.vec3(p[0], pd, p[2])
            v = wp.vec3(v[0], vd, v[2])
        else:
            p = wp.vec3(p[0], p[1], pd)
            v = wp.vec3(v[0], v[1], vd)

    positions[tid] = p
    velocities[tid] = v


@wp.kernel
def secrete_chemical(
    positions: wp.array[wp.vec3],
    chem_field: wp.array3d[wp.float32],
    domain_size: float,
    grid_size: int,
    amount: float,
):
    """Bacteria deposit chemoattractant at their grid position."""
    tid = wp.tid()

    p = positions[tid]
    scale = float(grid_size) / domain_size

    ix = wp.clamp(int(p[0] * scale), 0, grid_size - 1)
    iy = wp.clamp(int(p[1] * scale), 0, grid_size - 1)
    iz = wp.clamp(int(p[2] * scale), 0, grid_size - 1)

    wp.atomic_add(chem_field, ix, iy, iz, amount)


class Example:
    def __init__(self, stage_path="example_chemotaxis.usd", num_bacteria=50000, grid_size=64):
        self.num_bacteria = num_bacteria
        self.grid_size = grid_size
        self.domain_size = 20.0
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 30.0
        self.substeps = 5
        self.seed = 0

        # Bacteria: start uniformly distributed
        rng = np.random.default_rng(42)
        positions = rng.uniform(2.0, self.domain_size - 2.0, (num_bacteria, 3)).astype(np.float32)

        # Random initial velocities
        speed = 2.0
        dirs = rng.normal(0, 1, (num_bacteria, 3)).astype(np.float32)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
        velocities = dirs * speed

        self.positions = wp.array(positions, dtype=wp.vec3)
        self.velocities = wp.array(velocities, dtype=wp.vec3)

        # Chemical field (nutrient)
        n = grid_size
        chem = np.zeros((n, n, n), dtype=np.float32)
        # Initial nutrient hotspot at center
        cx, cy, cz = n // 2, n // 2, n // 2
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    d = ((i - cx)**2 + (j - cy)**2 + (k - cz)**2) ** 0.5
                    chem[i, j, k] = max(0, 1.0 - d / (n * 0.3))

        self.chem_field = wp.array(chem, dtype=wp.float32)
        self.chem_tmp = wp.zeros((n, n, n), dtype=wp.float32)

        # Parameters
        self.speed = speed
        self.tumble_rate = 3.0
        self.gradient_bias = 0.8
        self.diff_rate = 0.5
        self.decay = 0.02
        self.secrete_amount = 0.05

        if stage_path and stage_path.endswith((".usd", ".usda", ".usdc")):
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = wp.render.NativeRenderer(512, 512)
            self.renderer.setup_camera(pos=(25, 15, 25), target=(10, 10, 10), fov=50)
            self.renderer.bg_top = wp.vec3(0.05, 0.08, 0.05)
            self.renderer.bg_bottom = wp.vec3(0.02, 0.03, 0.02)
            self.renderer.shadows = False

    def step(self):
        with wp.ScopedTimer("step", active=False):
            n = self.grid_size
            dt = self.frame_dt / self.substeps

            for _ in range(self.substeps):
                self.seed += 1

                # Move bacteria
                wp.launch(
                    kernel=move_bacteria,
                    dim=self.num_bacteria,
                    inputs=[
                        self.positions, self.velocities, self.chem_field,
                        self.speed, self.tumble_rate, self.gradient_bias,
                        self.domain_size, self.grid_size, self.seed, dt,
                    ],
                )

                # Secrete attractant
                wp.launch(
                    kernel=secrete_chemical,
                    dim=self.num_bacteria,
                    inputs=[self.positions, self.chem_field, self.domain_size, self.grid_size, self.secrete_amount * dt],
                )

                # Diffuse chemical field
                wp.launch(
                    kernel=diffuse_field,
                    dim=(n, n, n),
                    inputs=[self.chem_field, self.chem_tmp, self.diff_rate, self.decay, dt],
                )
                self.chem_field, self.chem_tmp = self.chem_tmp, self.chem_field

            self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_points(
                points=self.positions,
                radius=0.08,
                name="bacteria",
                colors=(0.2, 0.85, 0.3),
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
    parser.add_argument("--num-bacteria", type=int, default=50000, help="Number of bacteria.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_bacteria=args.num_bacteria)

        for i in range(args.num_frames):
            example.step()
            example.render()

            if i % 50 == 0:
                c = example.chem_field.numpy()
                print(f"Frame {i}: chem max={c.max():.3f}")

        if example.renderer:
            if hasattr(example.renderer, 'save'):
                example.renderer.save()
            if hasattr(example.renderer, 'save_image'):
                example.renderer.save_image("example_chemotaxis.png")
                print("Saved example_chemotaxis.png")
