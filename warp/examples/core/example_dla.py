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
# Example Diffusion-Limited Aggregation (DLA)
#
# Simulates 3D Diffusion-Limited Aggregation: random-walking particles
# stick on contact with existing structure, building intricate fractal
# crystal patterns starting from a single seed at the center.
#
# The classic DLA algorithm is inherently serial per particle, so this
# implementation uses batched random walks — many walkers diffuse
# simultaneously and attach when they contact the aggregate. This
# trades some physical accuracy for GPU throughput.
#
# Demonstrates:
#   - Random walks with wp.rand_init / wp.randf
#   - Atomic read/write on 3D voxel grids
#   - Iterative batched computation pattern
#   - Point cloud extraction for visualization
#
###########################################################################

import numpy as np

import warp as wp
import warp.render


@wp.kernel
def random_walk_and_attach(
    grid: wp.array3d[wp.int32],
    walker_x: wp.array[wp.int32],
    walker_y: wp.array[wp.int32],
    walker_z: wp.array[wp.int32],
    walker_active: wp.array[wp.int32],
    attached: wp.array[wp.int32],
    seed: int,
    walk_steps: int,
):
    tid = wp.tid()

    if walker_active[tid] == 0:
        return

    nx = grid.shape[0]
    ny = grid.shape[1]
    nz = grid.shape[2]

    x = walker_x[tid]
    y = walker_y[tid]
    z = walker_z[tid]

    state = wp.rand_init(seed, tid)

    for _ in range(walk_steps):
        # Random step in one of 6 directions
        r = int(wp.randf(state) * 6.0) % 6

        dx = int(0)
        dy = int(0)
        dz = int(0)
        if r == 0:
            dx = 1
        elif r == 1:
            dx = -1
        elif r == 2:
            dy = 1
        elif r == 3:
            dy = -1
        elif r == 4:
            dz = 1
        else:
            dz = -1

        nx2 = x + dx
        ny2 = y + dy
        nz2 = z + dz

        # Boundary check — kill walker if it leaves the grid
        if nx2 < 0 or nx2 >= nx or ny2 < 0 or ny2 >= ny or nz2 < 0 or nz2 >= nz:
            walker_active[tid] = 0
            return

        # Check if neighbor is occupied (part of aggregate)
        if grid[nx2, ny2, nz2] == 1:
            # Attach at current position
            grid[x, y, z] = 1
            walker_active[tid] = 0
            wp.atomic_add(attached, 0, 1)
            return

        x = nx2
        y = ny2
        z = nz2

    walker_x[tid] = x
    walker_y[tid] = y
    walker_z[tid] = z


@wp.kernel
def respawn_walkers(
    walker_x: wp.array[wp.int32],
    walker_y: wp.array[wp.int32],
    walker_z: wp.array[wp.int32],
    walker_active: wp.array[wp.int32],
    grid_size: int,
    spawn_radius: int,
    seed: int,
):
    tid = wp.tid()

    if walker_active[tid] == 1:
        return

    # Respawn on a shell around the center
    state = wp.rand_init(seed, tid)
    center = grid_size / 2

    # Random point on sphere
    theta = wp.randf(state) * 6.2832
    phi = wp.acos(wp.randf(state) * 2.0 - 1.0)
    r = float(spawn_radius)

    x = int(float(center) + r * wp.sin(phi) * wp.cos(theta))
    y = int(float(center) + r * wp.sin(phi) * wp.sin(theta))
    z = int(float(center) + r * wp.cos(phi))

    x = wp.clamp(x, 1, grid_size - 2)
    y = wp.clamp(y, 1, grid_size - 2)
    z = wp.clamp(z, 1, grid_size - 2)

    walker_x[tid] = x
    walker_y[tid] = y
    walker_z[tid] = z
    walker_active[tid] = 1


@wp.kernel
def extract_points(
    grid: wp.array3d[wp.int32],
    positions: wp.array[wp.vec3],
    count: wp.array[wp.int32],
    max_points: int,
    scale: float,
    offset: float,
):
    i, j, k = wp.tid()

    if grid[i, j, k] == 1:
        idx = wp.atomic_add(count, 0, 1)
        if idx < max_points:
            positions[idx] = wp.vec3(
                float(i) * scale - offset,
                float(j) * scale - offset,
                float(k) * scale - offset,
            )


class Example:
    def __init__(self, stage_path="example_dla.usd", grid_size=128, num_walkers=100000):
        self.grid_size = grid_size
        self.num_walkers = num_walkers
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 30.0
        self.seed = 0

        n = grid_size

        # Voxel grid (0=empty, 1=aggregate)
        grid_init = np.zeros((n, n, n), dtype=np.int32)
        center = n // 2
        grid_init[center, center, center] = 1  # Single seed
        self.grid = wp.array(grid_init, dtype=wp.int32)

        # Walkers
        self.walker_x = wp.zeros(num_walkers, dtype=wp.int32)
        self.walker_y = wp.zeros(num_walkers, dtype=wp.int32)
        self.walker_z = wp.zeros(num_walkers, dtype=wp.int32)
        self.walker_active = wp.zeros(num_walkers, dtype=wp.int32)

        # Tracking
        self.attached_count = wp.zeros(1, dtype=wp.int32)
        self.spawn_radius = n // 3

        # Rendering
        self.max_points = n * n * n // 8
        self.positions = wp.zeros(self.max_points, dtype=wp.vec3)
        self.point_count = wp.zeros(1, dtype=wp.int32)

        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = None

    def step(self):
        with wp.ScopedTimer("step", active=False):
            self.seed += 1

            # Respawn dead/inactive walkers
            wp.launch(
                kernel=respawn_walkers,
                dim=self.num_walkers,
                inputs=[
                    self.walker_x,
                    self.walker_y,
                    self.walker_z,
                    self.walker_active,
                    self.grid_size,
                    self.spawn_radius,
                    self.seed,
                ],
            )

            self.seed += 1

            # Walk and attach
            wp.launch(
                kernel=random_walk_and_attach,
                dim=self.num_walkers,
                inputs=[
                    self.grid,
                    self.walker_x,
                    self.walker_y,
                    self.walker_z,
                    self.walker_active,
                    self.attached_count,
                    self.seed,
                    200,  # walk steps per iteration
                ],
            )

            self.sim_time += self.frame_dt

    def get_aggregate_points(self):
        self.point_count.zero_()
        n = self.grid_size
        scale = 1.0
        offset = n * scale / 2.0

        wp.launch(
            kernel=extract_points,
            dim=(n, n, n),
            inputs=[self.grid, self.positions, self.point_count, self.max_points, scale, offset],
        )

        count = self.point_count.numpy()[0]
        return self.positions.numpy()[:count]

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            positions = self.get_aggregate_points()
            self.renderer.begin_frame(self.sim_time)
            if len(positions) > 0:
                self.renderer.render_points(
                    points=positions,
                    radius=0.5,
                    name="aggregate",
                    colors=(0.85, 0.85, 0.9),
                )
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_dla.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=500, help="Total number of frames.")
    parser.add_argument("--grid-size", type=int, default=128, help="Grid resolution per axis.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, grid_size=args.grid_size)

        for i in range(args.num_frames):
            example.step()
            example.render()

            if i % 100 == 0:
                count = example.attached_count.numpy()[0]
                print(f"Frame {i}: {count} attached particles")

        if example.renderer:
            example.renderer.save()
