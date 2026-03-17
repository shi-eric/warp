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
# Example Cellular Automata 3D
#
# Implements a 3D cellular automaton (3D Game of Life variant) where
# cells live or die based on the number of living neighbors in a 3D
# Moore neighborhood (26 neighbors). This produces complex evolving
# 3D structures.
#
# The rule used is "4555" (born with 4 neighbors, survive with 5 neighbors)
# which tends to produce stable, interesting structures.
#
# Demonstrates:
#   - 3D stencil computation on integer arrays
#   - wp.launch() with 3D grid dimensions
#   - Rendering point clouds from sparse cell positions
#   - Efficient GPU-parallel neighbor counting
#
###########################################################################

import numpy as np

import warp as wp
import warp.render


@wp.kernel
def count_neighbors_and_update(
    cells: wp.array3d[wp.int32],
    new_cells: wp.array3d[wp.int32],
    birth_min: int,
    birth_max: int,
    survive_min: int,
    survive_max: int,
):
    i, j, k = wp.tid()

    nx = cells.shape[0]
    ny = cells.shape[1]
    nz = cells.shape[2]

    # Count living neighbors in 3D Moore neighborhood
    neighbors = int(0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue

                # Wrap around (periodic boundaries)
                ni = (i + di + nx) % nx
                nj = (j + dj + ny) % ny
                nk = (k + dk + nz) % nz

                neighbors += cells[ni, nj, nk]

    alive = cells[i, j, k]

    # Apply birth/survival rules
    if alive == 0:
        # Birth rule: dead cell becomes alive
        if neighbors >= birth_min and neighbors <= birth_max:
            new_cells[i, j, k] = 1
        else:
            new_cells[i, j, k] = 0
    else:
        # Survival rule: live cell stays alive
        if neighbors >= survive_min and neighbors <= survive_max:
            new_cells[i, j, k] = 1
        else:
            new_cells[i, j, k] = 0


@wp.kernel
def extract_live_cells(
    cells: wp.array3d[wp.int32],
    positions: wp.array[wp.vec3],
    count: wp.array[wp.int32],
    max_cells: int,
    scale: float,
):
    i, j, k = wp.tid()

    if cells[i, j, k] == 1:
        idx = wp.atomic_add(count, 0, 1)
        if idx < max_cells:
            positions[idx] = wp.vec3(float(i) * scale, float(j) * scale, float(k) * scale)


class Example:
    def __init__(self, stage_path="example_cellular_automata_3d.usd", grid_size=64):
        self.grid_size = grid_size
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 30.0

        # 3D Game of Life variant: "4555" rule (4 to birth, 5-5 to survive)
        # This produces stable structures with interesting patterns
        self.birth_min = 4
        self.birth_max = 4
        self.survive_min = 5
        self.survive_max = 5

        n = grid_size

        # Initialize with random live cells in center region
        rng = np.random.default_rng(42)
        cells_init = np.zeros((n, n, n), dtype=np.int32)

        # Random fill in center cubic region
        margin = n // 4
        center_slice = slice(margin, n - margin)
        cells_init[center_slice, center_slice, center_slice] = rng.integers(
            0, 2, size=(n - 2 * margin, n - 2 * margin, n - 2 * margin), dtype=np.int32
        )

        self.cells = wp.array(cells_init, dtype=wp.int32)
        self.new_cells = wp.zeros_like(self.cells)

        # For rendering: pre-allocate arrays
        self.max_cells = n * n * n // 4  # Max 25% occupancy
        self.positions = wp.zeros(self.max_cells, dtype=wp.vec3)
        self.cell_count = wp.zeros(1, dtype=wp.int32)
        self.point_radius = 0.4

        # Renderer
        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = None

    def step(self):
        with wp.ScopedTimer("step", active=False):
            n = self.grid_size
            wp.launch(
                kernel=count_neighbors_and_update,
                dim=(n, n, n),
                inputs=[
                    self.cells,
                    self.new_cells,
                    self.birth_min,
                    self.birth_max,
                    self.survive_min,
                    self.survive_max,
                ],
            )
            self.cells, self.new_cells = self.new_cells, self.cells
            self.sim_time += self.frame_dt

    def get_live_positions(self):
        """Extract positions of all live cells for rendering."""
        self.cell_count.zero_()
        n = self.grid_size
        scale = 1.0

        wp.launch(
            kernel=extract_live_cells,
            dim=(n, n, n),
            inputs=[self.cells, self.positions, self.cell_count, self.max_cells, scale],
        )

        count = self.cell_count.numpy()[0]
        return self.positions.numpy()[:count]

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            positions = self.get_live_positions()
            self.renderer.begin_frame(self.sim_time)
            if len(positions) > 0:
                self.renderer.render_points(
                    points=positions,
                    radius=self.point_radius,
                    name="cells",
                    colors=(0.2, 0.8, 0.3),
                )
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_cellular_automata_3d.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=200, help="Total number of frames.")
    parser.add_argument("--grid-size", type=int, default=64, help="Grid resolution per axis.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, grid_size=args.grid_size)

        for i in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
