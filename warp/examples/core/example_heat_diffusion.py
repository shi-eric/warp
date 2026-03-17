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
# Example Heat Diffusion
#
# Solves the 3D heat equation on a cubic domain using an explicit
# finite-difference scheme. Heat sources inject energy at fixed
# points, and the temperature field diffuses through the volume.
#
# The temperature field is visualized as colored isosurfaces at
# different temperature thresholds, creating a nested-shell effect.
#
# Demonstrates:
#   - 3D heat (diffusion) equation with explicit time stepping
#   - 7-point Laplacian stencil
#   - Multiple temperature sources
#   - Multi-level isosurface extraction
#   - Color-mapped rendering
#
###########################################################################

import numpy as np

import warp as wp
import warp.render


@wp.kernel
def heat_diffuse(
    current: wp.array3d[wp.float32],
    output: wp.array3d[wp.float32],
    alpha: float,
    dt: float,
):
    """Explicit finite-difference heat equation step."""
    i, j, k = wp.tid()

    nx = current.shape[0]
    ny = current.shape[1]
    nz = current.shape[2]

    # Insulating boundaries
    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
        output[i, j, k] = current[i, j, k]
        return

    # 7-point Laplacian
    u = current[i, j, k]
    lap = (
        current[i + 1, j, k]
        + current[i - 1, j, k]
        + current[i, j + 1, k]
        + current[i, j - 1, k]
        + current[i, j, k + 1]
        + current[i, j, k - 1]
        - 6.0 * u
    )

    output[i, j, k] = u + alpha * dt * lap


@wp.kernel
def apply_heat_source(
    field: wp.array3d[wp.float32],
    cx: int,
    cy: int,
    cz: int,
    radius: int,
    temperature: float,
):
    i, j, k = wp.tid()

    di = i - cx
    dj = j - cy
    dk = k - cz

    if di * di + dj * dj + dk * dk < radius * radius:
        field[i, j, k] = temperature


class Example:
    def __init__(self, stage_path="example_heat_diffusion.usd", grid_size=128):
        self.grid_size = grid_size
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 30.0
        self.substeps = 20

        n = grid_size

        # Heat equation parameters
        self.alpha = 0.15  # Thermal diffusivity
        self.dt = 0.05

        # Temperature fields (double-buffered)
        self.current = wp.zeros((n, n, n), dtype=wp.float32)
        self.output = wp.zeros((n, n, n), dtype=wp.float32)

        # Heat sources: 3 sources at different positions and temperatures
        self.sources = [
            (n // 4, n // 2, n // 4, 4, 1.0),  # (cx, cy, cz, radius, temperature)
            (3 * n // 4, n // 2, 3 * n // 4, 4, 0.8),
            (n // 2, n // 4, n // 2, 3, 0.6),
        ]

        # Marching cubes for isosurface rendering
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
                # Apply heat sources
                for cx, cy, cz, radius, temp in self.sources:
                    wp.launch(
                        kernel=apply_heat_source,
                        dim=(n, n, n),
                        inputs=[self.current, cx, cy, cz, radius, temp],
                    )

                # Diffuse
                wp.launch(
                    kernel=heat_diffuse,
                    dim=(n, n, n),
                    inputs=[self.current, self.output, self.alpha, self.dt],
                )

                # Swap
                self.current, self.output = self.output, self.current

            self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            self.renderer.begin_frame(self.sim_time)

            # Render multiple isosurface levels with different colors
            levels = [
                (0.3, (0.9, 0.2, 0.1)),  # Hot core: red
                (0.15, (0.9, 0.6, 0.1)),  # Warm: orange
                (0.05, (0.2, 0.4, 0.9)),  # Cool edge: blue
            ]

            for idx, (threshold, color) in enumerate(levels):
                self.mc.surface(self.current, threshold=threshold)
                if self.mc.verts is not None and len(self.mc.verts) > 0:
                    self.renderer.render_mesh(
                        points=self.mc.verts,
                        indices=self.mc.indices,
                        name=f"iso_{idx}",
                        colors=color,
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
    parser.add_argument("--num-frames", type=int, default=200, help="Total number of frames.")
    parser.add_argument("--grid-size", type=int, default=128, help="Grid resolution per axis.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, grid_size=args.grid_size)

        for i in range(args.num_frames):
            example.step()
            example.render()

            if i % 50 == 0:
                t = example.current.numpy()
                print(f"Frame {i}: T range [{t.min():.4f}, {t.max():.4f}]")

        if example.renderer:
            if hasattr(example.renderer, 'save'):
                example.renderer.save()
            if hasattr(example.renderer, 'save_image'):
                example.renderer.save_image("example_heat_diffusion.png")
                print("Saved example_heat_diffusion.png")
