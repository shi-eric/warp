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
# Example Reaction-Diffusion
#
# Implements the Gray-Scott reaction-diffusion model on a 3D grid.
# Two chemical species (U and V) interact and diffuse, producing
# complex self-organizing patterns such as spots, stripes, and
# labyrinthine structures. The isosurface of V is extracted using
# marching cubes to visualize the 3D patterns.
#
# Demonstrates:
#   - 3D stencil computation with multidimensional kernels
#   - wp.launch() with 3D grid dimensions
#   - Marching cubes isosurface extraction (wp.MarchingCubes)
#   - Double-buffered simulation with array swapping
#
# Reference:
#   Pearson, J.E. "Complex Patterns in a Simple System."
#   Science 261.5118 (1993): 189-192.
#
###########################################################################

import numpy as np

import warp as wp
import warp.render


@wp.kernel
def reaction_diffusion_step(
    U: wp.array3d[wp.float32],
    V: wp.array3d[wp.float32],
    new_U: wp.array3d[wp.float32],
    new_V: wp.array3d[wp.float32],
    Du: float,
    Dv: float,
    feed: float,
    kill: float,
    dt: float,
):
    i, j, k = wp.tid()

    nx = U.shape[0]
    ny = U.shape[1]
    nz = U.shape[2]

    # Periodic boundary conditions
    ip = (i + 1) % nx
    im = (i - 1 + nx) % nx
    jp = (j + 1) % ny
    jm = (j - 1 + ny) % ny
    kp = (k + 1) % nz
    km = (k - 1 + nz) % nz

    u = U[i, j, k]
    v = V[i, j, k]

    # 3D Laplacian (7-point stencil)
    lap_u = U[ip, j, k] + U[im, j, k] + U[i, jp, k] + U[i, jm, k] + U[i, j, kp] + U[i, j, km] - 6.0 * u
    lap_v = V[ip, j, k] + V[im, j, k] + V[i, jp, k] + V[i, jm, k] + V[i, j, kp] + V[i, j, km] - 6.0 * v

    # Gray-Scott reaction: U + 2V -> 3V
    uvv = u * v * v

    new_U[i, j, k] = u + dt * (Du * lap_u - uvv + feed * (1.0 - u))
    new_V[i, j, k] = v + dt * (Dv * lap_v + uvv - (feed + kill) * v)


class Example:
    def __init__(self, stage_path="example_reaction_diffusion.usd", grid_size=80):
        self.grid_size = grid_size
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 30.0

        # Gray-Scott parameters (coral/sponge growth)
        self.Du = 0.16
        self.Dv = 0.08
        self.feed = 0.04
        self.kill = 0.06
        self.sim_dt = 0.25
        self.substeps = 100

        n = grid_size

        # Initialize U=1, V=0 everywhere
        u_init = np.ones((n, n, n), dtype=np.float32)
        v_init = np.zeros((n, n, n), dtype=np.float32)

        # Seed V with small spherical seeds in center region
        rng = np.random.default_rng(42)
        center = n // 2
        spread = n // 3
        for _ in range(8):
            cx = rng.integers(center - spread, center + spread)
            cy = rng.integers(center - spread, center + spread)
            cz = rng.integers(center - spread, center + spread)
            r = rng.integers(2, 4)
            # Create spherical seed
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    for dz in range(-r, r+1):
                        if dx*dx + dy*dy + dz*dz <= r*r:
                            x = min(max(cx+dx, 0), n-1)
                            y = min(max(cy+dy, 0), n-1)
                            z = min(max(cz+dz, 0), n-1)
                            v_init[x, y, z] = 0.5
                            u_init[x, y, z] = 0.25

        self.U = wp.array(u_init, dtype=wp.float32)
        self.V = wp.array(v_init, dtype=wp.float32)
        self.new_U = wp.zeros_like(self.U)
        self.new_V = wp.zeros_like(self.V)

        # Marching cubes for isosurface extraction
        self.mc = wp.MarchingCubes(nx=n, ny=n, nz=n)

        # Renderer
        if stage_path and stage_path.endswith((".usd", ".usda", ".usdc")):
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = wp.render.NativeRenderer(512, 512)
            self.renderer.setup_camera(pos=(100, 60, 100), target=(40, 40, 40), fov=50)

    def step(self):
        with wp.ScopedTimer("step", active=False):
            n = self.grid_size
            for _ in range(self.substeps):
                wp.launch(
                    kernel=reaction_diffusion_step,
                    dim=(n, n, n),
                    inputs=[
                        self.U,
                        self.V,
                        self.new_U,
                        self.new_V,
                        self.Du,
                        self.Dv,
                        self.feed,
                        self.kill,
                        self.sim_dt,
                    ],
                )
                self.U, self.new_U = self.new_U, self.U
                self.V, self.new_V = self.new_V, self.V

            self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            # Extract isosurface of V at threshold
            self.mc.surface(self.V, threshold=0.15)

            self.renderer.begin_frame(self.sim_time)
            if self.mc.verts is not None and len(self.mc.verts) > 0:
                self.renderer.render_mesh(
                    points=self.mc.verts,
                    indices=self.mc.indices,
                    name="reaction_diffusion",
                    colors=(0.3, 0.8, 0.4),
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
    parser.add_argument("--num-frames", type=int, default=100, help="Total number of frames.")
    parser.add_argument("--grid-size", type=int, default=80, help="Grid resolution per axis.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, grid_size=args.grid_size)

        for i in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            if hasattr(example.renderer, 'save'):
                example.renderer.save()
            if hasattr(example.renderer, 'save_image'):
                example.renderer.save_image("example_reaction_diffusion.png")
                print("Saved example_reaction_diffusion.png")
