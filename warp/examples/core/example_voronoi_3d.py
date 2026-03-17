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
# Example 3D Voronoi
#
# Computes a 3D Voronoi diagram from random seed points and visualizes
# the cell boundaries using volume rendering. Each voxel is assigned
# to its nearest seed; the density field is high near cell boundaries
# where the distance to the two closest seeds is similar.
#
# Demonstrates:
#   - Brute-force nearest-neighbor search in a kernel
#   - 3D distance field computation
#   - Boundary detection (second-closest vs closest distance)
#   - Volume rendering of implicit surfaces
#
###########################################################################

import math

import numpy as np

import warp as wp


@wp.kernel
def compute_voronoi(
    seeds: wp.array[wp.vec3],
    volume: wp.array3d[wp.float32],
    num_seeds: int,
    vol_size: float,
):
    """Compute Voronoi boundary field: high values at cell boundaries."""
    i, j, k = wp.tid()

    nx = volume.shape[0]
    ny = volume.shape[1]
    nz = volume.shape[2]

    # Map voxel to world position
    px = (float(i) / float(nx - 1)) * vol_size
    py = (float(j) / float(ny - 1)) * vol_size
    pz = (float(k) / float(nz - 1)) * vol_size

    p = wp.vec3(px, py, pz)

    # Find closest and second-closest seed
    d1 = float(1.0e10)  # Closest distance
    d2 = float(1.0e10)  # Second closest distance

    for s in range(num_seeds):
        diff = p - seeds[s]
        d = wp.length(diff)

        if d < d1:
            d2 = d1
            d1 = d
        elif d < d2:
            d2 = d

    # Boundary detection: density is high where d2 - d1 is small
    boundary_width = vol_size / float(nx) * 2.5  # ~2.5 voxels wide
    boundary = wp.exp(-(d2 - d1) / boundary_width)

    volume[i, j, k] = boundary


@wp.kernel
def render_voronoi_volume(
    volume: wp.array3d[wp.float32],
    pixels: wp.array[wp.vec3],
    cam_pos: wp.vec3,
    cam_fwd: wp.vec3,
    cam_right: wp.vec3,
    cam_up: wp.vec3,
    fov: float,
    width: int,
    height: int,
    vol_size: float,
    step_size: float,
):
    tid = wp.tid()

    px = tid % width
    py = tid / width

    aspect = float(width) / float(height)
    half_fov = wp.tan(fov * 0.5)

    u = (2.0 * (float(px) + 0.5) / float(width) - 1.0) * half_fov * aspect
    v = (2.0 * (float(py) + 0.5) / float(height) - 1.0) * half_fov

    ray_dir = wp.normalize(cam_fwd + cam_right * u + cam_up * v)

    # Ray-AABB intersection (box [0, vol_size])
    inv_x = 1.0 / (ray_dir[0] + 1.0e-8)
    inv_y = 1.0 / (ray_dir[1] + 1.0e-8)
    inv_z = 1.0 / (ray_dir[2] + 1.0e-8)

    t1x = (0.0 - cam_pos[0]) * inv_x
    t2x = (vol_size - cam_pos[0]) * inv_x
    t1y = (0.0 - cam_pos[1]) * inv_y
    t2y = (vol_size - cam_pos[1]) * inv_y
    t1z = (0.0 - cam_pos[2]) * inv_z
    t2z = (vol_size - cam_pos[2]) * inv_z

    t_near = wp.max(wp.min(t1x, t2x), wp.max(wp.min(t1y, t2y), wp.min(t1z, t2z)))
    t_far = wp.min(wp.max(t1x, t2x), wp.min(wp.max(t1y, t2y), wp.max(t1z, t2z)))

    nv = float(py) / float(height)
    bg = wp.vec3(0.03, 0.03, 0.06 + nv * 0.04)

    if t_near >= t_far or t_far < 0.0:
        pixels[tid] = bg
        return

    t_near = wp.max(t_near, 0.0)

    nx = volume.shape[0]
    ny = volume.shape[1]
    nz = volume.shape[2]

    color = wp.vec3(0.0, 0.0, 0.0)
    alpha = float(0.0)
    t = t_near

    while t < t_far and alpha < 0.95:
        pos = cam_pos + ray_dir * t

        # Sample volume
        gx = wp.clamp(pos[0] / vol_size * float(nx - 1), 0.0, float(nx - 1))
        gy = wp.clamp(pos[1] / vol_size * float(ny - 1), 0.0, float(ny - 1))
        gz = wp.clamp(pos[2] / vol_size * float(nz - 1), 0.0, float(nz - 1))

        ix = int(gx)
        iy = int(gy)
        iz = int(gz)
        ix = wp.min(ix, nx - 1)
        iy = wp.min(iy, ny - 1)
        iz = wp.min(iz, nz - 1)

        d = volume[ix, iy, iz]

        if d > 0.1:
            a = 1.0 - wp.exp(-d * step_size * 12.0)

            # Blue-white wireframe color
            r = 0.3 + d * 0.7
            g = 0.4 + d * 0.6
            b = 0.8 + d * 0.2
            r = wp.min(r, 1.0)
            g = wp.min(g, 1.0)

            w = a * (1.0 - alpha)
            color = color + wp.vec3(r, g, b) * w
            alpha = alpha + w

        t = t + step_size

    color = color + bg * (1.0 - alpha)
    pixels[tid] = wp.vec3(
        wp.pow(wp.clamp(color[0], 0.0, 1.0), 0.4545),
        wp.pow(wp.clamp(color[1], 0.0, 1.0), 0.4545),
        wp.pow(wp.clamp(color[2], 0.0, 1.0), 0.4545),
    )


class Example:
    def __init__(self, width=1024, height=1024, grid_size=128, num_seeds=30):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.vol_size = 10.0

        n = grid_size

        # Random seed points
        rng = np.random.default_rng(42)
        seeds = rng.uniform(0.5, self.vol_size - 0.5, (num_seeds, 3)).astype(np.float32)
        self.seeds = wp.array(seeds, dtype=wp.vec3)
        self.num_seeds = num_seeds

        # Volume
        self.volume = wp.zeros((n, n, n), dtype=wp.float32)

        # Compute Voronoi field
        wp.launch(
            kernel=compute_voronoi,
            dim=(n, n, n),
            inputs=[self.seeds, self.volume, self.num_seeds, self.vol_size],
        )

        # Camera
        self.cam_pos = wp.vec3(self.vol_size * 1.5, self.vol_size * 0.8, self.vol_size * 1.5)
        self.cam_target = wp.vec3(self.vol_size / 2, self.vol_size / 2, self.vol_size / 2)
        self.fov = math.radians(50.0)
        self.step_size = self.vol_size / float(n)

        self.pixels = wp.zeros(width * height, dtype=wp.vec3)

    def render(self):
        with wp.ScopedTimer("render"):
            cam_fwd = wp.normalize(wp.vec3(
                self.cam_target[0] - self.cam_pos[0],
                self.cam_target[1] - self.cam_pos[1],
                self.cam_target[2] - self.cam_pos[2],
            ))
            world_up = wp.vec3(0.0, 1.0, 0.0)
            cam_right = wp.normalize(wp.cross(cam_fwd, world_up))
            cam_up = wp.cross(cam_right, cam_fwd)

            wp.launch(
                kernel=render_voronoi_volume,
                dim=self.width * self.height,
                inputs=[
                    self.volume,
                    self.pixels,
                    self.cam_pos,
                    cam_fwd,
                    cam_right,
                    cam_up,
                    self.fov,
                    self.width,
                    self.height,
                    self.vol_size,
                    self.step_size,
                ],
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )
    parser.add_argument("--width", type=int, default=1024, help="Output image width in pixels.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height in pixels.")
    parser.add_argument("--grid-size", type=int, default=128, help="Volume grid resolution per axis.")
    parser.add_argument("--num-seeds", type=int, default=30, help="Number of Voronoi seed points.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            width=args.width,
            height=args.height,
            grid_size=args.grid_size,
            num_seeds=args.num_seeds,
        )
        example.render()

        if not args.headless:
            import matplotlib.pyplot as plt

            plt.imshow(
                example.pixels.numpy().reshape((example.height, example.width, 3)),
                origin="lower",
                interpolation="antialiased",
            )
            plt.axis("off")
            plt.tight_layout()
            plt.show()
