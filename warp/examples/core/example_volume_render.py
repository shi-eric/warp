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
# Example Volume Render
#
# Implements direct volume rendering (DVR) by raymarching through a 3D
# density field. Each pixel casts a ray through the volume and
# accumulates color and opacity using front-to-back compositing with
# a customizable transfer function.
#
# The volume data is a Mandelbulb fractal — a 3D generalization of the
# Mandelbrot set defined by iterating a power mapping in spherical
# coordinates. The fractal's intricate self-similar structure makes it
# an ideal subject for volumetric rendering.
#
# Demonstrates:
#   - Ray-volume intersection (AABB)
#   - Trilinear interpolation for volume sampling
#   - Front-to-back alpha compositing
#   - Transfer function mapping (density → color, opacity)
#   - Camera ray generation from projection parameters
#
###########################################################################

import math

import numpy as np

import warp as wp


@wp.func
def intersect_aabb(
    ray_origin: wp.vec3,
    ray_dir_inv: wp.vec3,
    box_min: wp.vec3,
    box_max: wp.vec3,
):
    """Compute ray-AABB intersection, returning (t_near, t_far)."""
    t1x = (box_min[0] - ray_origin[0]) * ray_dir_inv[0]
    t2x = (box_max[0] - ray_origin[0]) * ray_dir_inv[0]
    t1y = (box_min[1] - ray_origin[1]) * ray_dir_inv[1]
    t2y = (box_max[1] - ray_origin[1]) * ray_dir_inv[1]
    t1z = (box_min[2] - ray_origin[2]) * ray_dir_inv[2]
    t2z = (box_max[2] - ray_origin[2]) * ray_dir_inv[2]

    t_near = wp.max(wp.min(t1x, t2x), wp.max(wp.min(t1y, t2y), wp.min(t1z, t2z)))
    t_far = wp.min(wp.max(t1x, t2x), wp.min(wp.max(t1y, t2y), wp.max(t1z, t2z)))

    return t_near, t_far


@wp.func
def sample_volume(
    volume: wp.array3d[wp.float32],
    p: wp.vec3,
    vol_min: wp.vec3,
    vol_max: wp.vec3,
):
    """Sample the volume with trilinear interpolation in world coordinates."""
    nx = volume.shape[0]
    ny = volume.shape[1]
    nz = volume.shape[2]

    # Map world position to grid coordinates
    extent = vol_max - vol_min
    gx = ((p[0] - vol_min[0]) / extent[0]) * float(nx - 1)
    gy = ((p[1] - vol_min[1]) / extent[1]) * float(ny - 1)
    gz = ((p[2] - vol_min[2]) / extent[2]) * float(nz - 1)

    # Clamp to grid bounds
    gx = wp.clamp(gx, 0.0, float(nx - 1))
    gy = wp.clamp(gy, 0.0, float(ny - 1))
    gz = wp.clamp(gz, 0.0, float(nz - 1))

    # Integer coordinates
    x0 = int(wp.floor(gx))
    y0 = int(wp.floor(gy))
    z0 = int(wp.floor(gz))
    x1 = wp.min(x0 + 1, nx - 1)
    y1 = wp.min(y0 + 1, ny - 1)
    z1 = wp.min(z0 + 1, nz - 1)

    # Fractional parts
    fx = gx - float(x0)
    fy = gy - float(y0)
    fz = gz - float(z0)

    # Trilinear interpolation
    c00 = volume[x0, y0, z0] * (1.0 - fx) + volume[x1, y0, z0] * fx
    c01 = volume[x0, y0, z1] * (1.0 - fx) + volume[x1, y0, z1] * fx
    c10 = volume[x0, y1, z0] * (1.0 - fx) + volume[x1, y1, z0] * fx
    c11 = volume[x0, y1, z1] * (1.0 - fx) + volume[x1, y1, z1] * fx

    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy

    return c0 * (1.0 - fz) + c1 * fz


@wp.func
def transfer_function(density: float):
    """Map density to color and opacity.

    Returns (color_r, color_g, color_b, alpha).
    Low density: transparent blue, high density: opaque orange/white.
    """
    # Opacity ramp
    alpha = wp.clamp(density * 3.0, 0.0, 1.0)

    # Color: blue (cold) → orange (warm) → white (hot)
    if density < 0.3:
        t = density / 0.3
        r = 0.1 + t * 0.5
        g = 0.2 + t * 0.2
        b = 0.6 - t * 0.2
    elif density < 0.7:
        t = (density - 0.3) / 0.4
        r = 0.6 + t * 0.35
        g = 0.4 + t * 0.3
        b = 0.4 - t * 0.25
    else:
        t = (density - 0.7) / 0.3
        r = 0.95 + t * 0.05
        g = 0.7 + t * 0.3
        b = 0.15 + t * 0.85

    return r, g, b, alpha


@wp.kernel
def volume_render(
    volume: wp.array3d[wp.float32],
    pixels: wp.array[wp.vec3],
    cam_pos: wp.vec3,
    cam_fwd: wp.vec3,
    cam_right: wp.vec3,
    cam_up: wp.vec3,
    fov: float,
    width: int,
    height: int,
    vol_min: wp.vec3,
    vol_max: wp.vec3,
    step_size: float,
    density_scale: float,
):
    tid = wp.tid()

    px = tid % width
    py = tid / width

    # Generate ray from pixel coordinates
    aspect = float(width) / float(height)
    half_fov = wp.tan(fov * 0.5)

    u = (2.0 * (float(px) + 0.5) / float(width) - 1.0) * half_fov * aspect
    v = (2.0 * (float(py) + 0.5) / float(height) - 1.0) * half_fov

    ray_dir = wp.normalize(cam_fwd + cam_right * u + cam_up * v)

    # Precompute inverse direction for AABB intersection
    ray_dir_inv = wp.vec3(
        1.0 / (ray_dir[0] + 1.0e-8),
        1.0 / (ray_dir[1] + 1.0e-8),
        1.0 / (ray_dir[2] + 1.0e-8),
    )

    # Ray-AABB intersection
    t_near, t_far = intersect_aabb(cam_pos, ray_dir_inv, vol_min, vol_max)

    # Background color (dark gradient)
    nv = float(py) / float(height)
    bg = wp.vec3(0.05 + nv * 0.1, 0.05 + nv * 0.1, 0.1 + nv * 0.15)

    if t_near >= t_far or t_far < 0.0:
        pixels[tid] = bg
        return

    t_near = wp.max(t_near, 0.0)

    # Front-to-back compositing
    color = wp.vec3(0.0, 0.0, 0.0)
    accumulated_alpha = float(0.0)

    t = t_near
    while t < t_far and accumulated_alpha < 0.98:
        pos = cam_pos + ray_dir * t

        density = sample_volume(volume, pos, vol_min, vol_max) * density_scale

        if density > 0.01:
            r, g, b, alpha = transfer_function(density)
            alpha = 1.0 - wp.exp(-alpha * step_size * 8.0)

            # Front-to-back compositing
            weight = alpha * (1.0 - accumulated_alpha)
            color = color + wp.vec3(r, g, b) * weight
            accumulated_alpha = accumulated_alpha + weight

        t = t + step_size

    # Blend with background
    color = color + bg * (1.0 - accumulated_alpha)

    # Gamma correction
    pixels[tid] = wp.vec3(
        wp.pow(wp.clamp(color[0], 0.0, 1.0), 0.4545),
        wp.pow(wp.clamp(color[1], 0.0, 1.0), 0.4545),
        wp.pow(wp.clamp(color[2], 0.0, 1.0), 0.4545),
    )


@wp.kernel
def generate_mandelbulb_volume(
    volume: wp.array3d[wp.float32],
    power: float,
    max_iter: int,
    bailout: float,
):
    """Generate a Mandelbulb fractal distance field on a 3D grid.

    The Mandelbulb is a 3D fractal defined by iterating a power mapping
    in spherical coordinates. The volume stores an approximate density
    based on how quickly points escape to infinity.
    """
    i, j, k = wp.tid()

    nx = volume.shape[0]
    ny = volume.shape[1]
    nz = volume.shape[2]

    # Map grid to [-1.5, 1.5] cube
    x = 3.0 * float(i) / float(nx - 1) - 1.5
    y = 3.0 * float(j) / float(ny - 1) - 1.5
    z = 3.0 * float(k) / float(nz - 1) - 1.5

    # Mandelbulb iteration
    zx = x
    zy = y
    zz = z
    dr = float(1.0)
    r = float(0.0)

    for _ in range(max_iter):
        r = wp.sqrt(zx * zx + zy * zy + zz * zz)
        if r > bailout:
            break

        # Convert to spherical coordinates
        theta = wp.acos(zz / r)
        phi = wp.atan2(zy, zx)

        # Scale the running derivative
        dr = wp.pow(r, power - 1.0) * power * dr + 1.0

        # Apply the power mapping
        zr = wp.pow(r, power)
        theta = theta * power
        phi = phi * power

        # Convert back to Cartesian
        zx = zr * wp.sin(theta) * wp.cos(phi) + x
        zy = zr * wp.sin(theta) * wp.sin(phi) + y
        zz = zr * wp.cos(theta) + z

    # Distance estimator
    dist = 0.5 * wp.log(r) * r / dr

    # Convert distance to density (closer = denser)
    density = wp.clamp(1.0 - dist * 15.0, 0.0, 1.0)
    volume[i, j, k] = density


class Example:
    def __init__(self, width=1024, height=1024, grid_size=128):
        self.width = width
        self.height = height
        self.grid_size = grid_size

        n = grid_size

        # Camera
        self.cam_pos = wp.vec3(2.0, 1.5, 2.5)
        self.cam_target = wp.vec3(0.0, 0.0, 0.0)
        self.fov = math.radians(50.0)

        # Volume bounds in world space (centered at origin)
        self.vol_min = wp.vec3(-1.5, -1.5, -1.5)
        self.vol_max = wp.vec3(1.5, 1.5, 1.5)
        self.step_size = 3.0 / float(n)  # Roughly 1 step per voxel
        self.density_scale = 1.0

        # Generate Mandelbulb fractal volume
        self.volume = wp.zeros((n, n, n), dtype=wp.float32)

        wp.launch(
            kernel=generate_mandelbulb_volume,
            dim=(n, n, n),
            inputs=[self.volume, 8.0, 15, 2.0],
        )

        # Pixel buffer
        self.pixels = wp.zeros(width * height, dtype=wp.vec3)

    def render(self):
        with wp.ScopedTimer("render"):
            # Compute camera basis vectors
            cam_fwd = wp.normalize(
                wp.vec3(
                    self.cam_target[0] - self.cam_pos[0],
                    self.cam_target[1] - self.cam_pos[1],
                    self.cam_target[2] - self.cam_pos[2],
                )
            )
            world_up = wp.vec3(0.0, 1.0, 0.0)
            cam_right = wp.normalize(wp.cross(cam_fwd, world_up))
            cam_up = wp.cross(cam_right, cam_fwd)

            wp.launch(
                kernel=volume_render,
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
                    self.vol_min,
                    self.vol_max,
                    self.step_size,
                    self.density_scale,
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

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(width=args.width, height=args.height, grid_size=args.grid_size)
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
