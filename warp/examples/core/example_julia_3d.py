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
# Example Julia Set 3D
#
# Renders a quaternion Julia set — a 4D fractal intersected with 3D
# space. Each voxel tests whether the quaternion iteration
# z → z² + c diverges, producing organic, coral-like fractal shapes.
#
# The fractal is volume-rendered with ray marching, using the distance
# estimator for efficient empty-space skipping.
#
# Demonstrates:
#   - Quaternion arithmetic in Warp kernels
#   - Distance estimation for fractal rendering
#   - Ray marching with adaptive step size
#   - Complex mathematical expressions in GPU code
#
###########################################################################

import math

import numpy as np

import warp as wp


@wp.func
def quat_mul(a: wp.vec4, b: wp.vec4):
    """Quaternion multiplication."""
    return wp.vec4(
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    )


@wp.func
def quat_sq(z: wp.vec4):
    """Quaternion square: z * z."""
    return wp.vec4(
        z[0] * z[0] - z[1] * z[1] - z[2] * z[2] - z[3] * z[3],
        2.0 * z[0] * z[1],
        2.0 * z[0] * z[2],
        2.0 * z[0] * z[3],
    )


@wp.func
def julia_de(
    p: wp.vec3,
    c: wp.vec4,
    max_iter: int,
):
    """Distance estimator for quaternion Julia set."""
    z = wp.vec4(p[0], p[1], p[2], 0.0)
    dz = wp.vec4(1.0, 0.0, 0.0, 0.0)

    r = float(0.0)

    for _ in range(max_iter):
        # dz = 2 * z * dz
        dz = wp.vec4(
            2.0 * (z[0] * dz[0] - z[1] * dz[1] - z[2] * dz[2] - z[3] * dz[3]),
            2.0 * (z[0] * dz[1] + z[1] * dz[0] + z[2] * dz[3] - z[3] * dz[2]),
            2.0 * (z[0] * dz[2] - z[1] * dz[3] + z[2] * dz[0] + z[3] * dz[1]),
            2.0 * (z[0] * dz[3] + z[1] * dz[2] - z[2] * dz[1] + z[3] * dz[0]),
        )

        # z = z^2 + c
        z = quat_sq(z)
        z = wp.vec4(z[0] + c[0], z[1] + c[1], z[2] + c[2], z[3] + c[3])

        r = wp.length(z)
        if r > 4.0:
            break

    dr = wp.length(dz)
    if dr < 1.0e-10:
        return 0.0

    return 0.5 * r * wp.log(r) / dr


@wp.kernel
def render_julia(
    pixels: wp.array[wp.vec3],
    width: int,
    height: int,
    cam_pos: wp.vec3,
    cam_fwd: wp.vec3,
    cam_right: wp.vec3,
    cam_up: wp.vec3,
    fov: float,
    c: wp.vec4,
    max_iter: int,
):
    tid = wp.tid()

    px = tid % width
    py = tid / width

    aspect = float(width) / float(height)
    half_fov = wp.tan(fov * 0.5)

    u = (2.0 * (float(px) + 0.5) / float(width) - 1.0) * half_fov * aspect
    v = (2.0 * (float(py) + 0.5) / float(height) - 1.0) * half_fov

    ray_dir = wp.normalize(cam_fwd + cam_right * u + cam_up * v)

    # Ray march with distance estimator
    t = float(0.0)
    max_t = float(10.0)
    min_dist = float(1.0e10)
    hit = int(0)
    iterations = int(0)

    for step in range(300):
        p = cam_pos + ray_dir * t
        d = julia_de(p, c, max_iter)

        if d < min_dist:
            min_dist = d

        if d < 0.0005:
            hit = 1
            iterations = step
            break

        t = t + d * 0.7  # Slightly conservative step
        if t > max_t:
            break

    # Background gradient
    nv = float(py) / float(height)
    bg = wp.vec3(0.02, 0.02, 0.05 + nv * 0.03)

    if hit == 0:
        # Glow from near-misses
        glow = wp.exp(-min_dist * 50.0) * 0.15
        pixels[tid] = wp.vec3(bg[0] + glow * 0.3, bg[1] + glow * 0.1, bg[2] + glow * 0.5)
        return

    # Compute normal via finite differences
    p = cam_pos + ray_dir * t
    eps = 0.001
    nx = julia_de(wp.vec3(p[0] + eps, p[1], p[2]), c, max_iter) - julia_de(
        wp.vec3(p[0] - eps, p[1], p[2]), c, max_iter
    )
    ny = julia_de(wp.vec3(p[0], p[1] + eps, p[2]), c, max_iter) - julia_de(
        wp.vec3(p[0], p[1] - eps, p[2]), c, max_iter
    )
    nz = julia_de(wp.vec3(p[0], p[1], p[2] + eps), c, max_iter) - julia_de(
        wp.vec3(p[0], p[1], p[2] - eps), c, max_iter
    )
    normal = wp.normalize(wp.vec3(nx, ny, nz))

    # Lighting
    light_dir = wp.normalize(wp.vec3(0.5, 0.8, 0.6))
    diff = wp.max(wp.dot(normal, light_dir), 0.0)
    spec_dir = wp.normalize(light_dir - ray_dir)
    spec = wp.pow(wp.max(wp.dot(normal, spec_dir), 0.0), 32.0)

    # Base color from normal
    base_r = 0.5 + 0.5 * normal[0]
    base_g = 0.5 + 0.5 * normal[1]
    base_b = 0.7 + 0.3 * normal[2]

    r = base_r * (0.15 + 0.7 * diff) + spec * 0.4
    g = base_g * (0.15 + 0.7 * diff) + spec * 0.4
    b = base_b * (0.15 + 0.7 * diff) + spec * 0.4

    # Fog
    fog = wp.exp(-t * 0.3)
    r = r * fog + bg[0] * (1.0 - fog)
    g = g * fog + bg[1] * (1.0 - fog)
    b = b * fog + bg[2] * (1.0 - fog)

    pixels[tid] = wp.vec3(
        wp.pow(wp.clamp(r, 0.0, 1.0), 0.4545),
        wp.pow(wp.clamp(g, 0.0, 1.0), 0.4545),
        wp.pow(wp.clamp(b, 0.0, 1.0), 0.4545),
    )


class Example:
    def __init__(self, width=1024, height=1024):
        self.width = width
        self.height = height

        # Julia constant (produces nice organic shapes)
        self.c = wp.vec4(-0.2, 0.6, 0.2, 0.0)
        self.max_iter = 12

        # Camera orbits around origin
        self.cam_distance = 3.0
        self.cam_height = 0.5
        self.cam_angle = 0.0
        self.fov = math.radians(55.0)

        self.pixels = wp.zeros(width * height, dtype=wp.vec3)

    def render(self, angle=None):
        if angle is not None:
            self.cam_angle = angle

        with wp.ScopedTimer("render", active=False):
            cam_x = self.cam_distance * math.cos(self.cam_angle)
            cam_z = self.cam_distance * math.sin(self.cam_angle)
            cam_pos = wp.vec3(cam_x, self.cam_height, cam_z)
            cam_target = wp.vec3(0.0, 0.0, 0.0)

            fwd = wp.normalize(wp.vec3(-cam_x, -self.cam_height, -cam_z))
            world_up = wp.vec3(0.0, 1.0, 0.0)
            right = wp.normalize(wp.cross(fwd, world_up))
            up = wp.cross(right, fwd)

            wp.launch(
                kernel=render_julia,
                dim=self.width * self.height,
                inputs=[
                    self.pixels,
                    self.width,
                    self.height,
                    cam_pos,
                    fwd,
                    right,
                    up,
                    self.fov,
                    self.c,
                    self.max_iter,
                ],
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode.",
    )
    parser.add_argument("--width", type=int, default=1024, help="Image width.")
    parser.add_argument("--height", type=int, default=1024, help="Image height.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(width=args.width, height=args.height)
        example.render(angle=0.5)

        if not args.headless:
            import matplotlib.pyplot as plt

            img = example.pixels.numpy().reshape((example.height, example.width, 3))
            plt.imshow(img, origin="lower", interpolation="antialiased")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
