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
# Example Strange Attractor
#
# Traces millions of particles through a strange attractor (Lorenz,
# Thomas, or Aizawa systems) and accumulates their trajectories into
# a 3D density volume. The resulting field is then volume-rendered
# with front-to-back compositing to reveal the attractor's intricate
# structure.
#
# Demonstrates:
#   - Massively parallel ODE integration
#   - Atomic histogram accumulation (wp.atomic_add on a 3D grid)
#   - Volume rendering via raymarching
#   - Combining simulation + visualization in a single pipeline
#
###########################################################################

import math

import numpy as np

import warp as wp


# ---- Attractor dynamics ----


@wp.func
def lorenz(x: float, y: float, z: float, sigma: float, rho: float, beta: float):
    """Lorenz system: the classic butterfly attractor."""
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz


@wp.func
def thomas(x: float, y: float, z: float, b: float):
    """Thomas' cyclically symmetric attractor."""
    dx = wp.sin(y) - b * x
    dy = wp.sin(z) - b * y
    dz = wp.sin(x) - b * z
    return dx, dy, dz


@wp.func
def aizawa(x: float, y: float, z: float):
    """Aizawa attractor — a chaotic system with a torus-like structure."""
    a = 0.95
    b = 0.7
    c = 0.6
    d = 3.5
    e = 0.25
    f = 0.1
    dx = (z - b) * x - d * y
    dy = d * x + (z - b) * y
    dz = c + a * z - z * z * z / 3.0 - (x * x + y * y) * (1.0 + e * z) + f * z * x * x * x
    return dx, dy, dz


# ---- Particle tracing kernel ----


@wp.kernel
def trace_particles(
    pos_x: wp.array[wp.float32],
    pos_y: wp.array[wp.float32],
    pos_z: wp.array[wp.float32],
    volume: wp.array3d[wp.float32],
    attractor_type: int,
    dt: float,
    steps: int,
    vol_min_x: float,
    vol_min_y: float,
    vol_min_z: float,
    vol_max_x: float,
    vol_max_y: float,
    vol_max_z: float,
):
    tid = wp.tid()

    x = pos_x[tid]
    y = pos_y[tid]
    z = pos_z[tid]

    nx = volume.shape[0]
    ny = volume.shape[1]
    nz = volume.shape[2]

    extent_x = vol_max_x - vol_min_x
    extent_y = vol_max_y - vol_min_y
    extent_z = vol_max_z - vol_min_z

    for _ in range(steps):
        # Compute derivatives based on attractor type
        if attractor_type == 0:
            # Lorenz
            dx, dy, dz = lorenz(x, y, z, 10.0, 28.0, 8.0 / 3.0)
        elif attractor_type == 1:
            # Thomas
            dx, dy, dz = thomas(x, y, z, 0.208186)
        else:
            # Aizawa
            dx, dy, dz = aizawa(x, y, z)

        # RK4 integration for accuracy
        k1x, k1y, k1z = dx, dy, dz

        x2 = x + k1x * dt * 0.5
        y2 = y + k1y * dt * 0.5
        z2 = z + k1z * dt * 0.5
        if attractor_type == 0:
            k2x, k2y, k2z = lorenz(x2, y2, z2, 10.0, 28.0, 8.0 / 3.0)
        elif attractor_type == 1:
            k2x, k2y, k2z = thomas(x2, y2, z2, 0.208186)
        else:
            k2x, k2y, k2z = aizawa(x2, y2, z2)

        x3 = x + k2x * dt * 0.5
        y3 = y + k2y * dt * 0.5
        z3 = z + k2z * dt * 0.5
        if attractor_type == 0:
            k3x, k3y, k3z = lorenz(x3, y3, z3, 10.0, 28.0, 8.0 / 3.0)
        elif attractor_type == 1:
            k3x, k3y, k3z = thomas(x3, y3, z3, 0.208186)
        else:
            k3x, k3y, k3z = aizawa(x3, y3, z3)

        x4 = x + k3x * dt
        y4 = y + k3y * dt
        z4 = z + k3z * dt
        if attractor_type == 0:
            k4x, k4y, k4z = lorenz(x4, y4, z4, 10.0, 28.0, 8.0 / 3.0)
        elif attractor_type == 1:
            k4x, k4y, k4z = thomas(x4, y4, z4, 0.208186)
        else:
            k4x, k4y, k4z = aizawa(x4, y4, z4)

        x = x + (k1x + 2.0 * k2x + 2.0 * k3x + k4x) * dt / 6.0
        y = y + (k1y + 2.0 * k2y + 2.0 * k3y + k4y) * dt / 6.0
        z = z + (k1z + 2.0 * k2z + 2.0 * k3z + k4z) * dt / 6.0

        # Map position to grid coordinates and accumulate
        gx = int((x - vol_min_x) / extent_x * float(nx))
        gy = int((y - vol_min_y) / extent_y * float(ny))
        gz = int((z - vol_min_z) / extent_z * float(nz))

        if 0 <= gx < nx and 0 <= gy < ny and 0 <= gz < nz:
            wp.atomic_add(volume, gx, gy, gz, 1.0)

    # Store final position for next iteration
    pos_x[tid] = x
    pos_y[tid] = y
    pos_z[tid] = z


# ---- Volume rendering (reused from example_volume_render.py) ----


@wp.func
def sample_volume(
    volume: wp.array3d[wp.float32],
    px: float,
    py: float,
    pz: float,
    vol_min_x: float,
    vol_min_y: float,
    vol_min_z: float,
    extent_x: float,
    extent_y: float,
    extent_z: float,
):
    nx = volume.shape[0]
    ny = volume.shape[1]
    nz = volume.shape[2]

    gx = wp.clamp((px - vol_min_x) / extent_x * float(nx - 1), 0.0, float(nx - 1))
    gy = wp.clamp((py - vol_min_y) / extent_y * float(ny - 1), 0.0, float(ny - 1))
    gz = wp.clamp((pz - vol_min_z) / extent_z * float(nz - 1), 0.0, float(nz - 1))

    x0 = int(wp.floor(gx))
    y0 = int(wp.floor(gy))
    z0 = int(wp.floor(gz))
    x1 = wp.min(x0 + 1, nx - 1)
    y1 = wp.min(y0 + 1, ny - 1)
    z1 = wp.min(z0 + 1, nz - 1)

    fx = gx - float(x0)
    fy = gy - float(y0)
    fz = gz - float(z0)

    c00 = volume[x0, y0, z0] * (1.0 - fx) + volume[x1, y0, z0] * fx
    c01 = volume[x0, y0, z1] * (1.0 - fx) + volume[x1, y0, z1] * fx
    c10 = volume[x0, y1, z0] * (1.0 - fx) + volume[x1, y1, z0] * fx
    c11 = volume[x0, y1, z1] * (1.0 - fx) + volume[x1, y1, z1] * fx

    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy

    return c0 * (1.0 - fz) + c1 * fz


@wp.kernel
def render_volume(
    volume: wp.array3d[wp.float32],
    pixels: wp.array[wp.vec3],
    cam_pos: wp.vec3,
    cam_fwd: wp.vec3,
    cam_right: wp.vec3,
    cam_up: wp.vec3,
    fov: float,
    width: int,
    height: int,
    vol_min_x: float,
    vol_min_y: float,
    vol_min_z: float,
    vol_max_x: float,
    vol_max_y: float,
    vol_max_z: float,
    step_size: float,
    brightness: float,
):
    tid = wp.tid()

    px = tid % width
    py = tid / width

    aspect = float(width) / float(height)
    half_fov = wp.tan(fov * 0.5)

    u = (2.0 * (float(px) + 0.5) / float(width) - 1.0) * half_fov * aspect
    v = (2.0 * (float(py) + 0.5) / float(height) - 1.0) * half_fov

    ray_dir = wp.normalize(cam_fwd + cam_right * u + cam_up * v)

    # Ray-AABB intersection
    inv_x = 1.0 / (ray_dir[0] + 1.0e-8)
    inv_y = 1.0 / (ray_dir[1] + 1.0e-8)
    inv_z = 1.0 / (ray_dir[2] + 1.0e-8)

    t1x = (vol_min_x - cam_pos[0]) * inv_x
    t2x = (vol_max_x - cam_pos[0]) * inv_x
    t1y = (vol_min_y - cam_pos[1]) * inv_y
    t2y = (vol_max_y - cam_pos[1]) * inv_y
    t1z = (vol_min_z - cam_pos[2]) * inv_z
    t2z = (vol_max_z - cam_pos[2]) * inv_z

    t_near = wp.max(wp.min(t1x, t2x), wp.max(wp.min(t1y, t2y), wp.min(t1z, t2z)))
    t_far = wp.min(wp.max(t1x, t2x), wp.min(wp.max(t1y, t2y), wp.max(t1z, t2z)))

    # Background
    nv = float(py) / float(height)
    bg = wp.vec3(0.02 + nv * 0.03, 0.02 + nv * 0.03, 0.04 + nv * 0.06)

    if t_near >= t_far or t_far < 0.0:
        pixels[tid] = bg
        return

    t_near = wp.max(t_near, 0.0)
    extent_x = vol_max_x - vol_min_x
    extent_y = vol_max_y - vol_min_y
    extent_z = vol_max_z - vol_min_z

    color = wp.vec3(0.0, 0.0, 0.0)
    alpha = float(0.0)
    t = t_near

    while t < t_far and alpha < 0.98:
        pos = cam_pos + ray_dir * t
        d = sample_volume(
            volume, pos[0], pos[1], pos[2],
            vol_min_x, vol_min_y, vol_min_z,
            extent_x, extent_y, extent_z,
        ) * brightness

        if d > 0.01:
            # Emission-absorption: warm color palette
            a = 1.0 - wp.exp(-d * step_size * 5.0)
            r = 0.1 + d * 2.5
            g = 0.05 + d * 0.8
            b = 0.3 + d * 1.2
            r = wp.min(r, 1.0)
            g = wp.min(g, 1.0)
            b = wp.min(b, 1.0)

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


# ---- Attractor configurations ----

ATTRACTORS = {
    "lorenz": {
        "type": 0,
        "dt": 0.005,
        "bounds": ((-25, -30, 0), (25, 30, 55)),
        "cam_pos": (50, 20, 30),
        "init_range": ((-0.1, -0.1, 20), (0.1, 0.1, 25)),
    },
    "thomas": {
        "type": 1,
        "dt": 0.05,
        "bounds": ((-5, -5, -5), (5, 5, 5)),
        "cam_pos": (10, 5, 10),
        "init_range": ((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)),
    },
    "aizawa": {
        "type": 2,
        "dt": 0.01,
        "bounds": ((-2, -2, -1), (2, 2, 2.5)),
        "cam_pos": (4, 2, 4),
        "init_range": ((-0.01, -0.01, 0.5), (0.01, 0.01, 1.0)),
    },
}


class Example:
    def __init__(self, width=1024, height=1024, grid_size=256, num_particles=500000, attractor="lorenz"):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.num_particles = num_particles

        config = ATTRACTORS[attractor]
        self.attractor_type = config["type"]
        self.dt = config["dt"]

        # Volume bounds
        bmin, bmax = config["bounds"]
        self.vol_min = bmin
        self.vol_max = bmax

        n = grid_size
        self.volume = wp.zeros((n, n, n), dtype=wp.float32)

        # Initialize particles
        rng = np.random.default_rng(42)
        imin, imax = config["init_range"]
        self.pos_x = wp.array(rng.uniform(imin[0], imax[0], num_particles).astype(np.float32))
        self.pos_y = wp.array(rng.uniform(imin[1], imax[1], num_particles).astype(np.float32))
        self.pos_z = wp.array(rng.uniform(imin[2], imax[2], num_particles).astype(np.float32))

        # Camera
        self.cam_pos = wp.vec3(*config["cam_pos"])
        center = [(bmin[i] + bmax[i]) / 2 for i in range(3)]
        self.cam_target = wp.vec3(*center)
        self.fov = math.radians(50.0)

        # Rendering
        self.step_size = max(bmax[i] - bmin[i] for i in range(3)) / float(n)
        self.pixels = wp.zeros(width * height, dtype=wp.vec3)

    def trace(self, steps=200):
        """Trace particles through the attractor, accumulating into the volume."""
        with wp.ScopedTimer("trace", active=False):
            wp.launch(
                kernel=trace_particles,
                dim=self.num_particles,
                inputs=[
                    self.pos_x,
                    self.pos_y,
                    self.pos_z,
                    self.volume,
                    self.attractor_type,
                    self.dt,
                    steps,
                    self.vol_min[0], self.vol_min[1], self.vol_min[2],
                    self.vol_max[0], self.vol_max[1], self.vol_max[2],
                ],
            )

    def render(self):
        """Volume render the accumulated density field."""
        with wp.ScopedTimer("render"):
            cam_fwd = wp.normalize(wp.vec3(
                self.cam_target[0] - self.cam_pos[0],
                self.cam_target[1] - self.cam_pos[1],
                self.cam_target[2] - self.cam_pos[2],
            ))
            world_up = wp.vec3(0.0, 1.0, 0.0)
            cam_right = wp.normalize(wp.cross(cam_fwd, world_up))
            cam_up = wp.cross(cam_right, cam_fwd)

            # Normalize volume for rendering
            v_max = float(self.volume.numpy().max())
            brightness = 1.0 / max(v_max, 1.0)

            wp.launch(
                kernel=render_volume,
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
                    self.vol_min[0], self.vol_min[1], self.vol_min[2],
                    self.vol_max[0], self.vol_max[1], self.vol_max[2],
                    self.step_size,
                    brightness,
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
    parser.add_argument("--grid-size", type=int, default=256, help="Volume grid resolution per axis.")
    parser.add_argument("--num-particles", type=int, default=500000, help="Number of tracer particles.")
    parser.add_argument(
        "--attractor",
        type=str,
        default="lorenz",
        choices=["lorenz", "thomas", "aizawa"],
        help="Attractor type.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            width=args.width,
            height=args.height,
            grid_size=args.grid_size,
            num_particles=args.num_particles,
            attractor=args.attractor,
        )
        example.trace(steps=500)
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
