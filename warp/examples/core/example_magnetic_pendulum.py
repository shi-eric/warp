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
# Example Magnetic Pendulum
#
# Simulates a magnetic pendulum over a plane of colored magnets.
# A charged bob on a spring swings over three magnets at 120° spacing.
# Depending on initial position, the bob settles at one of three
# attractors — the basin boundaries form intricate fractal patterns.
#
# Renders a fractal basin map by running thousands of pendulums in
# parallel and coloring each pixel by which magnet it converges to.
#
# Demonstrates:
#   - Massive parallelism: one simulation per pixel
#   - Magnetic dipole force computation
#   - Damped oscillator integration
#   - Fractal basin of attraction visualization
#
###########################################################################

import math

import numpy as np

import warp as wp


@wp.kernel
def compute_basins(
    pixels: wp.array[wp.vec3],
    width: int,
    height: int,
    magnets_x: wp.array[wp.float32],
    magnets_y: wp.array[wp.float32],
    num_magnets: int,
    magnet_strength: float,
    friction: float,
    spring_k: float,
    pendulum_height: float,
    sim_steps: int,
    dt: float,
    view_min: float,
    view_max: float,
):
    tid = wp.tid()

    px = tid % width
    py = tid / width

    # Map pixel to initial position
    x = view_min + (float(px) + 0.5) / float(width) * (view_max - view_min)
    y = view_min + (float(py) + 0.5) / float(height) * (view_max - view_min)

    vx = float(0.0)
    vy = float(0.0)

    # Simulate pendulum
    for _ in range(sim_steps):
        fx = float(0.0)
        fy = float(0.0)

        # Magnetic forces from each magnet
        for m in range(num_magnets):
            dx = magnets_x[m] - x
            dy = magnets_y[m] - y
            r_sq = dx * dx + dy * dy + pendulum_height * pendulum_height
            r = wp.sqrt(r_sq)
            r5 = r_sq * r_sq * r

            # Magnetic force ~ 1/r^3 (dipole)
            f = magnet_strength / (r5 + 1.0e-6)
            fx = fx + dx * f
            fy = fy + dy * f

        # Spring restoring force toward center
        fx = fx - spring_k * x
        fy = fy - spring_k * y

        # Friction
        fx = fx - friction * vx
        fy = fy - friction * vy

        # Integrate
        vx = vx + fx * dt
        vy = vy + fy * dt
        x = x + vx * dt
        y = y + vy * dt

    # Determine which magnet we ended up near
    min_dist = float(1.0e10)
    closest = int(0)

    for m in range(num_magnets):
        dx = magnets_x[m] - x
        dy = magnets_y[m] - y
        d = dx * dx + dy * dy
        if d < min_dist:
            min_dist = d
            closest = m

    # Color by closest magnet with distance-based brightness
    brightness = 1.0 / (1.0 + min_dist * 5.0)

    r = float(0.0)
    g = float(0.0)
    b = float(0.0)

    if closest == 0:
        r = 0.9 * brightness
        g = 0.2 * brightness
        b = 0.2 * brightness
    elif closest == 1:
        r = 0.2 * brightness
        g = 0.8 * brightness
        b = 0.3 * brightness
    elif closest == 2:
        r = 0.2 * brightness
        g = 0.3 * brightness
        b = 0.9 * brightness
    elif closest == 3:
        r = 0.9 * brightness
        g = 0.8 * brightness
        b = 0.2 * brightness
    else:
        r = 0.7 * brightness
        g = 0.7 * brightness
        b = 0.7 * brightness

    pixels[tid] = wp.vec3(
        wp.pow(r, 0.4545),
        wp.pow(g, 0.4545),
        wp.pow(b, 0.4545),
    )


class Example:
    def __init__(self, width=1024, height=1024, num_magnets=3):
        self.width = width
        self.height = height

        # Magnet positions (equidistant on a circle)
        radius = 1.0
        mx = []
        my = []
        for i in range(num_magnets):
            angle = 2.0 * math.pi * i / num_magnets
            mx.append(radius * math.cos(angle))
            my.append(radius * math.sin(angle))

        self.magnets_x = wp.array(np.array(mx, dtype=np.float32))
        self.magnets_y = wp.array(np.array(my, dtype=np.float32))
        self.num_magnets = num_magnets

        # Physics
        self.magnet_strength = 1.0
        self.friction = 0.2
        self.spring_k = 0.3
        self.pendulum_height = 0.3
        self.sim_steps = 2000
        self.dt = 0.02

        # View range
        self.view_min = -2.5
        self.view_max = 2.5

        self.pixels = wp.zeros(width * height, dtype=wp.vec3)

    def render(self):
        with wp.ScopedTimer("render"):
            wp.launch(
                kernel=compute_basins,
                dim=self.width * self.height,
                inputs=[
                    self.pixels,
                    self.width,
                    self.height,
                    self.magnets_x,
                    self.magnets_y,
                    self.num_magnets,
                    self.magnet_strength,
                    self.friction,
                    self.spring_k,
                    self.pendulum_height,
                    self.sim_steps,
                    self.dt,
                    self.view_min,
                    self.view_max,
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
    parser.add_argument("--num-magnets", type=int, default=3, help="Number of magnets.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(width=args.width, height=args.height, num_magnets=args.num_magnets)
        example.render()

        if not args.headless:
            import matplotlib.pyplot as plt

            img = example.pixels.numpy().reshape((example.height, example.width, 3))
            plt.imshow(img, origin="lower", interpolation="antialiased")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
