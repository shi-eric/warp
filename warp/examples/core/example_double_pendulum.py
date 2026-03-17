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
# Example Double Pendulum Fractal
#
# Simulates millions of double pendulums with slightly different initial
# conditions in parallel. The final state (which half of the plane the
# second bob ends up in) is mapped to color, revealing intricate fractal
# boundaries that separate chaotic from regular behavior.
#
# Each pixel corresponds to a unique (θ₁, θ₂) starting configuration.
# The Hamiltonian equations of motion are integrated with a symplectic
# (leapfrog) integrator.
#
# Demonstrates:
#   - Embarrassingly parallel simulation (one ODE per pixel)
#   - Symplectic integration for Hamiltonian systems
#   - Chaos and sensitivity to initial conditions
#   - Fractal basin boundary visualization
#
###########################################################################

import math

import numpy as np

import warp as wp


@wp.kernel
def simulate_pendulums(
    pixels: wp.array[wp.vec3],
    width: int,
    height: int,
    theta1_min: float,
    theta1_max: float,
    theta2_min: float,
    theta2_max: float,
    sim_time: float,
    dt: float,
    g: float,
    l1: float,
    l2: float,
    m1: float,
    m2: float,
):
    tid = wp.tid()

    px = tid % width
    py = tid / width

    # Map pixel to initial angles
    t1 = theta1_min + (float(px) + 0.5) / float(width) * (theta1_max - theta1_min)
    t2 = theta2_min + (float(py) + 0.5) / float(height) * (theta2_max - theta2_min)

    # Initial angular velocities = 0
    w1 = float(0.0)
    w2 = float(0.0)

    num_steps = int(sim_time / dt)

    # Symplectic Euler integration
    for _ in range(num_steps):
        # Equations of motion for double pendulum
        cos_delta = wp.cos(t1 - t2)
        sin_delta = wp.sin(t1 - t2)

        denom = m1 + m2 * (1.0 - cos_delta * cos_delta)
        if wp.abs(denom) < 1.0e-8:
            denom = 1.0e-8

        # Angular accelerations
        a1 = (
            -g * (m1 + m2) * wp.sin(t1)
            - m2 * g * wp.sin(t1 - 2.0 * t2)
            - 2.0 * m2 * sin_delta * (w2 * w2 * l2 + w1 * w1 * l1 * cos_delta)
        ) / (l1 * (2.0 * denom))

        a2 = (
            2.0
            * sin_delta
            * (
                w1 * w1 * l1 * (m1 + m2)
                + g * (m1 + m2) * wp.cos(t1)
                + w2 * w2 * l2 * m2 * cos_delta
            )
        ) / (l2 * (2.0 * denom))

        # Leapfrog
        w1 = w1 + a1 * dt
        w2 = w2 + a2 * dt
        t1 = t1 + w1 * dt
        t2 = t2 + w2 * dt

    # Compute final position of second bob
    x2 = l1 * wp.sin(t1) + l2 * wp.sin(t2)
    y2 = -l1 * wp.cos(t1) - l2 * wp.cos(t2)

    # Color by final angle of second pendulum (mapped to hue)
    angle = wp.atan2(y2, x2) / 3.14159265  # [-1, 1]
    energy = 0.5 * (w1 * w1 + w2 * w2)
    brightness = wp.clamp(1.0 / (1.0 + energy * 0.01), 0.3, 1.0)

    # Map to RGB using a three-zone coloring
    r = float(0.0)
    g_c = float(0.0)
    b = float(0.0)

    if angle < -0.33:
        t = (angle + 1.0) / 0.67
        r = 0.8 * (1.0 - t) + 0.2 * t
        g_c = 0.2 * (1.0 - t) + 0.7 * t
        b = 0.3
    elif angle < 0.33:
        t = (angle + 0.33) / 0.66
        r = 0.2
        g_c = 0.7 * (1.0 - t) + 0.3 * t
        b = 0.3 * (1.0 - t) + 0.9 * t
    else:
        t = (angle - 0.33) / 0.67
        r = 0.2 * (1.0 - t) + 0.9 * t
        g_c = 0.3
        b = 0.9 * (1.0 - t) + 0.2 * t

    r = r * brightness
    g_c = g_c * brightness
    b = b * brightness

    pixels[tid] = wp.vec3(
        wp.pow(r, 0.4545),
        wp.pow(g_c, 0.4545),
        wp.pow(b, 0.4545),
    )


class Example:
    def __init__(self, width=1024, height=1024):
        self.width = width
        self.height = height

        # Physics parameters
        self.g = 9.81
        self.l1 = 1.0
        self.l2 = 1.0
        self.m1 = 1.0
        self.m2 = 1.0

        # Simulation
        self.sim_time = 20.0  # Total time per pixel
        self.dt = 0.005

        # View range (initial angles)
        self.theta1_range = (-math.pi, math.pi)
        self.theta2_range = (-math.pi, math.pi)

        self.pixels = wp.zeros(width * height, dtype=wp.vec3)

    def render(self):
        with wp.ScopedTimer("render"):
            wp.launch(
                kernel=simulate_pendulums,
                dim=self.width * self.height,
                inputs=[
                    self.pixels,
                    self.width,
                    self.height,
                    self.theta1_range[0],
                    self.theta1_range[1],
                    self.theta2_range[0],
                    self.theta2_range[1],
                    self.sim_time,
                    self.dt,
                    self.g,
                    self.l1,
                    self.l2,
                    self.m1,
                    self.m2,
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
        example.render()

        if not args.headless:
            import matplotlib.pyplot as plt

            img = example.pixels.numpy().reshape((example.height, example.width, 3))
            plt.imshow(img, origin="lower", interpolation="antialiased")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
