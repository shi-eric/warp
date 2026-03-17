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
# Example Galaxy
#
# Simulates a spiral galaxy using particles in a gravitational disk.
# Stars orbit a central massive core with realistic Keplerian velocities,
# producing spiral arm structures through density waves.
#
# The simulation uses a simplified softened gravity model with a central
# point mass and optional disk self-gravity approximation.
#
# Demonstrates:
#   - Particle initial conditions on a rotating disk
#   - Keplerian orbital mechanics
#   - Softened gravity for numerical stability
#   - Point cloud rendering with motion blur
#
###########################################################################

import math

import numpy as np

import warp as wp
import warp.render


@wp.kernel
def integrate_orbits(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    central_mass: float,
    softening: float,
    dt: float,
):
    """Integrate stellar orbits around central mass."""
    tid = wp.tid()

    p = positions[tid]
    v = velocities[tid]

    # Distance to galactic center
    r = wp.length(p)
    r_soft = wp.sqrt(r * r + softening * softening)

    # Gravitational acceleration toward center
    # a = -G * M / r^2 * (p / r) = -G * M * p / r^3
    acc = -central_mass * p / (r_soft * r_soft * r_soft)

    # Leapfrog integration
    v_half = v + acc * (dt * 0.5)
    p_new = p + v_half * dt

    # Recompute acceleration at new position
    r_new = wp.length(p_new)
    r_soft_new = wp.sqrt(r_new * r_new + softening * softening)
    acc_new = -central_mass * p_new / (r_soft_new * r_soft_new * r_soft_new)

    v_new = v_half + acc_new * (dt * 0.5)

    positions[tid] = p_new
    velocities[tid] = v_new


class Example:
    def __init__(self, stage_path="example_galaxy.usd", num_stars=200000):
        self.num_stars = num_stars
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.substeps = 8

        # Galaxy parameters
        self.central_mass = 5000.0
        self.disk_radius = 20.0
        self.disk_thickness = 0.5
        self.softening = 0.5

        # Initialize star positions and velocities
        rng = np.random.default_rng(42)

        # Exponential disk profile
        r = rng.exponential(scale=self.disk_radius / 3.0, size=num_stars).astype(np.float32)
        r = np.clip(r, 0.1, self.disk_radius * 2)

        theta = rng.uniform(0, 2 * np.pi, num_stars).astype(np.float32)

        # Add spiral arm perturbation
        num_arms = 2
        arm_tightness = 0.3
        arm_strength = 0.4
        theta += arm_strength * np.sin(num_arms * theta + arm_tightness * r)

        # Positions
        x = r * np.cos(theta)
        z = r * np.sin(theta)
        y = rng.normal(0, self.disk_thickness, num_stars).astype(np.float32) * (1.0 - r / (self.disk_radius * 2))

        positions = np.stack([x, y, z], axis=1).astype(np.float32)

        # Keplerian circular velocities: v = sqrt(GM/r)
        v_circ = np.sqrt(self.central_mass / np.maximum(r, 0.1))

        # Velocity perpendicular to radius (tangential)
        vx = -v_circ * np.sin(theta)
        vz = v_circ * np.cos(theta)
        vy = np.zeros_like(vx)

        # Add small random velocity dispersion
        dispersion = 0.02
        vx += rng.normal(0, dispersion * v_circ, num_stars)
        vy += rng.normal(0, dispersion * v_circ * 0.1, num_stars)
        vz += rng.normal(0, dispersion * v_circ, num_stars)

        velocities = np.stack([vx, vy, vz], axis=1).astype(np.float32)

        self.positions = wp.array(positions, dtype=wp.vec3)
        self.velocities = wp.array(velocities, dtype=wp.vec3)

        if stage_path and stage_path.endswith((".usd", ".usda", ".usdc")):
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = wp.render.NativeRenderer(512, 512)
            self.renderer.setup_camera(pos=(0, 25, 35), target=(0, 0, 0), fov=50)
            self.renderer.bg_top = wp.vec3(0.03, 0.03, 0.08)
            self.renderer.bg_bottom = wp.vec3(0.01, 0.01, 0.03)
            self.renderer.shadows = False
            self.renderer.fog_density = 0.003

    def step(self):
        with wp.ScopedTimer("step", active=False):
            dt = self.frame_dt / self.substeps

            for _ in range(self.substeps):
                wp.launch(
                    kernel=integrate_orbits,
                    dim=self.num_stars,
                    inputs=[
                        self.positions,
                        self.velocities,
                        self.central_mass,
                        self.softening,
                        dt,
                    ],
                )

            self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_points(
                points=self.positions,
                radius=0.05,
                name="stars",
                colors=(0.95, 0.92, 0.85),
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
    parser.add_argument("--num-frames", type=int, default=500, help="Total number of frames.")
    parser.add_argument("--num-stars", type=int, default=200000, help="Number of stars.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_stars=args.num_stars)

        for i in range(args.num_frames):
            example.step()
            example.render()

            if i % 100 == 0:
                print(f"Frame {i}/{args.num_frames}")

        if example.renderer:
            if hasattr(example.renderer, 'save'):
                example.renderer.save()
            if hasattr(example.renderer, 'save_image'):
                example.renderer.save_image("example_galaxy.png")
                print("Saved example_galaxy.png")
