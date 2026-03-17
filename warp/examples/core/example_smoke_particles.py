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
# Example Smoke Particles
#
# Implements a simple Lagrangian smoke particle system where particles:
#   - Rise due to buoyancy
#   - Spread due to turbulent diffusion (random noise)
#   - Fade over time and are recycled at the source
#
# This demonstrates a particle-based approach to volumetric effects
# that is simpler than grid-based Eulerian methods.
#
# Demonstrates:
#   - Particle system management with wp.arrays
#   - Random number generation in kernels (wp.rand_init, wp.randf)
#   - Particle recycling for continuous emission
#   - Color interpolation based on particle age
#
###########################################################################

import numpy as np

import warp as wp
import warp.render


@wp.kernel
def update_particles(
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    life: wp.array[wp.float32],
    max_life: float,
    source_pos: wp.vec3,
    source_radius: float,
    buoyancy: float,
    turbulence: float,
    dt: float,
    seed: int,
):
    tid = wp.tid()

    p = pos[tid]
    v = vel[tid]
    t = life[tid]

    # Update life
    t = t + dt

    # Respawn dead particles at source
    if t >= max_life:
        t = 0.0
        # Random position in sphere around source
        state = wp.rand_init(seed, tid)
        rx = wp.randf(state) * 2.0 - 1.0
        ry = wp.randf(state) * 2.0 - 1.0
        rz = wp.randf(state) * 2.0 - 1.0
        p = source_pos + wp.vec3(rx, ry, rz) * source_radius
        v = wp.vec3(0.0, buoyancy * 0.5, 0.0)  # Initial upward velocity

    # Apply forces
    # Buoyancy (upward force that decreases with height)
    height_factor = wp.max(0.0, 1.0 - (p[1] - source_pos[1]) / 40.0)
    v = wp.vec3(v[0], v[1] + buoyancy * height_factor * dt, v[2])

    # Turbulent diffusion (random walk)
    state = wp.rand_init(seed + 1, tid + int(t * 1000.0))
    noise_x = (wp.randf(state) * 2.0 - 1.0) * turbulence
    noise_y = (wp.randf(state) * 2.0 - 1.0) * turbulence * 0.3
    noise_z = (wp.randf(state) * 2.0 - 1.0) * turbulence
    v = v + wp.vec3(noise_x, noise_y, noise_z) * dt

    # Damping
    v = v * 0.98

    # Integrate
    p = p + v * dt

    pos[tid] = p
    vel[tid] = v
    life[tid] = t


@wp.kernel
def compute_colors(
    life: wp.array[wp.float32],
    colors: wp.array[wp.vec3],
    max_life: float,
):
    tid = wp.tid()

    t = life[tid]
    age = t / max_life  # 0 to 1

    # Color gradient: white -> yellow -> orange -> dark (fading out)
    if age < 0.3:
        # White to light gray
        c = 0.95 - age * 0.5
        colors[tid] = wp.vec3(c, c, c)
    elif age < 0.6:
        # Gray fading
        c = 0.8 - (age - 0.3) * 0.8
        colors[tid] = wp.vec3(c, c * 0.95, c * 0.9)
    else:
        # Dark gray fading to invisible
        c = 0.56 - (age - 0.6) * 1.2
        c = wp.max(c, 0.0)
        colors[tid] = wp.vec3(c, c, c)


class Example:
    def __init__(self, stage_path="example_smoke_particles.usd", num_particles=50000):
        self.num_particles = num_particles
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.seed = 0

        # Simulation parameters
        self.max_life = 4.0
        self.source_pos = wp.vec3(0.0, 0.0, 0.0)
        self.source_radius = 3.0
        self.buoyancy = 12.0
        self.turbulence = 20.0

        n = num_particles
        rng = np.random.default_rng(42)

        # Initialize particles with random ages for staggered emission
        positions = rng.uniform(-self.source_radius, self.source_radius, (n, 3)).astype(np.float32)
        positions[:, 1] = rng.uniform(0, 20, n).astype(np.float32)  # Spread vertically
        velocities = np.zeros((n, 3), dtype=np.float32)
        velocities[:, 1] = rng.uniform(1, 3, n).astype(np.float32)
        lifetimes = rng.uniform(0, self.max_life, n).astype(np.float32)

        self.pos = wp.array(positions, dtype=wp.vec3)
        self.vel = wp.array(velocities, dtype=wp.vec3)
        self.life = wp.array(lifetimes, dtype=wp.float32)
        self.colors = wp.zeros(n, dtype=wp.vec3)

        if stage_path and stage_path.endswith((".usd", ".usda", ".usdc")):
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = wp.render.NativeRenderer(512, 512)
            self.renderer.setup_camera(pos=(15, 15, 30), target=(0, 8, 0), fov=50)
            self.renderer.bg_top = wp.vec3(0.08, 0.08, 0.12)
            self.renderer.bg_bottom = wp.vec3(0.02, 0.02, 0.04)

    def step(self):
        with wp.ScopedTimer("step", active=False):
            self.seed += 1

            wp.launch(
                kernel=update_particles,
                dim=self.num_particles,
                inputs=[
                    self.pos,
                    self.vel,
                    self.life,
                    self.max_life,
                    self.source_pos,
                    self.source_radius,
                    self.buoyancy,
                    self.turbulence,
                    self.frame_dt,
                    self.seed,
                ],
            )

            wp.launch(
                kernel=compute_colors,
                dim=self.num_particles,
                inputs=[self.life, self.colors, self.max_life],
            )

            self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_points(
                points=self.pos,
                radius=0.15,
                name="smoke",
                colors=self.colors.numpy(),
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
    parser.add_argument("--num-particles", type=int, default=50000, help="Number of particles.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_particles=args.num_particles)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            if hasattr(example.renderer, 'save'):
                example.renderer.save()
            if hasattr(example.renderer, 'save_image'):
                example.renderer.save_image("example_smoke_particles.png")
                print("Saved example_smoke_particles.png")
