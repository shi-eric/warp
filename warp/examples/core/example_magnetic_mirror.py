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
# Example Magnetic Mirror (Plasma Confinement)
#
# Simulates charged particles trapped in a magnetic mirror — two
# regions of strong magnetic field that reflect particles back and
# forth. Particles gyrate around field lines and bounce between the
# mirror points, demonstrating:
#   - Cyclotron (Larmor) gyration
#   - Magnetic mirroring / adiabatic invariant conservation
#   - Loss cone — particles with too much parallel velocity escape
#
# Uses the Boris integrator, the gold standard for charged-particle
# pushing in electromagnetic fields.
#
# Demonstrates:
#   - Boris particle pusher (exact rotation in B field)
#   - Magnetic mirror field geometry
#   - Particle trapping and loss cone physics
#   - GPU-parallel particle trajectories
#
###########################################################################

import math

import numpy as np

import warp as wp
import warp.render


@wp.func
def mirror_field(pos: wp.vec3, b0: float, mirror_ratio: float, mirror_length: float):
    """Magnetic mirror field: B increases at z = ±mirror_length/2.

    B_z(z) = B0 * (1 + (R-1) * (2z/L)^2) where R is mirror ratio.
    Includes a radial component for ∇·B = 0.
    """
    z_norm = 2.0 * pos[2] / mirror_length
    z2 = z_norm * z_norm

    bz = b0 * (1.0 + (mirror_ratio - 1.0) * z2)

    # Radial component from ∇·B = 0: Br = -r/2 * dBz/dz
    dbz_dz = b0 * (mirror_ratio - 1.0) * 4.0 * z_norm / mirror_length
    bx = -pos[0] * 0.5 * dbz_dz
    by = -pos[1] * 0.5 * dbz_dz

    return wp.vec3(bx, by, bz)


@wp.kernel
def boris_push(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    qm: float,
    b0: float,
    mirror_ratio: float,
    mirror_length: float,
    dt: float,
):
    """Boris integrator for charged particles in a magnetic field."""
    tid = wp.tid()

    p = positions[tid]
    v = velocities[tid]

    # Get B field at particle position
    b = mirror_field(p, b0, mirror_ratio, mirror_length)

    # Boris rotation
    t = b * (qm * dt * 0.5)
    t_mag2 = wp.dot(t, t)
    s = t * (2.0 / (1.0 + t_mag2))

    # v- = v + qE*dt/2 (no E field here)
    v_minus = v

    # v' = v- + v- × t
    v_prime = v_minus + wp.cross(v_minus, t)

    # v+ = v- + v' × s
    v_plus = v_minus + wp.cross(v_prime, s)

    # v_new = v+ + qE*dt/2
    v = v_plus

    # Update position
    p = p + v * dt

    # Absorbing boundary at mirror ends (particles that escape are marked)
    # They continue to propagate freely but are outside the confinement region
    positions[tid] = p
    velocities[tid] = v


class Example:
    def __init__(self, stage_path="example_magnetic_mirror.usd", num_particles=100000):
        self.num_particles = num_particles
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.substeps = 20

        # Plasma parameters (normalized units)
        self.qm = 10.0  # charge/mass ratio (determines gyrofrequency)
        self.b0 = 1.0  # Central magnetic field strength
        self.mirror_ratio = 5.0  # B_max/B_min
        self.mirror_length = 20.0  # Distance between mirrors
        self.dt = 0.005

        rng = np.random.default_rng(42)

        # Initialize particles near center with thermal distribution
        positions = np.zeros((num_particles, 3), dtype=np.float32)
        positions[:, 0] = rng.normal(0, 0.5, num_particles)  # Small radial spread
        positions[:, 1] = rng.normal(0, 0.5, num_particles)
        positions[:, 2] = rng.normal(0, 3.0, num_particles)  # Spread along field line

        # Thermal velocities — isotropic distribution
        v_thermal = 3.0
        velocities = np.zeros((num_particles, 3), dtype=np.float32)
        velocities[:, 0] = rng.normal(0, v_thermal, num_particles)
        velocities[:, 1] = rng.normal(0, v_thermal, num_particles)
        velocities[:, 2] = rng.normal(0, v_thermal, num_particles)  # Isotropic

        self.positions = wp.array(positions, dtype=wp.vec3)
        self.velocities = wp.array(velocities, dtype=wp.vec3)

        if stage_path and stage_path.endswith((".usd", ".usda", ".usdc")):
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = wp.render.NativeRenderer(512, 512)
            self.renderer.setup_camera(pos=(8, 6, 20), target=(0, 0, 0), fov=45)
            self.renderer.bg_top = wp.vec3(0.03, 0.04, 0.10)
            self.renderer.bg_bottom = wp.vec3(0.01, 0.01, 0.04)
            self.renderer.shadows = False

    def step(self):
        with wp.ScopedTimer("step", active=False):
            for _ in range(self.substeps):
                wp.launch(
                    kernel=boris_push,
                    dim=self.num_particles,
                    inputs=[
                        self.positions, self.velocities,
                        self.qm, self.b0, self.mirror_ratio,
                        self.mirror_length, self.dt,
                    ],
                )

            self.sim_time += self.frame_dt

    def compute_magnetic_moments(self):
        """Compute the magnetic moment μ = mv⊥²/(2B) for each particle.

        The magnetic moment is an adiabatic invariant — it should be
        conserved as particles move through the varying B field.
        Deviation of μ from its initial value measures the quality of
        the Boris integrator. For a good integrator, μ should be
        conserved to within a few percent.
        """
        pos = self.positions.numpy()
        vel = self.velocities.numpy()

        # Compute B at each particle position
        z_norm = 2.0 * pos[:, 2] / self.mirror_length
        B_z = self.b0 * (1.0 + (self.mirror_ratio - 1.0) * z_norm**2)
        # Approximate |B| ≈ B_z for particles near axis
        B_mag = B_z

        # v_perp² = vx² + vy² (perpendicular to z-axis / B-field)
        v_perp_sq = vel[:, 0]**2 + vel[:, 1]**2

        # μ = m * v_perp² / (2B), with m=1
        mu = v_perp_sq / (2.0 * np.abs(B_mag) + 1e-10)

        return mu

    def compute_loss_cone_fraction(self):
        """Compute the fraction of particles that have escaped the mirror.

        A particle escapes if |z| > mirror_length/2 (passed through a
        mirror point). For an isotropic velocity distribution, the
        theoretical trapped fraction is:

            f_trapped = √(1 - 1/R)

        So the escape fraction is 1 - √(1 - 1/R).
        With mirror_ratio R=5: f_escape = 1 - √(0.8) ≈ 10.6%.
        """
        pos = self.positions.numpy()
        half_l = self.mirror_length * 0.5

        escaped = np.abs(pos[:, 2]) > half_l
        escape_fraction = np.mean(escaped)

        theoretical_escape = 1.0 - np.sqrt(1.0 - 1.0 / self.mirror_ratio)

        return escape_fraction, theoretical_escape

    def compute_energy_conservation(self):
        """Compute total kinetic energy (should be conserved in static B field).

        Since B does no work (F ⊥ v), the Boris integrator should
        conserve kinetic energy exactly. Any drift indicates numerical error.
        """
        vel = self.velocities.numpy()
        return 0.5 * np.sum(vel**2)

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_points(
                points=self.positions,
                radius=0.03,
                name="plasma",
                colors=(0.4, 0.7, 1.0),
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
    parser.add_argument("--num-frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument("--num-particles", type=int, default=100000, help="Number of plasma particles.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_particles=args.num_particles)

        for i in range(args.num_frames):
            example.step()
            example.render()

            if i % 50 == 0:
                pos = example.positions.numpy()
                print(f"Frame {i}: z_range=[{pos[:,2].min():.2f}, {pos[:,2].max():.2f}]")

        if example.renderer:
            if hasattr(example.renderer, 'save'):
                example.renderer.save()
            if hasattr(example.renderer, 'save_image'):
                example.renderer.save_image("example_magnetic_mirror.png")
                print("Saved example_magnetic_mirror.png")
