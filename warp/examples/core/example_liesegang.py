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
# Example Liesegang Rings
#
# Simulates the formation of Liesegang rings — periodic precipitation
# bands that form when two reactants diffuse into each other in a gel.
# This classic chemistry phenomenon produces strikingly regular
# concentric shells in 3D.
#
# The model uses three coupled reaction-diffusion fields:
#   A (outer electrolyte, diffusing inward)
#   B (inner electrolyte, uniformly distributed)
#   C (precipitate, forms when A*B exceeds a supersaturation threshold)
#
# Demonstrates:
#   - Multi-species reaction-diffusion system
#   - Nucleation threshold (supersaturation criterion)
#   - Precipitation and depletion dynamics
#   - 3D concentric shell pattern formation
#   - Marching cubes visualization of precipitate field
#
###########################################################################

import numpy as np

import warp as wp
import warp.render


@wp.kernel
def reaction_diffusion_step(
    a_in: wp.array3d[wp.float32],
    b_in: wp.array3d[wp.float32],
    c_in: wp.array3d[wp.float32],
    a_out: wp.array3d[wp.float32],
    b_out: wp.array3d[wp.float32],
    c_out: wp.array3d[wp.float32],
    da: float,
    db: float,
    threshold: float,
    precip_rate: float,
    dt: float,
):
    """One step of the Liesegang reaction-diffusion system."""
    i, j, k = wp.tid()

    nx = a_in.shape[0]
    ny = a_in.shape[1]
    nz = a_in.shape[2]

    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
        a_out[i, j, k] = a_in[i, j, k]
        b_out[i, j, k] = b_in[i, j, k]
        c_out[i, j, k] = c_in[i, j, k]
        return

    a = a_in[i, j, k]
    b = b_in[i, j, k]
    c = c_in[i, j, k]

    # Laplacians
    lap_a = (
        a_in[i + 1, j, k] + a_in[i - 1, j, k]
        + a_in[i, j + 1, k] + a_in[i, j - 1, k]
        + a_in[i, j, k + 1] + a_in[i, j, k - 1]
        - 6.0 * a
    )

    lap_b = (
        b_in[i + 1, j, k] + b_in[i - 1, j, k]
        + b_in[i, j + 1, k] + b_in[i, j - 1, k]
        + b_in[i, j, k + 1] + b_in[i, j, k - 1]
        - 6.0 * b
    )

    # Precipitation reaction: A + B → C when A*B > threshold
    product = a * b
    reaction = float(0.0)
    if product > threshold:
        reaction = precip_rate * (product - threshold)

    # Also, existing precipitate catalyzes further precipitation
    if c > 0.01 and product > threshold * 0.5:
        reaction = reaction + precip_rate * 0.5 * c * product

    # Cap reaction to available reactants
    reaction = wp.min(reaction, wp.min(a, b) / dt)

    a_out[i, j, k] = a + (da * lap_a - reaction) * dt
    b_out[i, j, k] = b + (db * lap_b - reaction) * dt
    c_out[i, j, k] = c + reaction * dt


@wp.kernel
def set_boundary_source(
    a: wp.array3d[wp.float32],
    source_value: float,
    radius: int,
):
    """Maintain A concentration at domain center (source)."""
    i, j, k = wp.tid()

    nx = a.shape[0]
    ny = a.shape[1]
    nz = a.shape[2]

    cx = float(nx) / 2.0
    cy = float(ny) / 2.0
    cz = float(nz) / 2.0

    di = float(i) - cx
    dj = float(j) - cy
    dk = float(k) - cz
    dist_sq = di * di + dj * dj + dk * dk

    if dist_sq < float(radius * radius):
        a[i, j, k] = source_value


class Example:
    def __init__(self, stage_path="example_liesegang.usd", grid_size=128):
        self.grid_size = grid_size
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 30.0
        self.substeps = 50

        n = grid_size

        # Reaction-diffusion parameters
        self.da = 0.5    # Diffusion of A (outer electrolyte) — fast
        self.db = 0.01   # Diffusion of B (inner electrolyte) — slow/immobile
        self.threshold = 0.15  # Supersaturation threshold
        self.precip_rate = 2.0
        self.dt = 0.005

        # Initial conditions:
        # A: high concentration at center (source)
        # B: uniform low concentration throughout
        a_init = np.zeros((n, n, n), dtype=np.float32)
        b_init = np.ones((n, n, n), dtype=np.float32) * 0.5

        # A source at center
        cx, cy, cz = n // 2, n // 2, n // 2
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    d = ((i - cx)**2 + (j - cy)**2 + (k - cz)**2) ** 0.5
                    if d < 5:
                        a_init[i, j, k] = 5.0

        self.a = wp.array(a_init, dtype=wp.float32)
        self.b = wp.array(b_init, dtype=wp.float32)
        self.c = wp.zeros((n, n, n), dtype=wp.float32)

        self.a_tmp = wp.zeros((n, n, n), dtype=wp.float32)
        self.b_tmp = wp.zeros((n, n, n), dtype=wp.float32)
        self.c_tmp = wp.zeros((n, n, n), dtype=wp.float32)

        self.source_radius = 4

        # Marching cubes
        self.mc = wp.MarchingCubes(nx=n, ny=n, nz=n)

        if stage_path and stage_path.endswith((".usd", ".usda", ".usdc")):
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = wp.render.NativeRenderer(512, 512)
            self.renderer.setup_camera(pos=(120, 70, 120), target=(48, 48, 48), fov=50)

    def step(self):
        with wp.ScopedTimer("step", active=False):
            n = self.grid_size

            for _ in range(self.substeps):
                # Maintain source
                wp.launch(
                    kernel=set_boundary_source,
                    dim=(n, n, n),
                    inputs=[self.a, 5.0, self.source_radius],
                )

                # Reaction-diffusion step
                wp.launch(
                    kernel=reaction_diffusion_step,
                    dim=(n, n, n),
                    inputs=[
                        self.a, self.b, self.c,
                        self.a_tmp, self.b_tmp, self.c_tmp,
                        self.da, self.db, self.threshold,
                        self.precip_rate, self.dt,
                    ],
                )

                self.a, self.a_tmp = self.a_tmp, self.a
                self.b, self.b_tmp = self.b_tmp, self.b
                self.c, self.c_tmp = self.c_tmp, self.c

            self.sim_time += self.frame_dt

    def measure_ring_positions(self, threshold=0.05):
        """Measure radial positions of Liesegang precipitation rings.

        Scans the precipitate field along a radial line from the center
        and identifies ring positions as local maxima. The spacing ratio
        (Jablczynski law) predicts:

            x_{n+1} / x_n = const = 1 + p

        where p depends on diffusion coefficients and concentrations.
        A constant ratio confirms the classic Liesegang pattern.
        """
        c = self.c.numpy()
        n = self.grid_size
        center = n // 2

        # Radial profile (average over angular directions)
        max_r = center - 2
        radial = np.zeros(max_r)
        counts = np.zeros(max_r)

        for i in range(n):
            for j in range(n):
                r = np.sqrt((i - center)**2 + (j - center)**2)
                ir = int(r)
                if 0 <= ir < max_r:
                    radial[ir] += c[i, j, center]
                    counts[ir] += 1

        radial = radial / np.maximum(counts, 1)

        # Find peaks (local maxima above threshold)
        ring_positions = []
        for i in range(2, len(radial) - 2):
            if (radial[i] > threshold
                    and radial[i] > radial[i - 1]
                    and radial[i] > radial[i + 1]
                    and radial[i] > radial[i - 2]
                    and radial[i] > radial[i + 2]):
                ring_positions.append(float(i))

        # Compute spacing ratios
        ratios = []
        for i in range(1, len(ring_positions)):
            if ring_positions[i - 1] > 0:
                ratios.append(ring_positions[i] / ring_positions[i - 1])

        return ring_positions, ratios, radial

    def compute_mass_conservation(self):
        """Check total mass conservation: A + B + C should be constant.

        Since A + B → C, the total mass (A + B + C) should be conserved
        (modulo boundary source injection). This validates the reaction
        stoichiometry.
        """
        a = self.a.numpy()
        b = self.b.numpy()
        c = self.c.numpy()
        return a.sum(), b.sum(), c.sum(), a.sum() + b.sum() + c.sum()

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            self.mc.surface(self.c, threshold=0.1)

            self.renderer.begin_frame(self.sim_time)
            if self.mc.verts is not None and len(self.mc.verts) > 0:
                self.renderer.render_mesh(
                    points=self.mc.verts,
                    indices=self.mc.indices,
                    name="precipitate",
                    colors=(0.95, 0.85, 0.6),
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
    parser.add_argument("--num-frames", type=int, default=200, help="Total number of frames.")
    parser.add_argument("--grid-size", type=int, default=128, help="Grid resolution per axis.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, grid_size=args.grid_size)

        for i in range(args.num_frames):
            example.step()
            example.render()

            if i % 50 == 0:
                c = example.c.numpy()
                print(f"Frame {i}: precipitate max={c.max():.3f}, total={c.sum():.1f}")

        if example.renderer:
            if hasattr(example.renderer, 'save'):
                example.renderer.save()
            if hasattr(example.renderer, 'save_image'):
                example.renderer.save_image("example_liesegang.png")
                print("Saved example_liesegang.png")
