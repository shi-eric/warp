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
# Example Cell Morphogenesis (Vertex Model)
#
# Simulates cell morphogenesis: soft spheres grow, divide, and push
# neighbors in a 3D domain with a diffusing nutrient field. Cells
# consume nutrient to accumulate growth material; when large enough
# they divide along a random axis.
#
# Physics:
#   - Cell-cell repulsion (soft sphere overlap penalty)
#   - Cell-cell adhesion (spring-like attraction at intermediate range)
#   - Nutrient diffusion + consumption on a 3D grid
#   - Growth and division mechanics
#
# Demonstrates:
#   - Coupled particle-grid simulation
#   - Dynamic particle count (cell division)
#   - HashGrid neighbor queries for O(N) interactions
#   - Exponential-to-plateau growth dynamics
#
# Validation:
#   - Cell count growth curve (exponential → plateau)
#   - Spatial density uniformity
#   - Pair correlation function g(r)
#
###########################################################################

import numpy as np

import warp as wp
import warp.render


@wp.kernel
def compute_cell_forces(
    grid: wp.uint64,
    pos: wp.array[wp.vec3],
    radii: wp.array[wp.float32],
    vel: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
    num_cells: int,
    k_repulsion: float,
    k_adhesion: float,
    adhesion_range: float,
    damping: float,
):
    """Compute pairwise cell-cell forces: soft repulsion + adhesion."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i >= num_cells:
        return

    p = pos[i]
    ri = radii[i]
    v = vel[i]
    f = wp.vec3(0.0, 0.0, 0.0)

    query_radius = ri * 2.0 + adhesion_range
    neighbors = wp.hash_grid_query(grid, p, query_radius)

    for j in neighbors:
        if j == i or j >= num_cells:
            continue

        rj = radii[j]
        diff = p - pos[j]
        dist = wp.length(diff)

        if dist < 0.001:
            continue

        n = diff / dist
        overlap = ri + rj - dist

        if overlap > 0.0:
            # Soft-sphere repulsion (Hertz-like)
            f = f + n * k_repulsion * overlap * wp.sqrt(overlap)

        elif dist < ri + rj + adhesion_range:
            # Spring adhesion in the adhesion shell
            gap = dist - (ri + rj)
            f = f - n * k_adhesion * gap

    # Viscous damping
    f = f - v * damping

    forces[i] = f


@wp.kernel
def integrate_cells(
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
    num_cells: int,
    dt: float,
    domain: float,
):
    tid = wp.tid()
    if tid >= num_cells:
        return

    v_new = vel[tid] + forces[tid] * dt
    p_new = pos[tid] + v_new * dt

    # Reflective boundaries
    for dim in range(3):
        pd = p_new[dim]
        vd = v_new[dim]
        if pd < 0.5:
            pd = 1.0 - pd
            vd = wp.abs(vd)
        elif pd > domain - 0.5:
            pd = 2.0 * (domain - 0.5) - pd
            vd = -wp.abs(vd)
        if dim == 0:
            p_new = wp.vec3(pd, p_new[1], p_new[2])
            v_new = wp.vec3(vd, v_new[1], v_new[2])
        elif dim == 1:
            p_new = wp.vec3(p_new[0], pd, p_new[2])
            v_new = wp.vec3(v_new[0], vd, v_new[2])
        else:
            p_new = wp.vec3(p_new[0], p_new[1], pd)
            v_new = wp.vec3(v_new[0], v_new[1], vd)

    vel[tid] = v_new
    pos[tid] = p_new


@wp.kernel
def diffuse_nutrient(
    current: wp.array3d[wp.float32],
    output: wp.array3d[wp.float32],
    diff_rate: float,
    dt: float,
):
    """Diffuse nutrient field with zero-flux (Neumann) boundaries."""
    i, j, k = wp.tid()
    nx = current.shape[0]
    ny = current.shape[1]
    nz = current.shape[2]

    u = current[i, j, k]

    # Clamped indexing for Neumann BC
    ip = wp.min(i + 1, nx - 1)
    im = wp.max(i - 1, 0)
    jp = wp.min(j + 1, ny - 1)
    jm = wp.max(j - 1, 0)
    kp = wp.min(k + 1, nz - 1)
    km = wp.max(k - 1, 0)

    lap = (current[ip, j, k] + current[im, j, k]
           + current[i, jp, k] + current[i, jm, k]
           + current[i, j, kp] + current[i, j, km]
           - 6.0 * u)

    output[i, j, k] = u + diff_rate * lap * dt


@wp.kernel
def consume_nutrient(
    pos: wp.array[wp.vec3],
    nutrient_accum: wp.array[wp.float32],
    nutrient_field: wp.array3d[wp.float32],
    num_cells: int,
    domain: float,
    grid_size: int,
    consumption_rate: float,
    dt: float,
):
    """Cells consume nutrient from the grid and accumulate growth resource."""
    tid = wp.tid()
    if tid >= num_cells:
        return

    p = pos[tid]
    scale = float(grid_size) / domain
    ix = wp.clamp(int(p[0] * scale), 0, grid_size - 1)
    iy = wp.clamp(int(p[1] * scale), 0, grid_size - 1)
    iz = wp.clamp(int(p[2] * scale), 0, grid_size - 1)

    available = nutrient_field[ix, iy, iz]
    consumed = wp.min(available, consumption_rate * dt)

    # Reduce nutrient (atomically since multiple cells may share a voxel)
    wp.atomic_sub(nutrient_field, ix, iy, iz, consumed)

    # Accumulate growth resource
    nutrient_accum[tid] = nutrient_accum[tid] + consumed


@wp.kernel
def grow_cells(
    radii: wp.array[wp.float32],
    nutrient_accum: wp.array[wp.float32],
    num_cells: int,
    growth_rate: float,
    max_radius: float,
    dt: float,
):
    """Cells grow by converting accumulated nutrient into radius."""
    tid = wp.tid()
    if tid >= num_cells:
        return

    accum = nutrient_accum[tid]
    if accum > 0.01:
        r = radii[tid]
        # Growth proportional to nutrient reserve
        dr = growth_rate * dt
        dr = wp.min(dr, accum)  # Can't grow more than what's stored
        radii[tid] = wp.min(r + dr, max_radius)
        nutrient_accum[tid] = accum - dr


class Example:
    def __init__(self, stage_path="example_morphogenesis.usd",
                 num_initial=100, max_cells=2500, grid_size=48):
        self.max_cells = max_cells
        self.grid_size = grid_size
        self.domain = 30.0
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 30.0
        self.substeps = 4
        self.num_cells = num_initial

        # Physics
        self.k_repulsion = 200.0
        self.k_adhesion = 5.0
        self.adhesion_range = 0.8
        self.damping = 10.0
        self.growth_rate = 50.0
        self.min_radius = 0.4
        self.max_radius = 0.9
        self.divide_radius = 0.85
        self.nutrient_diffusion = 2.0
        self.consumption_rate = 2.0

        # Initialize cells in a sphere
        rng = np.random.default_rng(42)
        center = self.domain / 2.0
        positions = rng.normal(center, 2.0, (num_initial, 3)).astype(np.float32)
        positions = np.clip(positions, 1.0, self.domain - 1.0)

        # Preallocate arrays for max_cells
        pos_buf = np.zeros((max_cells, 3), dtype=np.float32)
        pos_buf[:num_initial] = positions
        self.pos = wp.array(pos_buf, dtype=wp.vec3)
        self.vel = wp.zeros(max_cells, dtype=wp.vec3)
        self.forces = wp.zeros(max_cells, dtype=wp.vec3)

        radii_buf = np.full(max_cells, self.min_radius, dtype=np.float32)
        self.radii = wp.array(radii_buf, dtype=wp.float32)
        self.nutrient_accum = wp.zeros(max_cells, dtype=wp.float32)

        # Generation tracker (for coloring)
        self.generation = np.zeros(max_cells, dtype=np.int32)

        # Nutrient field — uniform initial concentration
        n = grid_size
        nutrient = np.ones((n, n, n), dtype=np.float32) * 1.0
        self.nutrient_field = wp.array(nutrient, dtype=wp.float32)
        self.nutrient_tmp = wp.zeros((n, n, n), dtype=wp.float32)

        # HashGrid
        grid_dim = 64
        self.hash_grid = wp.HashGrid(grid_dim, grid_dim, grid_dim)
        self.cell_size = (self.max_radius * 2.0 + self.adhesion_range) * 1.1

        # Cell count history for validation
        self.count_history = [num_initial]

        # RNG for division
        self.rng = rng

        if stage_path and stage_path.endswith((".usd", ".usda", ".usdc")):
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = wp.render.NativeRenderer(512, 512)
            half = self.domain / 2.0
            self.renderer.setup_camera(
                pos=(half + self.domain * 0.6, half + self.domain * 0.3, half + self.domain * 0.6),
                target=(half, half, half),
                fov=50,
            )
            self.renderer.set_environment("dark")

    def _divide_cells(self):
        """Check which cells should divide, perform division on CPU."""
        if self.num_cells >= self.max_cells:
            return

        radii_np = self.radii.numpy()
        pos_np = self.pos.numpy()
        vel_np = self.vel.numpy()
        accum_np = self.nutrient_accum.numpy()

        new_count = self.num_cells
        for i in range(self.num_cells):
            if new_count >= self.max_cells:
                break
            if radii_np[i] >= self.divide_radius:
                # Divide along random axis
                axis = self.rng.normal(0, 1, 3).astype(np.float32)
                axis /= np.linalg.norm(axis) + 1e-8

                # Daughter cell radius
                new_r = radii_np[i] * 0.6
                offset = axis * new_r * 0.5

                # Mother shrinks
                radii_np[i] = new_r
                pos_np[i] = pos_np[i] + offset

                # Daughter
                j = new_count
                pos_np[j] = pos_np[i] - 2.0 * offset
                vel_np[j] = vel_np[i] * 0.5
                radii_np[j] = new_r
                accum_np[j] = 0.0
                self.generation[j] = self.generation[i] + 1
                accum_np[i] = 0.0

                new_count += 1

        if new_count > self.num_cells:
            self.num_cells = new_count
            self.pos = wp.array(pos_np, dtype=wp.vec3)
            self.vel = wp.array(vel_np, dtype=wp.vec3)
            self.radii = wp.array(radii_np, dtype=wp.float32)
            self.nutrient_accum = wp.array(accum_np, dtype=wp.float32)

    def step(self):
        with wp.ScopedTimer("step", active=False):
            dt = self.frame_dt / self.substeps
            n = self.num_cells
            gs = self.grid_size

            for _ in range(self.substeps):
                # Diffuse nutrient
                wp.launch(
                    kernel=diffuse_nutrient,
                    dim=(gs, gs, gs),
                    inputs=[self.nutrient_field, self.nutrient_tmp, self.nutrient_diffusion, dt],
                )
                self.nutrient_field, self.nutrient_tmp = self.nutrient_tmp, self.nutrient_field

                # Consume nutrient
                wp.launch(
                    kernel=consume_nutrient,
                    dim=n,
                    inputs=[
                        self.pos, self.nutrient_accum, self.nutrient_field,
                        n, self.domain, self.grid_size, self.consumption_rate, dt,
                    ],
                )

                # Grow
                wp.launch(
                    kernel=grow_cells,
                    dim=n,
                    inputs=[self.radii, self.nutrient_accum, n,
                            self.growth_rate, self.max_radius, dt],
                )

                # Build hash grid
                self.hash_grid.build(self.pos, self.cell_size)

                # Compute forces
                self.forces.zero_()
                wp.launch(
                    kernel=compute_cell_forces,
                    dim=n,
                    inputs=[
                        self.hash_grid.id,
                        self.pos, self.radii, self.vel, self.forces,
                        n, self.k_repulsion, self.k_adhesion,
                        self.adhesion_range, self.damping,
                    ],
                )

                # Integrate
                wp.launch(
                    kernel=integrate_cells,
                    dim=n,
                    inputs=[self.pos, self.vel, self.forces, n, dt, self.domain],
                )

            # Division (on CPU, infrequent)
            self._divide_cells()
            self.count_history.append(self.num_cells)
            self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_ground(y=0.0)

            n = self.num_cells
            # Color by generation
            max_gen = max(self.generation[:n].max(), 1)
            colors = np.zeros((n, 3), dtype=np.float32)
            for i in range(n):
                t = self.generation[i] / float(max_gen)
                # Blue → Cyan → Green → Yellow gradient
                if t < 0.33:
                    s = t / 0.33
                    colors[i] = [0.1, 0.2 + 0.6 * s, 0.8 - 0.3 * s]
                elif t < 0.66:
                    s = (t - 0.33) / 0.33
                    colors[i] = [0.1 + 0.5 * s, 0.8, 0.3 - 0.2 * s]
                else:
                    s = (t - 0.66) / 0.34
                    colors[i] = [0.6 + 0.3 * s, 0.8 - 0.2 * s, 0.1]

            # Slice arrays to current count
            pos_view = self.pos.numpy()[:n]
            radii_view = self.radii.numpy()[:n]

            self.renderer.render_points(
                name="cells",
                points=pos_view,
                radius=radii_view,
                colors=colors,
            )
            self.renderer.end_frame()

    # ---- Validation methods ----

    def get_growth_curve(self):
        """Return cell count over time (frame indices)."""
        return np.array(self.count_history)

    def compute_spatial_density(self, num_bins=10):
        """Compute 3D histogram of cell density."""
        pos = self.pos.numpy()[:self.num_cells]
        bins = np.linspace(0, self.domain, num_bins + 1)
        hist, _ = np.histogramdd(pos, bins=[bins, bins, bins])
        vol = (self.domain / num_bins) ** 3
        return hist / vol  # cells per unit volume

    def compute_pair_correlation(self, num_bins=50, r_max=10.0):
        """Compute radial pair correlation function g(r).

        Peaks at integer multiples of cell diameter indicate crystalline
        ordering; a smooth decay indicates liquid-like structure.
        """
        pos = self.pos.numpy()[:self.num_cells]
        n = len(pos)
        dr = r_max / num_bins
        hist = np.zeros(num_bins)

        for i in range(n):
            diff = pos - pos[i]
            dists = np.sqrt(np.sum(diff ** 2, axis=1))
            dists[i] = r_max + 1.0
            bins = (dists / dr).astype(int)
            mask = bins < num_bins
            np.add.at(hist, bins[mask], 1)

        # Normalize
        density = n / (self.domain ** 3)
        r = (np.arange(num_bins) + 0.5) * dr
        shell_vol = 4.0 * np.pi * r ** 2 * dr
        g_r = hist / (n * shell_vol * density + 1e-10)
        return r, g_r


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default=None,
        help="Path to output USD file. If None, uses NativeRenderer.",
    )
    parser.add_argument("--num-frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument("--num-initial", type=int, default=100, help="Initial number of cells.")
    parser.add_argument("--max-cells", type=int, default=2500, help="Maximum number of cells.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            stage_path=args.stage_path,
            num_initial=args.num_initial,
            max_cells=args.max_cells,
        )

        for i in range(args.num_frames):
            example.step()
            example.render()

            if i % 50 == 0:
                print(f"Frame {i}: cells={example.num_cells}, "
                      f"nutrient_mean={example.nutrient_field.numpy().mean():.3f}")

        if example.renderer:
            if hasattr(example.renderer, "save"):
                example.renderer.save()
            if hasattr(example.renderer, "save_image"):
                example.renderer.save_image("example_morphogenesis.png")
