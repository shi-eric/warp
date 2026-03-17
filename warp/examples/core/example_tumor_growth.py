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
# Example 3D Tumor Growth
#
# Simulates avascular tumor growth with oxygen-dependent proliferation.
# Cells are agent-based soft spheres in a 3D domain. Oxygen diffuses
# from the boundary and is consumed by cells. Cells proliferate when
# oxygen is above a threshold, become quiescent at intermediate levels,
# and undergo apoptosis (death) when oxygen is too low.
#
# This produces the classic tumor morphology:
#   - Proliferating rim (high oxygen, green)
#   - Quiescent layer (moderate oxygen, yellow)
#   - Necrotic core (low oxygen, dead cells, dark)
#
# Physics:
#   - Oxygen diffusion + consumption (reaction-diffusion on grid)
#   - Soft-sphere cell mechanics (repulsion + friction)
#   - Stochastic proliferation and apoptosis
#
# Validates against:
#   - Gompertz growth curve for tumor radius
#   - Necrotic core fraction increasing over time
#   - Radial oxygen concentration profile
#
###########################################################################

import numpy as np

import warp as wp
import warp.render

# Cell states
PROLIFERATING = 0
QUIESCENT = 1
NECROTIC = 2

@wp.kernel
def diffuse_oxygen(
    current: wp.array3d[wp.float32],
    output: wp.array3d[wp.float32],
    diff_rate: float,
    boundary_value: float,
    dt: float,
):
    """Diffuse oxygen with Dirichlet boundary conditions (oxygen at boundary)."""
    i, j, k = wp.tid()
    nx = current.shape[0]
    ny = current.shape[1]
    nz = current.shape[2]

    # Dirichlet BC: boundary voxels hold fixed oxygen
    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
        output[i, j, k] = boundary_value
        return

    u = current[i, j, k]
    lap = (current[i + 1, j, k] + current[i - 1, j, k]
           + current[i, j + 1, k] + current[i, j - 1, k]
           + current[i, j, k + 1] + current[i, j, k - 1]
           - 6.0 * u)

    output[i, j, k] = wp.max(u + diff_rate * lap * dt, 0.0)


@wp.kernel
def consume_oxygen(
    cell_pos: wp.array[wp.vec3],
    cell_state: wp.array[wp.int32],
    oxygen: wp.array3d[wp.float32],
    num_cells: int,
    domain: float,
    grid_size: int,
    consumption: float,
    dt: float,
):
    """Living cells consume oxygen from the grid."""
    tid = wp.tid()
    if tid >= num_cells:
        return

    state = cell_state[tid]
    if state == 2:  # NECROTIC
        return

    p = cell_pos[tid]
    scale = float(grid_size) / domain
    ix = wp.clamp(int(p[0] * scale), 0, grid_size - 1)
    iy = wp.clamp(int(p[1] * scale), 0, grid_size - 1)
    iz = wp.clamp(int(p[2] * scale), 0, grid_size - 1)

    available = oxygen[ix, iy, iz]
    consumed = wp.min(available, consumption * dt)
    wp.atomic_sub(oxygen, ix, iy, iz, consumed)


@wp.kernel
def update_cell_states(
    cell_pos: wp.array[wp.vec3],
    cell_state: wp.array[wp.int32],
    oxygen: wp.array3d[wp.float32],
    num_cells: int,
    domain: float,
    grid_size: int,
    prolif_threshold: float,
    death_threshold: float,
):
    """Update cell state based on local oxygen level."""
    tid = wp.tid()
    if tid >= num_cells:
        return

    state = cell_state[tid]
    if state == 2:  # Already necrotic
        return

    p = cell_pos[tid]
    scale = float(grid_size) / domain
    ix = wp.clamp(int(p[0] * scale), 0, grid_size - 1)
    iy = wp.clamp(int(p[1] * scale), 0, grid_size - 1)
    iz = wp.clamp(int(p[2] * scale), 0, grid_size - 1)

    o2 = oxygen[ix, iy, iz]

    if o2 >= prolif_threshold:
        cell_state[tid] = 0  # PROLIFERATING
    elif o2 >= death_threshold:
        cell_state[tid] = 1  # QUIESCENT
    else:
        cell_state[tid] = 2  # NECROTIC (apoptosis)


@wp.kernel
def compute_tumor_forces(
    grid: wp.uint64,
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
    cell_state: wp.array[wp.int32],
    num_cells: int,
    cell_radius: float,
    k_repulsion: float,
    damping: float,
):
    """Soft-sphere repulsion between cells."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i >= num_cells:
        return

    p = pos[i]
    v = vel[i]
    f = wp.vec3(0.0, 0.0, 0.0)

    query_r = cell_radius * 3.0
    neighbors = wp.hash_grid_query(grid, p, query_r)

    for j in neighbors:
        if j == i or j >= num_cells:
            continue

        diff = p - pos[j]
        dist = wp.length(diff)

        if dist < 0.001:
            continue

        overlap = 2.0 * cell_radius - dist
        if overlap > 0.0:
            n = diff / dist
            f = f + n * k_repulsion * overlap

    # Damping
    f = f - v * damping

    forces[i] = f


@wp.kernel
def integrate_tumor_cells(
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
    cell_state: wp.array[wp.int32],
    num_cells: int,
    dt: float,
    domain: float,
):
    tid = wp.tid()
    if tid >= num_cells:
        return

    # Necrotic cells don't move
    if cell_state[tid] == 2:
        vel[tid] = wp.vec3(0.0, 0.0, 0.0)
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


class Example:
    def __init__(self, stage_path="example_tumor_growth.usd",
                 max_cells=6000, grid_size=48):
        self.max_cells = max_cells
        self.grid_size = grid_size
        self.domain = 40.0
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 30.0
        self.substeps = 4

        # Cell parameters
        self.cell_radius = 0.5
        self.k_repulsion = 100.0
        self.damping = 8.0

        # Oxygen parameters (dimensionless, inspired by literature)
        # Typical: D_O2 ~ 2e-5 cm²/s, consumption ~ 6.25e-17 mol/cell/s
        self.oxygen_diffusion = 3.0  # Scaled diffusion coefficient
        self.oxygen_consumption = 0.15  # Per-cell consumption rate
        self.boundary_oxygen = 1.0  # Normalized boundary O2
        self.prolif_threshold = 0.3  # O2 > 0.3 → proliferating
        self.death_threshold = 0.05  # O2 < 0.05 → necrotic
        self.prolif_prob = 0.15  # Probability of division per frame (if proliferating)

        # Start with single cell at center
        center = self.domain / 2.0
        self.num_cells = 1

        pos_buf = np.zeros((max_cells, 3), dtype=np.float32)
        pos_buf[0] = [center, center, center]
        self.pos = wp.array(pos_buf, dtype=wp.vec3)
        self.vel = wp.zeros(max_cells, dtype=wp.vec3)
        self.forces = wp.zeros(max_cells, dtype=wp.vec3)
        self.cell_state = wp.zeros(max_cells, dtype=wp.int32)  # All PROLIFERATING

        # Oxygen field — initialized to boundary value
        n = grid_size
        oxygen = np.ones((n, n, n), dtype=np.float32) * self.boundary_oxygen
        self.oxygen = wp.array(oxygen, dtype=wp.float32)
        self.oxygen_tmp = wp.zeros((n, n, n), dtype=wp.float32)

        # HashGrid
        self.hash_grid = wp.HashGrid(64, 64, 64)
        self.cell_size = self.cell_radius * 3.0

        # Tracking
        self.rng = np.random.default_rng(42)
        self.radius_history = []
        self.necrotic_fraction_history = []
        self.count_history = []

        if stage_path and stage_path.endswith((".usd", ".usda", ".usdc")):
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = wp.render.NativeRenderer(512, 512)
            half = self.domain / 2.0
            self.renderer.setup_camera(
                pos=(half + self.domain * 0.5, half + self.domain * 0.3, half + self.domain * 0.5),
                target=(half, half, half),
                fov=50,
            )
            self.renderer.set_environment("dark")

    def _proliferate(self):
        """Stochastic cell division for proliferating cells."""
        if self.num_cells >= self.max_cells:
            return

        state_np = self.cell_state.numpy()
        pos_np = self.pos.numpy()
        vel_np = self.vel.numpy()

        new_count = self.num_cells
        for i in range(self.num_cells):
            if new_count >= self.max_cells:
                break
            if state_np[i] == PROLIFERATING and self.rng.random() < self.prolif_prob:
                # Divide: daughter cell placed nearby
                axis = self.rng.normal(0, 1, 3).astype(np.float32)
                axis /= np.linalg.norm(axis) + 1e-8
                offset = axis * self.cell_radius * 0.6

                j = new_count
                pos_np[j] = pos_np[i] + offset
                pos_np[i] = pos_np[i] - offset
                vel_np[j] = vel_np[i] * 0.1
                state_np[j] = PROLIFERATING
                new_count += 1

        if new_count > self.num_cells:
            self.num_cells = new_count
            self.pos = wp.array(pos_np, dtype=wp.vec3)
            self.vel = wp.array(vel_np, dtype=wp.vec3)
            self.cell_state = wp.array(state_np, dtype=wp.int32)

    def step(self):
        with wp.ScopedTimer("step", active=False):
            dt = self.frame_dt / self.substeps
            n = self.num_cells
            gs = self.grid_size

            for _ in range(self.substeps):
                # Diffuse oxygen
                wp.launch(
                    kernel=diffuse_oxygen,
                    dim=(gs, gs, gs),
                    inputs=[self.oxygen, self.oxygen_tmp,
                            self.oxygen_diffusion, self.boundary_oxygen, dt],
                )
                self.oxygen, self.oxygen_tmp = self.oxygen_tmp, self.oxygen

                # Consume oxygen
                wp.launch(
                    kernel=consume_oxygen,
                    dim=n,
                    inputs=[self.pos, self.cell_state, self.oxygen,
                            n, self.domain, gs, self.oxygen_consumption, dt],
                )

                # Update cell states
                wp.launch(
                    kernel=update_cell_states,
                    dim=n,
                    inputs=[self.pos, self.cell_state, self.oxygen,
                            n, self.domain, gs,
                            self.prolif_threshold, self.death_threshold],
                )

                # Cell mechanics
                self.hash_grid.build(self.pos, self.cell_size)
                self.forces.zero_()
                wp.launch(
                    kernel=compute_tumor_forces,
                    dim=n,
                    inputs=[self.hash_grid.id, self.pos, self.vel, self.forces,
                            self.cell_state, n, self.cell_radius,
                            self.k_repulsion, self.damping],
                )

                wp.launch(
                    kernel=integrate_tumor_cells,
                    dim=n,
                    inputs=[self.pos, self.vel, self.forces, self.cell_state,
                            n, dt, self.domain],
                )

            # Proliferation (CPU, stochastic)
            self._proliferate()

            # Track metrics
            self.count_history.append(self.num_cells)
            state_np = self.cell_state.numpy()[:self.num_cells]
            n_necrotic = np.sum(state_np == NECROTIC)
            self.necrotic_fraction_history.append(
                n_necrotic / max(self.num_cells, 1))

            # Tumor radius (RMS distance from center of mass)
            pos_np = self.pos.numpy()[:self.num_cells]
            com = pos_np.mean(axis=0)
            dists = np.sqrt(np.sum((pos_np - com) ** 2, axis=1))
            self.radius_history.append(float(np.sqrt(np.mean(dists ** 2))))

            self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            self.renderer.begin_frame(self.sim_time)

            n = self.num_cells
            state_np = self.cell_state.numpy()[:n]
            colors = np.zeros((n, 3), dtype=np.float32)

            # Green = proliferating, Yellow = quiescent, Dark gray = necrotic
            for i in range(n):
                s = state_np[i]
                if s == PROLIFERATING:
                    colors[i] = [0.1, 0.85, 0.2]
                elif s == QUIESCENT:
                    colors[i] = [0.9, 0.85, 0.15]
                else:
                    colors[i] = [0.15, 0.12, 0.1]

            self.renderer.render_points(
                name="tumor",
                points=self.pos.numpy()[:n],
                radius=self.cell_radius,
                colors=colors,
            )
            self.renderer.end_frame()

    # ---- Validation methods ----

    def get_growth_curve(self):
        """Return (times, cell_counts) for Gompertz fitting."""
        times = np.arange(len(self.count_history)) * self.frame_dt
        return times, np.array(self.count_history)

    def get_radius_history(self):
        """Return tumor RMS radius over time."""
        return np.array(self.radius_history)

    def get_necrotic_fraction(self):
        """Return necrotic fraction over time."""
        return np.array(self.necrotic_fraction_history)

    def get_oxygen_profile(self):
        """Compute radial oxygen profile from tumor center.

        Returns (r, O2(r)) — should show oxygen drop toward center.
        """
        o2 = self.oxygen.numpy()
        gs = self.grid_size
        center = gs // 2
        num_bins = gs // 2

        r_vals = np.zeros(num_bins)
        o2_vals = np.zeros(num_bins)
        counts = np.zeros(num_bins)

        for i in range(gs):
            for j in range(gs):
                for k in range(gs):
                    dx = i - center
                    dy = j - center
                    dz = k - center
                    r = np.sqrt(dx * dx + dy * dy + dz * dz)
                    b = int(r)
                    if b < num_bins:
                        r_vals[b] += r
                        o2_vals[b] += o2[i, j, k]
                        counts[b] += 1

        mask = counts > 0
        r_vals[mask] /= counts[mask]
        o2_vals[mask] /= counts[mask]
        # Scale r to domain coordinates
        scale = self.domain / gs
        return r_vals * scale, o2_vals


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
    parser.add_argument("--num-frames", type=int, default=400, help="Total number of frames.")
    parser.add_argument("--max-cells", type=int, default=6000, help="Maximum number of cells.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, max_cells=args.max_cells)

        for i in range(args.num_frames):
            example.step()
            example.render()

            if i % 50 == 0:
                state = example.cell_state.numpy()[:example.num_cells]
                n_p = np.sum(state == PROLIFERATING)
                n_q = np.sum(state == QUIESCENT)
                n_n = np.sum(state == NECROTIC)
                print(f"Frame {i}: cells={example.num_cells} "
                      f"(P={n_p}, Q={n_q}, N={n_n}), "
                      f"radius={example.radius_history[-1]:.2f}")

        if example.renderer:
            if hasattr(example.renderer, "save"):
                example.renderer.save()
            if hasattr(example.renderer, "save_image"):
                example.renderer.save_image("example_tumor_growth.png")
