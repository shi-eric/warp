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
# Example Flocking with Predator-Prey
#
# Simulates a 3D boid flock (Reynolds rules: separation, alignment,
# cohesion) with predator-prey dynamics. Prey boids form cohesive
# schools while predator agents chase the nearest prey. Prey scatter
# when a predator is close and regroup when safe.
#
# Demonstrates:
#   - Reynolds flocking rules on GPU via HashGrid neighbor queries
#   - Predator-prey pursuit and evasion behaviors
#   - Per-particle coloring by velocity direction
#   - Emergent schooling and scattering patterns
#
# Validation:
#   - Order parameter (average alignment) — approaches 1.0 for flocks
#   - Cluster size distribution — power-law or peaked
#   - Predator capture rate — captures per unit time
#
###########################################################################

import numpy as np

import warp as wp
import warp.render


@wp.func
def wrap_delta(a: float, b: float, domain: float):
    """Compute wrapped difference a - b in periodic domain [0, domain)."""
    d = a - b
    half = domain * 0.5
    if d > half:
        d = d - domain
    elif d < -half:
        d = d + domain
    return d


@wp.func
def wrap_pos(x: float, domain: float):
    if x < 0.0:
        x = x + domain
    elif x >= domain:
        x = x - domain
    return x


@wp.kernel
def flock_step(
    grid: wp.uint64,
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    pred_pos: wp.array[wp.vec3],
    new_vel: wp.array[wp.vec3],
    # Obstacles (cylindrical pillars)
    obs_pos: wp.array[wp.vec3],
    obs_radius: wp.array[wp.float32],
    num_obs: int,
    #
    num_prey: int,
    num_pred: int,
    sep_radius: float,
    align_radius: float,
    cohesion_radius: float,
    fear_radius: float,
    max_speed: float,
    sep_weight: float,
    align_weight: float,
    cohesion_weight: float,
    fear_weight: float,
    domain: float,
    dt: float,
):
    """Compute new velocity for each prey boid using Reynolds rules + predator avoidance."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)

    if i >= num_prey:
        return

    p = pos[i]
    v = vel[i]

    # Accumulators for flocking rules (dynamic vars for loop mutation)
    sep = wp.vec3(0.0, 0.0, 0.0)
    avg_vel = wp.vec3(0.0, 0.0, 0.0)
    avg_pos = wp.vec3(0.0, 0.0, 0.0)
    n_align = int(0)
    n_coh = int(0)

    # Query neighbors
    neighbors = wp.hash_grid_query(grid, p, cohesion_radius)
    for idx in neighbors:
        if idx == i or idx >= num_prey:
            continue

        dx = wrap_delta(p[0], pos[idx][0], domain)
        dy = wrap_delta(p[1], pos[idx][1], domain)
        dz = wrap_delta(p[2], pos[idx][2], domain)
        diff = wp.vec3(dx, dy, dz)
        dist = wp.length(diff)

        if dist < sep_radius and dist > 0.001:
            sep = sep + diff / (dist * dist)

        if dist < align_radius:
            avg_vel = avg_vel + vel[idx]
            n_align = n_align + 1

        if dist < cohesion_radius:
            avg_pos = avg_pos + pos[idx] - wp.vec3(dx, dy, dz) + p
            n_coh = n_coh + 1

    steer = wp.vec3(0.0, 0.0, 0.0)

    # Separation
    steer = steer + sep * sep_weight

    # Alignment
    if n_align > 0:
        desired = avg_vel / float(n_align)
        steer = steer + (desired - v) * align_weight

    # Cohesion
    if n_coh > 0:
        center = avg_pos / float(n_coh)
        toward = center - p
        steer = steer + toward * cohesion_weight

    # Predator avoidance
    for pi in range(num_pred):
        dx = wrap_delta(p[0], pred_pos[pi][0], domain)
        dy = wrap_delta(p[1], pred_pos[pi][1], domain)
        dz = wrap_delta(p[2], pred_pos[pi][2], domain)
        diff = wp.vec3(dx, dy, dz)
        dist = wp.length(diff)
        if dist < fear_radius and dist > 0.001:
            # Strong repulsion from predator
            steer = steer + diff / (dist * dist) * fear_weight

    # Obstacle avoidance (cylindrical pillars — avoid in xz plane)
    for oi in range(num_obs):
        # Distance in xz plane to cylinder axis
        ox = p[0] - obs_pos[oi][0]
        oz = p[2] - obs_pos[oi][2]
        dist_xz = wp.sqrt(ox * ox + oz * oz)
        r = obs_radius[oi]
        avoid_dist = r + 3.0  # Start avoiding 3 units from surface

        if dist_xz < avoid_dist and dist_xz > 0.01:
            # Repel radially away from cylinder axis
            penetration = avoid_dist - dist_xz
            force = penetration * penetration * 2.0  # Quadratic repulsion
            steer = steer + wp.vec3(ox / dist_xz * force, 0.0, oz / dist_xz * force)

    # Soft boundary containment — pull toward domain center
    dom_center = domain * 0.5
    dom_margin = domain * 0.35
    for dim in range(3):
        dist_from_center = p[dim] - dom_center
        if wp.abs(dist_from_center) > dom_margin:
            overshoot = wp.abs(dist_from_center) - dom_margin
            pull = -wp.sign(dist_from_center) * overshoot * 2.0
            if dim == 0:
                steer = steer + wp.vec3(pull, 0.0, 0.0)
            elif dim == 1:
                steer = steer + wp.vec3(0.0, pull, 0.0)
            else:
                steer = steer + wp.vec3(0.0, 0.0, pull)

    # Vertical confinement — keep boids in a flat band
    y_center = domain * 0.5
    y_margin = domain * 0.15
    dy = p[1] - y_center
    if wp.abs(dy) > y_margin:
        steer = steer + wp.vec3(0.0, -wp.sign(dy) * (wp.abs(dy) - y_margin) * 4.0, 0.0)

    # Update velocity
    v_new = v + steer * dt
    spd = wp.length(v_new)
    if spd > max_speed:
        v_new = v_new * (max_speed / spd)
    if spd < 0.5:
        v_new = v_new * (0.5 / (spd + 0.001))

    new_vel[i] = v_new


@wp.kernel
def predator_step(
    pred_pos: wp.array[wp.vec3],
    pred_vel: wp.array[wp.vec3],
    prey_pos: wp.array[wp.vec3],
    obs_pos: wp.array[wp.vec3],
    obs_radius: wp.array[wp.float32],
    num_obs: int,
    num_prey: int,
    max_speed: float,
    chase_weight: float,
    domain: float,
    dt: float,
):
    """Predator chases nearest prey, avoids obstacles."""
    pid = wp.tid()

    p = pred_pos[pid]
    v = pred_vel[pid]

    # Find nearest prey
    best_dist = float(1.0e10)
    best_dir = wp.vec3(0.0, 0.0, 0.0)

    for i in range(num_prey):
        dx = wrap_delta(prey_pos[i][0], p[0], domain)
        dy = wrap_delta(prey_pos[i][1], p[1], domain)
        dz = wrap_delta(prey_pos[i][2], p[2], domain)
        diff = wp.vec3(dx, dy, dz)
        dist = wp.length(diff)
        if dist < best_dist:
            best_dist = dist
            best_dir = diff

    if best_dist > 0.001:
        chase = wp.normalize(best_dir) * chase_weight
        v = v + chase * dt

    # Obstacle avoidance for predators
    for oi in range(num_obs):
        ox = p[0] - obs_pos[oi][0]
        oz = p[2] - obs_pos[oi][2]
        dist_xz = wp.sqrt(ox * ox + oz * oz)
        r = obs_radius[oi]
        avoid_dist = r + 4.0

        if dist_xz < avoid_dist and dist_xz > 0.01:
            penetration = avoid_dist - dist_xz
            force = penetration * penetration * 3.0
            v = v + wp.vec3(ox / dist_xz * force, 0.0, oz / dist_xz * force) * dt

    spd = wp.length(v)
    if spd > max_speed:
        v = v * (max_speed / spd)

    pred_vel[pid] = v
    px = wrap_pos(p[0] + v[0] * dt, domain)
    py = wrap_pos(p[1] + v[1] * dt, domain)
    pz = wrap_pos(p[2] + v[2] * dt, domain)
    pred_pos[pid] = wp.vec3(px, py, pz)


@wp.kernel
def integrate_prey(
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    new_vel: wp.array[wp.vec3],
    num_prey: int,
    domain: float,
    dt: float,
):
    tid = wp.tid()
    if tid >= num_prey:
        return
    v = new_vel[tid]
    vel[tid] = v
    p = pos[tid]
    px = wrap_pos(p[0] + v[0] * dt, domain)
    py = wrap_pos(p[1] + v[1] * dt, domain)
    pz = wrap_pos(p[2] + v[2] * dt, domain)
    pos[tid] = wp.vec3(px, py, pz)


@wp.kernel
def check_captures(
    pred_pos: wp.array[wp.vec3],
    prey_pos: wp.array[wp.vec3],
    prey_alive: wp.array[wp.int32],
    num_prey: int,
    num_pred: int,
    capture_radius: float,
    domain: float,
    capture_count: wp.array[wp.int32],
):
    """Count prey captured (within capture_radius of any predator)."""
    tid = wp.tid()
    if tid >= num_prey:
        return
    if prey_alive[tid] == 0:
        return
    p = prey_pos[tid]
    for pi in range(num_pred):
        dx = wrap_delta(p[0], pred_pos[pi][0], domain)
        dy = wrap_delta(p[1], pred_pos[pi][1], domain)
        dz = wrap_delta(p[2], pred_pos[pi][2], domain)
        dist = wp.length(wp.vec3(dx, dy, dz))
        if dist < capture_radius:
            prey_alive[tid] = 0
            wp.atomic_add(capture_count, 0, 1)
            return


@wp.kernel
def update_glow(
    grid: wp.uint64,
    pos: wp.array[wp.vec3],
    glow: wp.array[wp.float32],
    num_prey: int,
    query_radius: float,
    glow_alpha: float,
):
    """Update neighbor density glow with exponential moving average."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i >= num_prey:
        return

    p = pos[i]
    count = int(0)

    neighbors = wp.hash_grid_query(grid, p, query_radius)
    for idx in neighbors:
        if idx != i and idx < num_prey:
            count = count + 1

    # Normalize to [0, 1] — 40 neighbors = fully glowing
    raw_glow = wp.min(1.0, float(count) / 40.0)

    # Exponential moving average for smooth transitions
    glow[i] = glow_alpha * raw_glow + (1.0 - glow_alpha) * glow[i]


@wp.kernel
def compute_colors_from_density(
    glow: wp.array[wp.float32],
    groups: wp.array[wp.int32],
    colors: wp.array[wp.vec3],
    num: int,
):
    """Color boids by neighbor density glow with per-group color ramps.

    Group 0: dark red → red → orange → yellow
    Group 1: dark blue → blue → cyan → white
    """
    tid = wp.tid()
    if tid >= num:
        return

    g = glow[tid]
    group = groups[tid]

    r = float(0.0)
    green = float(0.0)
    b = float(0.0)

    if group == 0:
        # Red ramp
        if g < 0.33:
            t = g / 0.33
            r = 0.3 + 0.5 * t
            green = 0.0
            b = 0.0
        elif g < 0.66:
            t = (g - 0.33) / 0.33
            r = 0.8 + 0.2 * t
            green = 0.3 * t
            b = 0.0
        else:
            t = (g - 0.66) / 0.34
            r = 1.0
            green = 0.3 + 0.5 * t
            b = 0.2 * t
    else:
        # Blue ramp
        if g < 0.33:
            t = g / 0.33
            r = 0.0
            green = 0.0
            b = 0.3 + 0.5 * t
        elif g < 0.66:
            t = (g - 0.33) / 0.33
            r = 0.0
            green = 0.3 * t
            b = 0.8 + 0.2 * t
        else:
            t = (g - 0.66) / 0.34
            r = 0.3 * t
            green = 0.3 + 0.5 * t
            b = 1.0

    colors[tid] = wp.vec3(r, green, b)


class Example:
    def __init__(self, stage_path="example_flocking_predator.usd", num_prey=20000, num_pred=3):
        self.num_prey = num_prey
        self.num_pred = num_pred
        self.domain = 50.0
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 30.0
        self.substeps = 4
        self.total_captures = 0

        # Flocking parameters
        self.sep_radius = 2.0
        self.align_radius = 5.0
        self.cohesion_radius = 8.0
        self.fear_radius = 10.0
        self.max_speed_prey = 8.0
        self.max_speed_pred = 6.0  # Predators slightly slower (sustained)
        self.sep_weight = 15.0
        self.align_weight = 4.0
        self.cohesion_weight = 1.0
        self.fear_weight = 50.0
        self.chase_weight = 5.0
        self.capture_radius = 1.0

        # Initialize prey in a cluster
        rng = np.random.default_rng(42)
        center = self.domain / 2.0
        prey_pos = rng.normal(center, 5.0, (num_prey, 3)).astype(np.float32)
        prey_pos = np.clip(prey_pos, 0.0, self.domain - 0.01)
        prey_vel = rng.normal(0, 2.0, (num_prey, 3)).astype(np.float32)

        self.pos = wp.array(prey_pos, dtype=wp.vec3)
        self.vel = wp.array(prey_vel, dtype=wp.vec3)
        self.new_vel = wp.zeros(num_prey, dtype=wp.vec3)
        self.prey_alive = wp.ones(num_prey, dtype=wp.int32)
        self.capture_count = wp.zeros(1, dtype=wp.int32)
        self.prey_colors = wp.zeros(num_prey, dtype=wp.vec3)

        # Groups: split boids into 2 groups for dynamic inter-group behavior
        groups = (rng.random(num_prey) * 2).astype(np.int32)
        self.groups = wp.array(groups, dtype=wp.int32)
        self.glow = wp.zeros(num_prey, dtype=wp.float32)  # Neighbor density glow

        # Initialize predators spread around domain
        pred_pos = np.array([
            [10.0, 25.0, 25.0],
            [40.0, 25.0, 25.0],
            [25.0, 25.0, 10.0],
        ][:num_pred], dtype=np.float32)
        pred_vel = np.zeros((num_pred, 3), dtype=np.float32)

        self.pred_pos = wp.array(pred_pos, dtype=wp.vec3)
        self.pred_vel = wp.array(pred_vel, dtype=wp.vec3)

        # Cylindrical pillar obstacles — arranged as a maze-like corridor
        obs_positions = []
        obs_radii = []
        center = self.domain / 2.0

        # Two rows of staggered pillars creating channels
        for row in [-1, 1]:
            for col in range(6):
                x = center + row * 8.0 + (col % 2) * 4.0
                z = center - 15.0 + col * 6.0
                obs_positions.append([x, center, z])
                obs_radii.append(2.0 + 0.5 * (col % 3))

        # A few large pillars in the center
        for angle_deg in [0, 90, 180, 270]:
            a = np.radians(angle_deg + 45)
            r = 12.0
            obs_positions.append([center + r * np.cos(a), center, center + r * np.sin(a)])
            obs_radii.append(3.0)

        self.num_obstacles = len(obs_positions)
        if self.num_obstacles > 0:
            self.obs_pos = wp.array(np.array(obs_positions, dtype=np.float32), dtype=wp.vec3)
            self.obs_radius = wp.array(np.array(obs_radii, dtype=np.float32))
        else:
            self.obs_pos = wp.zeros(1, dtype=wp.vec3)
            self.obs_radius = wp.zeros(1, dtype=wp.float32)

        # Pre-compute cylinder mesh for rendering (unit cylinder)
        self._build_cylinder_mesh()

        # HashGrid for neighbor queries
        grid_dim = 64
        self.grid = wp.HashGrid(grid_dim, grid_dim, grid_dim)
        self.grid_cell_size = self.cohesion_radius

        if stage_path and stage_path.endswith((".usd", ".usda", ".usdc")):
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = wp.render.NativeRenderer(512, 512)
            half = self.domain / 2.0
            self.renderer.setup_camera(
                pos=(half + self.domain * 0.8, half + self.domain * 0.4, half + self.domain * 0.8),
                target=(half, half, half),
                fov=50,
            )
            self.renderer.set_environment("dark")

    def _build_cylinder_mesh(self):
        """Build a cylinder mesh for rendering obstacles."""
        segments = 16
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        verts = []
        indices = []

        # Unit cylinder: radius=1, height from y=0 to y=1
        for a in angles:
            verts.append([np.cos(a), 0.0, np.sin(a)])
            verts.append([np.cos(a), 1.0, np.sin(a)])

        # Side faces
        for s in range(segments):
            b0 = s * 2
            b1 = ((s + 1) % segments) * 2
            indices.extend([b0, b1, b0 + 1])
            indices.extend([b1, b1 + 1, b0 + 1])

        self._cyl_verts = np.array(verts, dtype=np.float32)
        self._cyl_indices = np.array(indices, dtype=np.int32)

    def step(self):
        with wp.ScopedTimer("step", active=False):
            dt = self.frame_dt / self.substeps
            n = self.num_prey

            for _ in range(self.substeps):
                # Build spatial hash
                self.grid.build(self.pos, self.grid_cell_size)

                # Prey flocking
                wp.launch(
                    kernel=flock_step,
                    dim=n,
                    inputs=[
                        self.grid.id,
                        self.pos, self.vel, self.pred_pos, self.new_vel,
                        self.obs_pos, self.obs_radius, self.num_obstacles,
                        self.num_prey, self.num_pred,
                        self.sep_radius, self.align_radius, self.cohesion_radius,
                        self.fear_radius, self.max_speed_prey,
                        self.sep_weight, self.align_weight,
                        self.cohesion_weight, self.fear_weight,
                        self.domain, dt,
                    ],
                )

                # Integrate prey
                wp.launch(
                    kernel=integrate_prey,
                    dim=n,
                    inputs=[self.pos, self.vel, self.new_vel, self.num_prey, self.domain, dt],
                )

                # Predator pursuit
                wp.launch(
                    kernel=predator_step,
                    dim=self.num_pred,
                    inputs=[
                        self.pred_pos, self.pred_vel, self.pos,
                        self.obs_pos, self.obs_radius, self.num_obstacles,
                        self.num_prey, self.max_speed_pred,
                        self.chase_weight, self.domain, dt,
                    ],
                )

                # Check captures
                self.capture_count.zero_()
                wp.launch(
                    kernel=check_captures,
                    dim=n,
                    inputs=[
                        self.pred_pos, self.pos, self.prey_alive,
                        self.num_prey, self.num_pred,
                        self.capture_radius, self.domain, self.capture_count,
                    ],
                )
                self.total_captures += int(self.capture_count.numpy()[0])

            # Update glow (neighbor density) and colors
            wp.launch(
                kernel=update_glow,
                dim=n,
                inputs=[self.grid.id, self.pos, self.glow, n, self.cohesion_radius, 0.3],
            )
            wp.launch(
                kernel=compute_colors_from_density,
                dim=n,
                inputs=[self.glow, self.groups, self.prey_colors, n],
            )

            self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_ground(y=0.0)

            # Render cylindrical obstacles (instanced)
            if self.num_obstacles > 0:
                obs_np = self.obs_pos.numpy()
                rad_np = self.obs_radius.numpy()
                pillar_height = self.domain * 0.6

                positions = obs_np.copy()
                positions[:, 1] = 0.0  # Base at ground

                scales = np.zeros((self.num_obstacles, 3), dtype=np.float32)
                scales[:, 0] = rad_np      # Radius X
                scales[:, 2] = rad_np      # Radius Z
                scales[:, 1] = pillar_height  # Height

                self.renderer.render_mesh_instanced(
                    name="pillars",
                    points=self._cyl_verts,
                    indices=self._cyl_indices,
                    positions=positions,
                    scales=scales,
                    color=(0.5, 0.45, 0.4),
                )

            # Prey colored by velocity
            self.renderer.render_points(
                name="prey",
                points=self.pos,
                radius=0.3,
                colors=self.prey_colors,
            )

            # Predators in red, larger
            self.renderer.render_points(
                name="predators",
                points=self.pred_pos,
                radius=0.8,
                colors=(0.95, 0.1, 0.1),
            )

            self.renderer.end_frame()

    # ---- Validation methods ----

    def compute_order_parameter(self):
        """Compute alignment order parameter ψ = |<v̂>|.

        ψ → 1 means all boids aligned (strong flocking).
        ψ → 0 means random orientations (disordered).
        """
        vel = self.vel.numpy()
        norms = np.linalg.norm(vel, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        unit_vel = vel / norms
        mean_dir = np.mean(unit_vel, axis=0)
        return float(np.linalg.norm(mean_dir))

    def compute_cluster_sizes(self, radius=5.0):
        """Compute cluster size distribution using union-find.

        Two boids within `radius` belong to the same cluster. Returns
        sorted list of cluster sizes (largest first).
        """
        pos = self.pos.numpy()
        n = len(pos)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # O(N^2) — fine for validation on CPU
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                # Wrap
                d = d - self.domain * np.round(d / self.domain)
                if np.dot(d, d) < radius * radius:
                    union(i, j)

        from collections import Counter
        roots = [find(i) for i in range(n)]
        sizes = sorted(Counter(roots).values(), reverse=True)
        return sizes

    def get_capture_rate(self):
        """Return total prey captured so far."""
        return self.total_captures


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
    parser.add_argument("--num-prey", type=int, default=20000, help="Number of prey boids.")
    parser.add_argument("--num-pred", type=int, default=3, help="Number of predators.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            stage_path=args.stage_path,
            num_prey=args.num_prey,
            num_pred=args.num_pred,
        )

        for i in range(args.num_frames):
            example.step()
            example.render()

            if i % 50 == 0:
                psi = example.compute_order_parameter()
                print(f"Frame {i}: order={psi:.3f}, captures={example.total_captures}")

        if example.renderer:
            if hasattr(example.renderer, "save"):
                example.renderer.save()
            if hasattr(example.renderer, "save_image"):
                example.renderer.save_image("example_flocking_predator.png")
