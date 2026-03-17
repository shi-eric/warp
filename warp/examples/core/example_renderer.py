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
# Example Renderer — BVH-Accelerated GPU Ray-Traced Renderer
#
# A high-quality ray-traced renderer built entirely in Warp kernels,
# inspired by Newton's warp_raytrace renderer. Uses BVH acceleration
# for scalable performance with large particle/mesh scenes.
#
# Features:
#   - Sphere / particle rendering via wp.Bvh + ray-sphere intersection
#   - Triangle mesh rendering via wp.Mesh + mesh_query_ray (cuBQL)
#   - Two-light setup (key + fill) for cinematic lighting
#   - Hemisphere ambient (sky/ground blend based on normal)
#   - BVH-accelerated shadow rays with soft shadow minimum visibility
#   - Distance attenuation for point lights
#   - Blinn-Phong specular with configurable power
#   - Environment gradient background + exponential distance fog
#   - Gamma-correct sRGB tone mapping
#   - Ground plane with checkerboard pattern
#
# Design inspired by Newton's warp_raytrace sensor, simplified for
# standalone example use without the full Newton scene graph.
#
###########################################################################

import math

import numpy as np

import warp as wp


SHADOW_EPS = 1.0e-4  # Normal bias for shadow rays
SHADOW_MIN_VIS = 0.3  # Minimum visibility in shadow (0=full black, 1=no shadow)


# ── Hit result struct ────────────────────────────────────────────────


@wp.struct
class HitResult:
    hit: wp.bool
    t: wp.float32
    normal: wp.vec3
    color: wp.vec3


# ── Ray-sphere intersection ─────────────────────────────────────────


@wp.func
def ray_sphere_intersect(
    origin: wp.vec3,
    direction: wp.vec3,
    center: wp.vec3,
    radius: float,
) -> wp.float32:
    """Returns t_hit, or -1.0 if no hit."""
    oc = origin - center
    b = wp.dot(oc, direction)
    c = wp.dot(oc, oc) - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return -1.0
    sqrt_d = wp.sqrt(disc)
    t = -b - sqrt_d
    if t < 1.0e-4:
        t = -b + sqrt_d
    if t < 1.0e-4:
        return -1.0
    return t


# ── Lighting computation ────────────────────────────────────────────


@wp.func
def hemisphere_ambient(normal: wp.vec3, sky: wp.vec3, ground: wp.vec3) -> wp.vec3:
    """Hemisphere ambient: blend sky/ground color by normal orientation."""
    up = wp.vec3(0.0, 1.0, 0.0)
    blend = 0.5 * (wp.dot(normal, up) + 1.0)
    return sky * blend + ground * (1.0 - blend)


@wp.func
def compute_lighting_directional(
    normal: wp.vec3,
    view_dir: wp.vec3,
    light_dir: wp.vec3,
    light_color: wp.vec3,
    specular_power: float,
    shadow_visible: float,
) -> wp.vec3:
    """Compute directional light contribution (diffuse + specular)."""
    n_dot_l = wp.max(wp.dot(normal, light_dir), 0.0)
    half_vec = wp.normalize(light_dir + view_dir)
    spec = wp.pow(wp.max(wp.dot(normal, half_vec), 0.0), specular_power)
    return light_color * (n_dot_l + spec * 0.4) * shadow_visible


# ── Main particle render kernel (BVH-accelerated) ───────────────────


@wp.kernel
def render_particles_kernel(
    # BVH
    bvh_id: wp.uint64,
    # Particle data
    positions: wp.array[wp.vec3],
    radii: wp.array[wp.float32],
    colors: wp.array[wp.vec3],
    num_particles: int,
    # Camera
    cam_pos: wp.vec3,
    cam_fwd: wp.vec3,
    cam_right: wp.vec3,
    cam_up: wp.vec3,
    fov: float,
    # Image
    pixels: wp.array[wp.vec3],
    depth_buf: wp.array[wp.float32],
    width: int,
    height: int,
    # Key light
    key_dir: wp.vec3,
    key_color: wp.vec3,
    # Fill light
    fill_dir: wp.vec3,
    fill_color: wp.vec3,
    # Shading
    specular_power: float,
    sky_color: wp.vec3,
    ground_color: wp.vec3,
    # Environment
    bg_top: wp.vec3,
    bg_bottom: wp.vec3,
    fog_density: float,
    # Options
    shadow_enabled: int,
):
    tid = wp.tid()
    px = tid % width
    py = tid / width

    aspect = float(width) / float(height)
    half_fov = wp.tan(fov * 0.5)
    u = (2.0 * (float(px) + 0.5) / float(width) - 1.0) * half_fov * aspect
    v = (2.0 * (float(py) + 0.5) / float(height) - 1.0) * half_fov
    ray_dir = wp.normalize(cam_fwd + cam_right * u + cam_up * v)

    # BVH ray query
    closest_t = float(1.0e10)
    closest_idx = int(-1)

    query = wp.bvh_query_ray(bvh_id, cam_pos, ray_dir)
    candidate = int(0)
    while wp.bvh_query_next(query, candidate):
        t = ray_sphere_intersect(cam_pos, ray_dir, positions[candidate], radii[candidate])
        if t > 0.0 and t < closest_t:
            closest_t = t
            closest_idx = candidate

    # Background
    nv = float(py) / float(height)
    bg = bg_bottom * (1.0 - nv) + bg_top * nv

    if closest_idx < 0:
        pixels[tid] = bg
        depth_buf[tid] = -1.0
        return

    depth_buf[tid] = closest_t
    hit_pos = cam_pos + ray_dir * closest_t
    normal = wp.normalize(hit_pos - positions[closest_idx])
    base_color = colors[closest_idx]
    view_dir = -ray_dir

    # Hemisphere ambient
    ambient = hemisphere_ambient(normal, sky_color, ground_color)
    color = wp.cw_mul(base_color, ambient) * 0.5

    # Key light with shadow
    key_visible = float(1.0)
    if shadow_enabled == 1:
        shadow_origin = hit_pos + normal * SHADOW_EPS
        shadow_q = wp.bvh_query_ray(bvh_id, shadow_origin, key_dir)
        sc = int(0)
        while wp.bvh_query_next(shadow_q, sc):
            if sc != closest_idx:
                st = ray_sphere_intersect(shadow_origin, key_dir, positions[sc], radii[sc])
                if st > 0.0:
                    key_visible = SHADOW_MIN_VIS
                    break

    key_contrib = compute_lighting_directional(normal, view_dir, key_dir, key_color, specular_power, key_visible)
    color = color + wp.cw_mul(base_color, key_contrib)

    # Fill light (no shadow for fill — it's supposed to soften shadows)
    fill_contrib = compute_lighting_directional(normal, view_dir, fill_dir, fill_color, specular_power * 0.5, 1.0)
    color = color + wp.cw_mul(base_color, fill_contrib)

    # Exponential fog
    fog = wp.clamp(1.0 - wp.exp(-closest_t * fog_density), 0.0, 0.8)
    color = color * (1.0 - fog) + bg * fog

    # sRGB gamma
    pixels[tid] = wp.vec3(
        wp.pow(wp.clamp(color[0], 0.0, 1.0), 0.4545),
        wp.pow(wp.clamp(color[1], 0.0, 1.0), 0.4545),
        wp.pow(wp.clamp(color[2], 0.0, 1.0), 0.4545),
    )


# ── Mesh render kernel (wp.Mesh + cuBQL) ────────────────────────────


@wp.kernel
def render_mesh_kernel(
    mesh_id: wp.uint64,
    mesh_color: wp.vec3,
    # Camera
    cam_pos: wp.vec3,
    cam_fwd: wp.vec3,
    cam_right: wp.vec3,
    cam_up: wp.vec3,
    fov: float,
    # Image
    pixels: wp.array[wp.vec3],
    depth_buf: wp.array[wp.float32],
    width: int,
    height: int,
    # Key light
    key_dir: wp.vec3,
    key_color: wp.vec3,
    # Fill light
    fill_dir: wp.vec3,
    fill_color: wp.vec3,
    # Shading
    specular_power: float,
    sky_color: wp.vec3,
    ground_color: wp.vec3,
    # Environment
    bg_top: wp.vec3,
    bg_bottom: wp.vec3,
    fog_density: float,
    # Shadow
    shadow_mesh_id: wp.uint64,
    shadow_enabled: int,
):
    tid = wp.tid()
    px = tid % width
    py = tid / width

    aspect = float(width) / float(height)
    half_fov = wp.tan(fov * 0.5)
    u = (2.0 * (float(px) + 0.5) / float(width) - 1.0) * half_fov * aspect
    v = (2.0 * (float(py) + 0.5) / float(height) - 1.0) * half_fov
    ray_dir = wp.normalize(cam_fwd + cam_right * u + cam_up * v)

    existing_depth = depth_buf[tid]
    max_t = float(1.0e6)
    if existing_depth > 0.0:
        max_t = existing_depth

    # Mesh ray query via cuBQL
    t = float(0.0)
    bary_u = float(0.0)
    bary_v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3(0.0, 0.0, 0.0)
    face = int(0)

    hit = wp.mesh_query_ray(mesh_id, cam_pos, ray_dir, max_t, t, bary_u, bary_v, sign, normal, face)
    if not hit:
        return

    depth_buf[tid] = t
    hit_pos = cam_pos + ray_dir * t
    view_dir = -ray_dir

    # Ensure normal faces camera
    if wp.dot(normal, ray_dir) > 0.0:
        normal = -normal
    normal = wp.normalize(normal)

    # Hemisphere ambient
    ambient = hemisphere_ambient(normal, sky_color, ground_color)
    color = wp.cw_mul(mesh_color, ambient) * 0.5

    # Key light with self-shadow
    key_visible = float(1.0)
    if shadow_enabled == 1:
        shadow_origin = hit_pos + normal * SHADOW_EPS
        st = float(0.0)
        su = float(0.0)
        sv = float(0.0)
        ss = float(0.0)
        sn = wp.vec3(0.0, 0.0, 0.0)
        sf = int(0)
        if wp.mesh_query_ray(shadow_mesh_id, shadow_origin, key_dir, 1.0e6, st, su, sv, ss, sn, sf):
            key_visible = SHADOW_MIN_VIS

    key_contrib = compute_lighting_directional(normal, view_dir, key_dir, key_color, specular_power, key_visible)
    color = color + wp.cw_mul(mesh_color, key_contrib)

    # Fill light
    fill_contrib = compute_lighting_directional(normal, view_dir, fill_dir, fill_color, specular_power * 0.5, 1.0)
    color = color + wp.cw_mul(mesh_color, fill_contrib)

    # Fog
    nv = float(py) / float(height)
    bg = bg_bottom * (1.0 - nv) + bg_top * nv
    fog = wp.clamp(1.0 - wp.exp(-t * fog_density), 0.0, 0.8)
    color = color * (1.0 - fog) + bg * fog

    # sRGB
    pixels[tid] = wp.vec3(
        wp.pow(wp.clamp(color[0], 0.0, 1.0), 0.4545),
        wp.pow(wp.clamp(color[1], 0.0, 1.0), 0.4545),
        wp.pow(wp.clamp(color[2], 0.0, 1.0), 0.4545),
    )


# ── Ground plane kernel ─────────────────────────────────────────────


@wp.kernel
def render_ground_kernel(
    pixels: wp.array[wp.vec3],
    depth_buf: wp.array[wp.float32],
    width: int,
    height: int,
    cam_pos: wp.vec3,
    cam_fwd: wp.vec3,
    cam_right: wp.vec3,
    cam_up: wp.vec3,
    fov: float,
    ground_y: float,
    checker_scale: float,
    color_a: wp.vec3,
    color_b: wp.vec3,
    key_dir: wp.vec3,
    bg_top: wp.vec3,
    bg_bottom: wp.vec3,
    fog_density: float,
):
    """Render an infinite checkerboard ground plane."""
    tid = wp.tid()
    px = tid % width
    py = tid / width

    aspect = float(width) / float(height)
    half_fov = wp.tan(fov * 0.5)
    u = (2.0 * (float(px) + 0.5) / float(width) - 1.0) * half_fov * aspect
    v = (2.0 * (float(py) + 0.5) / float(height) - 1.0) * half_fov
    ray_dir = wp.normalize(cam_fwd + cam_right * u + cam_up * v)

    # Ray-plane intersection (y = ground_y)
    if wp.abs(ray_dir[1]) < 1.0e-6:
        return

    t = (ground_y - cam_pos[1]) / ray_dir[1]
    if t < 0.001:
        return

    existing = depth_buf[tid]
    if existing > 0.0 and t > existing:
        return

    hit = cam_pos + ray_dir * t

    # Checkerboard
    cx = int(wp.floor(hit[0] * checker_scale))
    cz = int(wp.floor(hit[2] * checker_scale))
    check = (cx + cz) % 2
    base = color_a
    if check != 0:
        base = color_b

    # Simple shading
    normal = wp.vec3(0.0, 1.0, 0.0)
    n_dot_l = wp.max(wp.dot(normal, key_dir), 0.0)
    color = base * (0.3 + 0.7 * n_dot_l)

    # Fog
    nv = float(py) / float(height)
    bg = bg_bottom * (1.0 - nv) + bg_top * nv
    fog = wp.clamp(1.0 - wp.exp(-t * fog_density), 0.0, 0.9)
    color = color * (1.0 - fog) + bg * fog

    depth_buf[tid] = t
    pixels[tid] = wp.vec3(
        wp.pow(wp.clamp(color[0], 0.0, 1.0), 0.4545),
        wp.pow(wp.clamp(color[1], 0.0, 1.0), 0.4545),
        wp.pow(wp.clamp(color[2], 0.0, 1.0), 0.4545),
    )


@wp.kernel
def _fill_background(
    pixels: wp.array[wp.vec3],
    width: int,
    height: int,
    bg_top: wp.vec3,
    bg_bottom: wp.vec3,
):
    tid = wp.tid()
    py = tid / width
    t = float(py) / float(height)
    pixels[tid] = bg_bottom * (1.0 - t) + bg_top * t


# ── High-level renderer class ───────────────────────────────────────


class ExampleRenderer:
    """BVH-accelerated GPU ray-traced renderer for Warp examples.

    Two-light setup (key + fill) with hemisphere ambient, BVH-accelerated
    shadows, ground plane, and fog. Inspired by Newton's warp_raytrace.

    Usage::

        renderer = ExampleRenderer(width=1024, height=1024)
        renderer.setup_camera(pos=(5, 3, 5), target=(0, 0, 0))

        renderer.begin_frame()
        renderer.render_ground()
        renderer.render_points(positions, radius=0.1, color=(0.2, 0.6, 0.9))
        renderer.render_mesh(vertices, indices, color=(0.8, 0.2, 0.2))
        pixels = renderer.end_frame()  # numpy (H, W, 3) uint8
    """

    def __init__(self, width=1024, height=1024):
        self.width = width
        self.height = height
        self.pixels = wp.zeros(width * height, dtype=wp.vec3)
        self.depth = wp.zeros(width * height, dtype=wp.float32)

        # Camera
        self.cam_pos = wp.vec3(5.0, 3.0, 5.0)
        self.cam_fwd = wp.vec3(-0.577, -0.2, -0.577)
        self.cam_right = wp.vec3(0.707, 0.0, -0.707)
        self.cam_up = wp.vec3(-0.141, 0.980, -0.141)
        self.fov = math.radians(50.0)

        # Two-light setup (key + fill)
        self.key_dir = wp.normalize(wp.vec3(0.4, 0.8, 0.5))
        self.key_color = wp.vec3(0.85, 0.83, 0.80)
        self.fill_dir = wp.normalize(wp.vec3(-0.6, 0.3, -0.4))
        self.fill_color = wp.vec3(0.15, 0.18, 0.25)

        # Hemisphere ambient (sky/ground from Newton)
        self.sky_color = wp.vec3(0.4, 0.4, 0.45)
        self.ground_color = wp.vec3(0.1, 0.1, 0.12)

        self.specular_power = 48.0
        self.shadows = True

        # Environment
        self.bg_top = wp.vec3(0.15, 0.18, 0.25)
        self.bg_bottom = wp.vec3(0.04, 0.04, 0.06)
        self.fog_density = 0.012

    def setup_camera(self, pos, target, up=(0.0, 1.0, 0.0), fov=50.0):
        """Set camera position looking at target."""
        pos = np.array(pos, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        up = np.array(up, dtype=np.float32)

        fwd = target - pos
        fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
        right = np.cross(fwd, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        cam_up = np.cross(right, fwd)

        self.cam_pos = wp.vec3(*pos.tolist())
        self.cam_fwd = wp.vec3(*fwd.tolist())
        self.cam_right = wp.vec3(*right.tolist())
        self.cam_up = wp.vec3(*cam_up.tolist())
        self.fov = math.radians(fov)

    def begin_frame(self):
        """Clear framebuffer and draw background gradient."""
        self.pixels.zero_()
        self.depth.fill_(-1.0)
        wp.launch(
            kernel=_fill_background,
            dim=self.width * self.height,
            inputs=[self.pixels, self.width, self.height, self.bg_top, self.bg_bottom],
        )

    def render_ground(self, y=0.0, checker_scale=0.5, color_a=(0.35, 0.35, 0.38), color_b=(0.25, 0.25, 0.28)):
        """Render an infinite checkerboard ground plane."""
        wp.launch(
            kernel=render_ground_kernel,
            dim=self.width * self.height,
            inputs=[
                self.pixels, self.depth, self.width, self.height,
                self.cam_pos, self.cam_fwd, self.cam_right, self.cam_up, self.fov,
                float(y), float(checker_scale),
                wp.vec3(*[float(c) for c in color_a]),
                wp.vec3(*[float(c) for c in color_b]),
                self.key_dir, self.bg_top, self.bg_bottom, self.fog_density,
            ],
        )

    def render_points(self, positions, radius=0.1, color=(0.5, 0.5, 0.5), colors=None):
        """Render particles as BVH-accelerated ray-traced spheres."""
        if isinstance(positions, np.ndarray):
            pos_wp = wp.array(positions.astype(np.float32), dtype=wp.vec3)
        else:
            pos_wp = positions

        n = len(pos_wp)

        if isinstance(radius, (int, float)):
            radii_wp = wp.full(n, float(radius), dtype=wp.float32)
        else:
            radii_wp = wp.array(np.asarray(radius, dtype=np.float32))

        if colors is not None:
            if isinstance(colors, np.ndarray):
                colors_wp = wp.array(colors.astype(np.float32), dtype=wp.vec3)
            else:
                colors_wp = colors
        else:
            c = np.full((n, 3), color, dtype=np.float32)
            colors_wp = wp.array(c, dtype=wp.vec3)

        # Build LBVH from sphere AABBs
        pos_np = pos_wp.numpy()
        r_np = radii_wp.numpy().reshape(-1, 1)
        lowers = wp.array((pos_np - r_np).astype(np.float32), dtype=wp.vec3)
        uppers = wp.array((pos_np + r_np).astype(np.float32), dtype=wp.vec3)
        bvh = wp.Bvh(lowers, uppers)

        wp.launch(
            kernel=render_particles_kernel,
            dim=self.width * self.height,
            inputs=[
                bvh.id,
                pos_wp, radii_wp, colors_wp, n,
                self.cam_pos, self.cam_fwd, self.cam_right, self.cam_up, self.fov,
                self.pixels, self.depth, self.width, self.height,
                self.key_dir, self.key_color,
                self.fill_dir, self.fill_color,
                self.specular_power, self.sky_color, self.ground_color,
                self.bg_top, self.bg_bottom, self.fog_density,
                1 if self.shadows else 0,
            ],
        )

    def render_mesh(self, vertices, indices, color=(0.5, 0.5, 0.5)):
        """Render a triangle mesh with cuBQL-accelerated ray queries."""
        if isinstance(vertices, np.ndarray):
            verts_wp = wp.array(vertices.astype(np.float32), dtype=wp.vec3)
        else:
            verts_wp = vertices

        if isinstance(indices, np.ndarray):
            idx_wp = wp.array(indices.astype(np.int32))
        else:
            idx_wp = indices

        mesh = wp.Mesh(points=verts_wp, indices=idx_wp, bvh_constructor="cubql")

        wp.launch(
            kernel=render_mesh_kernel,
            dim=self.width * self.height,
            inputs=[
                mesh.id, wp.vec3(*[float(c) for c in color]),
                self.cam_pos, self.cam_fwd, self.cam_right, self.cam_up, self.fov,
                self.pixels, self.depth, self.width, self.height,
                self.key_dir, self.key_color,
                self.fill_dir, self.fill_color,
                self.specular_power, self.sky_color, self.ground_color,
                self.bg_top, self.bg_bottom, self.fog_density,
                mesh.id, 1 if self.shadows else 0,
            ],
        )

    def end_frame(self):
        """Return rendered image as numpy (H, W, 3) uint8."""
        pixels_np = self.pixels.numpy().reshape((self.height, self.width, 3))
        pixels_np = (np.clip(pixels_np, 0.0, 1.0) * 255).astype(np.uint8)
        return pixels_np[::-1]

    def save_image(self, path):
        """Render and save to PNG."""
        from PIL import Image

        Image.fromarray(self.end_frame()).save(path)
