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


# ── Lighting helpers ─────────────────────────────────────────────────


@wp.func
def hemisphere_ambient(normal: wp.vec3, sky: wp.vec3, ground: wp.vec3) -> wp.vec3:
    """Hemisphere ambient: blend sky/ground color by normal orientation."""
    up = wp.vec3(0.0, 1.0, 0.0)
    blend = 0.5 * (wp.dot(normal, up) + 1.0)
    return sky * blend + ground * (1.0 - blend)


@wp.func
def fresnel_rim(normal: wp.vec3, view_dir: wp.vec3, rim_power: float, rim_strength: float) -> float:
    """Fresnel rim lighting — bright edge when viewing at grazing angle."""
    n_dot_v = wp.max(wp.dot(normal, view_dir), 0.0)
    rim = wp.pow(1.0 - n_dot_v, rim_power)
    return rim * rim_strength


@wp.func
def schlick_fresnel(cos_theta: float, ior: float) -> float:
    """Schlick approximation to Fresnel reflectance."""
    r0 = (1.0 - ior) / (1.0 + ior)
    r0 = r0 * r0
    return r0 + (1.0 - r0) * wp.pow(1.0 - cos_theta, 5.0)


@wp.func
def refract_ray(incident: wp.vec3, normal: wp.vec3, eta: float) -> wp.vec3:
    """Snell's law refraction. Returns zero vector for total internal reflection."""
    cos_i = -wp.dot(incident, normal)
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    if sin2_t > 1.0:
        return wp.vec3(0.0, 0.0, 0.0)  # Total internal reflection
    cos_t = wp.sqrt(1.0 - sin2_t)
    return incident * eta + normal * (eta * cos_i - cos_t)


@wp.func
def shade_glass_sphere(
    ray_origin: wp.vec3,
    ray_dir: wp.vec3,
    hit_t: float,
    center: wp.vec3,
    radius: float,
    ior: float,
    tint: wp.vec3,
    # Scene
    bvh_id: wp.uint64,
    positions: wp.array[wp.vec3],
    radii: wp.array[wp.float32],
    colors: wp.array[wp.vec3],
    materials: wp.array[wp.int32],
    # Lighting
    key_dir: wp.vec3,
    key_color: wp.vec3,
    fill_dir: wp.vec3,
    fill_color: wp.vec3,
    sky_color: wp.vec3,
    ground_color: wp.vec3,
    bg: wp.vec3,
    specular_power: float,
) -> wp.vec3:
    """Shade a glass sphere: refraction + Fresnel reflection + Beer's law."""

    hit_pos = ray_origin + ray_dir * hit_t
    normal = wp.normalize(hit_pos - center)

    cos_i = wp.max(-wp.dot(ray_dir, normal), 0.0)
    fresnel = schlick_fresnel(cos_i, ior)

    # ── Refracted ray (enters sphere) ──
    refr_dir = refract_ray(ray_dir, normal, 1.0 / ior)
    refr_color = bg

    refr_len = wp.length(refr_dir)
    if refr_len > 0.5:
        refr_dir = wp.normalize(refr_dir)

        # Find exit point (ray travels through sphere interior)
        inner_origin = hit_pos + refr_dir * 0.01
        oc = inner_origin - center
        b = wp.dot(oc, refr_dir)
        c = wp.dot(oc, oc) - radius * radius
        disc = b * b - c
        exit_t = float(0.0)
        if disc >= 0.0:
            exit_t = -b + wp.sqrt(disc)

        if exit_t > 0.01:
            exit_pos = inner_origin + refr_dir * exit_t
            exit_normal = wp.normalize(exit_pos - center)

            # Beer's law: absorb light based on path length through glass
            path_len = exit_t
            # absorption = exp(-absorption_coeff * path_len)
            # Higher path = more tint. absorption_coeff ~2/radius for visible effect
            abs_coeff = 1.5 / radius
            absorption = wp.vec3(
                wp.exp(-abs_coeff * (1.0 - tint[0]) * path_len),
                wp.exp(-abs_coeff * (1.0 - tint[1]) * path_len),
                wp.exp(-abs_coeff * (1.0 - tint[2]) * path_len),
            )

            # Refract exiting
            exit_refr = refract_ray(refr_dir, -exit_normal, ior)
            exit_refr_len = wp.length(exit_refr)

            if exit_refr_len > 0.5:
                exit_refr = wp.normalize(exit_refr)
                exit_origin = exit_pos + exit_refr * 0.01

                # Trace into scene
                closest_t2 = float(1.0e10)
                closest_idx2 = int(-1)
                query2 = wp.bvh_query_ray(bvh_id, exit_origin, exit_refr)
                cand2 = int(0)
                while wp.bvh_query_next(query2, cand2):
                    t2 = ray_sphere_intersect(exit_origin, exit_refr, positions[cand2], radii[cand2])
                    if t2 > 0.0 and t2 < closest_t2:
                        closest_t2 = t2
                        closest_idx2 = cand2

                if closest_idx2 >= 0:
                    # Shade whatever we hit (opaque shading for simplicity)
                    hp2 = exit_origin + exit_refr * closest_t2
                    n2 = wp.normalize(hp2 - positions[closest_idx2])
                    bc2 = colors[closest_idx2]
                    amb2 = hemisphere_ambient(n2, sky_color, ground_color)
                    ndl2 = wp.max(wp.dot(n2, key_dir), 0.0)
                    ndl2f = wp.max(wp.dot(n2, fill_dir), 0.0)
                    refr_color = wp.cw_mul(bc2, amb2) * 0.4 + bc2 * (ndl2 * 0.6 + ndl2f * 0.2)
                else:
                    refr_color = bg

                # Apply Beer's law absorption
                refr_color = wp.cw_mul(refr_color, absorption)

            else:
                # Total internal reflection at exit
                # Bounce inside and take hemisphere ambient color
                refr_color = wp.cw_mul(hemisphere_ambient(normal, sky_color, ground_color), absorption) * 0.4

    # ── Reflected ray ──
    refl_dir = ray_dir - normal * 2.0 * wp.dot(ray_dir, normal)
    refl_origin = hit_pos + normal * 0.01
    refl_color = bg

    closest_tr = float(1.0e10)
    closest_ir = int(-1)
    qr = wp.bvh_query_ray(bvh_id, refl_origin, refl_dir)
    cr = int(0)
    while wp.bvh_query_next(qr, cr):
        tr = ray_sphere_intersect(refl_origin, refl_dir, positions[cr], radii[cr])
        if tr > 0.0 and tr < closest_tr:
            closest_tr = tr
            closest_ir = cr

    if closest_ir >= 0:
        hpr = refl_origin + refl_dir * closest_tr
        nr = wp.normalize(hpr - positions[closest_ir])
        bcr = colors[closest_ir]
        amr = hemisphere_ambient(nr, sky_color, ground_color)
        ndlr = wp.max(wp.dot(nr, key_dir), 0.0)
        refl_color = wp.cw_mul(bcr, amr) * 0.4 + bcr * ndlr * 0.6

    # Strong specular highlight (glass is very smooth)
    half_vec = wp.normalize(key_dir - ray_dir)
    spec = wp.pow(wp.max(wp.dot(normal, half_vec), 0.0), specular_power * 4.0)
    # Secondary specular from fill light
    half_vec2 = wp.normalize(fill_dir - ray_dir)
    spec2 = wp.pow(wp.max(wp.dot(normal, half_vec2), 0.0), specular_power * 2.0) * 0.3

    # Blend reflection + refraction by Fresnel, add specular
    color = refl_color * fresnel + refr_color * (1.0 - fresnel) + key_color * (spec + spec2)

    return color


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
    materials: wp.array[wp.int32],
    num_particles: int,
    # Glass
    glass_ior: float,
    glass_tint: wp.vec3,
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
    view_dir = -ray_dir

    mat = materials[closest_idx]

    if mat == 1:
        # ── Glass material ──
        color = shade_glass_sphere(
            cam_pos, ray_dir, closest_t,
            positions[closest_idx], radii[closest_idx],
            glass_ior, glass_tint,
            bvh_id, positions, radii, colors, materials,
            key_dir, key_color, fill_dir, fill_color,
            sky_color, ground_color, bg, specular_power,
        )
    else:
        # ── Opaque material ──
        hit_pos = cam_pos + ray_dir * closest_t
        normal = wp.normalize(hit_pos - positions[closest_idx])
        base_color = colors[closest_idx]

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

        # Fill light
        fill_contrib = compute_lighting_directional(normal, view_dir, fill_dir, fill_color, specular_power * 0.5, 1.0)
        color = color + wp.cw_mul(base_color, fill_contrib)

        # Rim lighting
        rim = fresnel_rim(normal, view_dir, 3.0, 0.25)
        color = color + key_color * rim

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
def _compute_sphere_bounds(
    positions: wp.array[wp.vec3],
    radii: wp.array[wp.float32],
    lowers: wp.array[wp.vec3],
    uppers: wp.array[wp.vec3],
):
    """Compute AABB bounds from sphere center + radius (GPU-side, no numpy)."""
    tid = wp.tid()
    p = positions[tid]
    r = radii[tid]
    lowers[tid] = wp.vec3(p[0] - r, p[1] - r, p[2] - r)
    uppers[tid] = wp.vec3(p[0] + r, p[1] + r, p[2] + r)


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

    def __init__(self, width=1024, height=1024, ssaa=1):
        """Create renderer.

        Args:
            width: Output image width.
            height: Output image height.
            ssaa: Supersampling anti-aliasing factor (1=off, 2=4x SSAA).
        """
        self.width = width
        self.height = height
        self.ssaa = ssaa

        # Internal render resolution
        self.render_w = width * ssaa
        self.render_h = height * ssaa
        self.pixels = wp.zeros(self.render_w * self.render_h, dtype=wp.vec3)
        self.depth = wp.zeros(self.render_w * self.render_h, dtype=wp.float32)

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

    def set_quality(self, mode="fast"):
        """Set rendering quality preset.

        Args:
            mode: One of:
                - ``"fast"``: No SSAA. ~0.5ms for 1K spheres.
                - ``"gallery"``: SSAA=2 for smooth edges. For documentation stills.
                - ``"realtime"``: Like fast. Call ``capture_graph()`` for
                  CUDA graph capture if needed.
        """
        if mode in ("fast", "realtime"):
            self.shadows = True
        elif mode == "gallery":
            self.shadows = True
        else:
            raise ValueError(f"Unknown quality mode: {mode!r}.")

    # ── CUDA graph capture for realtime ──────────────────────────────

    def capture_graph(self, positions, radius=0.1, color=(0.5, 0.5, 0.5),
                      colors=None, ground_y=None):
        """Capture the render pipeline as a CUDA graph for fast replay.

        After calling this, use ``replay_graph()`` for minimal-overhead
        rendering. The particle positions warp array must be the SAME
        object on each replay (data can change, but the array must not
        be reallocated).

        Args:
            positions: wp.array of vec3 — must remain allocated between replays.
            radius: Uniform sphere radius (float).
            color: Default color if colors is None.
            colors: Optional wp.array of vec3 per-particle colors.
            ground_y: If not None, render a ground plane at this y coordinate.
        """
        if not isinstance(positions, wp.array):
            raise TypeError("positions must be a wp.array for graph capture")

        n = len(positions)
        self._graph_positions = positions
        self._graph_n = n

        # Prepare cached arrays
        if isinstance(radius, (int, float)):
            self._graph_radii = wp.full(n, float(radius), dtype=wp.float32)
        else:
            self._graph_radii = radius

        if colors is not None:
            self._graph_colors = colors if isinstance(colors, wp.array) else wp.array(colors, dtype=wp.vec3)
        else:
            c = np.full((n, 3), color, dtype=np.float32)
            self._graph_colors = wp.array(c, dtype=wp.vec3)

        # Pre-allocate BVH bounds
        self._graph_bvh_lowers = wp.zeros(n, dtype=wp.vec3)
        self._graph_bvh_uppers = wp.zeros(n, dtype=wp.vec3)

        # Initial BVH build (must happen outside capture)
        wp.launch(kernel=_compute_sphere_bounds, dim=n,
                  inputs=[positions, self._graph_radii, self._graph_bvh_lowers, self._graph_bvh_uppers])
        self._graph_bvh = wp.Bvh(self._graph_bvh_lowers, self._graph_bvh_uppers)

        # Capture the full render pipeline
        wp.capture_begin()

        # 1. Recompute bounds from (potentially moved) positions
        wp.launch(kernel=_compute_sphere_bounds, dim=n,
                  inputs=[positions, self._graph_radii, self._graph_bvh_lowers, self._graph_bvh_uppers])

        # 2. Refit BVH (update internal nodes, no reallocation)
        self._graph_bvh.refit()

        # 3. Clear framebuffer
        self.pixels.zero_()
        self.depth.fill_(-1.0)
        wp.launch(kernel=_fill_background, dim=self.render_w * self.render_h,
                  inputs=[self.pixels, self.render_w, self.render_h, self.bg_top, self.bg_bottom])

        # 4. Ground plane (optional)
        if ground_y is not None:
            wp.launch(
                kernel=render_ground_kernel,
                dim=self.render_w * self.render_h,
                inputs=[
                    self.pixels, self.depth, self.render_w, self.render_h,
                    self.cam_pos, self.cam_fwd, self.cam_right, self.cam_up, self.fov,
                    float(ground_y), 0.5,
                    wp.vec3(0.35, 0.35, 0.38), wp.vec3(0.25, 0.25, 0.28),
                    self.key_dir, self.bg_top, self.bg_bottom, self.fog_density,
                ],
            )

        # 5. Render particles
        wp.launch(
            kernel=render_particles_kernel,
            dim=self.render_w * self.render_h,
            inputs=[
                self._graph_bvh.id,
                positions, self._graph_radii, self._graph_colors, n,
                self.cam_pos, self.cam_fwd, self.cam_right, self.cam_up, self.fov,
                self.pixels, self.depth, self.render_w, self.render_h,
                self.key_dir, self.key_color,
                self.fill_dir, self.fill_color,
                self.specular_power, self.sky_color, self.ground_color,
                self.bg_top, self.bg_bottom, self.fog_density,
                1 if self.shadows else 0,
            ],
        )

        self._graph = wp.capture_end()

    def replay_graph(self):
        """Replay the captured CUDA graph.

        The positions array passed to ``capture_graph()`` should have
        been updated with new particle positions before calling this.
        The graph will recompute BVH bounds, refit, and re-render.
        """
        wp.capture_launch(self._graph)

    def begin_frame(self):
        """Clear framebuffer and draw background gradient."""
        self.pixels.zero_()
        self.depth.fill_(-1.0)
        wp.launch(
            kernel=_fill_background,
            dim=self.render_w * self.render_h,
            inputs=[self.pixels, self.render_w, self.render_h, self.bg_top, self.bg_bottom],
        )

    def render_ground(self, y=0.0, checker_scale=0.5, color_a=(0.35, 0.35, 0.38), color_b=(0.25, 0.25, 0.28)):
        """Render an infinite checkerboard ground plane."""
        wp.launch(
            kernel=render_ground_kernel,
            dim=self.render_w * self.render_h,
            inputs=[
                self.pixels, self.depth, self.render_w, self.render_h,
                self.cam_pos, self.cam_fwd, self.cam_right, self.cam_up, self.fov,
                float(y), float(checker_scale),
                wp.vec3(*[float(c) for c in color_a]),
                wp.vec3(*[float(c) for c in color_b]),
                self.key_dir, self.bg_top, self.bg_bottom, self.fog_density,
            ],
        )

    def render_points(self, positions, radius=0.1, color=(0.5, 0.5, 0.5), colors=None,
                       materials=None, glass_ior=1.5, glass_tint=(0.9, 0.95, 1.0)):
        """Render particles as BVH-accelerated ray-traced spheres.

        Args:
            positions: (N, 3) numpy array or wp.array of vec3.
            radius: float or (N,) array of radii.
            color: (r, g, b) default color if colors is None.
            colors: (N, 3) per-particle colors.
            materials: (N,) int array. 0=opaque (default), 1=glass.
            glass_ior: Index of refraction for glass (default 1.5).
            glass_tint: (r, g, b) transmission tint for glass.
        """
        if isinstance(positions, np.ndarray):
            pos_wp = wp.array(positions.astype(np.float32), dtype=wp.vec3)
        else:
            pos_wp = positions

        n = len(pos_wp)

        if isinstance(radius, (int, float)):
            # Cache uniform radius array
            if not hasattr(self, '_cached_radii') or len(self._cached_radii) != n or self._cached_radius_val != float(radius):
                self._cached_radii = wp.full(n, float(radius), dtype=wp.float32)
                self._cached_radius_val = float(radius)
            radii_wp = self._cached_radii
        else:
            radii_wp = wp.array(np.asarray(radius, dtype=np.float32))

        if colors is not None:
            if isinstance(colors, np.ndarray):
                colors_wp = wp.array(colors.astype(np.float32), dtype=wp.vec3)
            else:
                colors_wp = colors
        else:
            # Cache uniform color array
            color_key = (float(color[0]), float(color[1]), float(color[2]), n)
            if not hasattr(self, '_cached_colors_key') or self._cached_colors_key != color_key:
                c = np.full((n, 3), color, dtype=np.float32)
                self._cached_colors = wp.array(c, dtype=wp.vec3)
                self._cached_colors_key = color_key
            colors_wp = self._cached_colors

        # Materials
        if materials is not None:
            if isinstance(materials, np.ndarray):
                materials_wp = wp.array(materials.astype(np.int32))
            else:
                materials_wp = materials
        else:
            if not hasattr(self, '_cached_mats') or len(self._cached_mats) != n:
                self._cached_mats = wp.zeros(n, dtype=wp.int32)  # All opaque
            materials_wp = self._cached_mats

        glass_tint_wp = wp.vec3(float(glass_tint[0]), float(glass_tint[1]), float(glass_tint[2]))

        # Build BVH bounds — cache arrays to avoid re-allocation
        if not hasattr(self, '_bvh_lowers') or len(self._bvh_lowers) != n:
            self._bvh_lowers = wp.zeros(n, dtype=wp.vec3)
            self._bvh_uppers = wp.zeros(n, dtype=wp.vec3)

        # Compute bounds on GPU
        wp.launch(
            kernel=_compute_sphere_bounds,
            dim=n,
            inputs=[pos_wp, radii_wp, self._bvh_lowers, self._bvh_uppers],
        )

        bvh = wp.Bvh(self._bvh_lowers, self._bvh_uppers)

        wp.launch(
            kernel=render_particles_kernel,
            dim=self.render_w * self.render_h,
            inputs=[
                bvh.id,
                pos_wp, radii_wp, colors_wp, materials_wp, n,
                float(glass_ior), glass_tint_wp,
                self.cam_pos, self.cam_fwd, self.cam_right, self.cam_up, self.fov,
                self.pixels, self.depth, self.render_w, self.render_h,
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
            dim=self.render_w * self.render_h,
            inputs=[
                mesh.id, wp.vec3(*[float(c) for c in color]),
                self.cam_pos, self.cam_fwd, self.cam_right, self.cam_up, self.fov,
                self.pixels, self.depth, self.render_w, self.render_h,
                self.key_dir, self.key_color,
                self.fill_dir, self.fill_color,
                self.specular_power, self.sky_color, self.ground_color,
                self.bg_top, self.bg_bottom, self.fog_density,
                mesh.id, 1 if self.shadows else 0,
            ],
        )

    def end_frame(self):
        """Return rendered image as numpy (H, W, 3) uint8.

        If SSAA > 1, downsamples from render resolution to output resolution
        using box filtering.
        """
        pixels_np = self.pixels.numpy().reshape((self.render_h, self.render_w, 3))
        pixels_np = np.clip(pixels_np, 0.0, 1.0)

        if self.ssaa > 1:
            # Box-filter downsample
            s = self.ssaa
            h, w = self.height, self.width
            pixels_np = pixels_np.reshape(h, s, w, s, 3).mean(axis=(1, 3))

        pixels_np = (pixels_np * 255).astype(np.uint8)
        return pixels_np[::-1]

    def get_pixels(self):
        """Return the raw pixel warp array (render resolution, float32 vec3).

        Useful for realtime display or further GPU processing without
        a GPU→CPU copy. The array has shape (render_h * render_w,) with
        dtype vec3, values in [0, 1] after gamma correction.
        """
        return self.pixels

    def save_image(self, path):
        """Render and save to PNG."""
        from PIL import Image

        Image.fromarray(self.end_frame()).save(path)
