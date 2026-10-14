// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "builtin.h"

#include "intersect.h"

namespace wp {

CUDA_CALLABLE inline uint32_t bvh_ray_abs_bits(float value)
{
#if defined(__CUDA_ARCH__)
    return __float_as_uint(value) & 0x7fffffffU;
#else
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits & 0x7fffffffU;
#endif
}

CUDA_CALLABLE inline bool bvh_ray_use_standard_slabs(const vec3& start, const vec3& direction, const vec3& rcp)
{
// CPU branches can reject outside parallel slabs early. Keep the usual
// nonparallel predicate there; CUDA uses one common path to avoid divergence.
#if !defined(__CUDA_ARCH__)
    if (bvh_ray_abs_bits(direction[0]) != 0 && bvh_ray_abs_bits(direction[1]) != 0
        && bvh_ray_abs_bits(direction[2]) != 0)
        return true;
#endif
    // Classify bits so fast math cannot mistake a subnormal direction for zero.
    for (int i = 0; i < 3; ++i) {
        const uint32_t d = bvh_ray_abs_bits(direction[i]);
        const uint32_t r = bvh_ray_abs_bits(rcp[i]);
        // Keep the existing arithmetic for nonfinite queries and for moving
        // axes whose reciprocal overflows or underflows. An infinite reciprocal
        // on such an axis does not make its slab unconstrained.
        if (bvh_ray_abs_bits(start[i]) >= 0x7f800000U || d >= 0x7f800000U
            || (d != 0 && (r < 0x00800000U || r >= 0x7f800000U)))
            return true;
    }
    return false;
}

// Ordinary masks are 0 or -1. Positive values in the first mask select a cold
// path without keeping another query flag live throughout CUDA traversal.
constexpr int32_t BVH_RAY_STANDARD_SLABS = 1;
constexpr int32_t BVH_RAY_STATIONARY_SLABS = 2;

CUDA_CALLABLE inline vec3i bvh_ray_sign_masks(const vec3& rcp_dir)
{
    vec3i masks(rcp_dir[0] < 0.0f ? -1 : 0, rcp_dir[1] < 0.0f ? -1 : 0, rcp_dir[2] < 0.0f ? -1 : 0);
#if defined(__CUDA_ARCH__)
    // Keep bitwise selects as LOP3 instructions instead of repeating the sign
    // comparisons for every node visited by the ray.
    asm("" : "+r"(masks[0]), "+r"(masks[1]), "+r"(masks[2]));
#endif
    return masks;
}

CUDA_CALLABLE inline float bvh_ray_bit_select(float a, float b, int32_t mask)
{
#if defined(__CUDA_ARCH__)
    const uint32_t m = static_cast<uint32_t>(mask);
    return __uint_as_float((__float_as_uint(a) & ~m) | (__float_as_uint(b) & m));
#else
    return mask ? b : a;
#endif
}

CUDA_CALLABLE inline void bvh_ray_slab_times(float start, float rcp_dir, float lower, float upper, float& a, float& b)
{
#if defined(__CUDA_ARCH__)
    // Preserve subnormal differences under fast math: a nearby parallel miss
    // must not become a boundary's 0 * infinity.
    float da, db;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(da) : "f"(lower), "f"(start));
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(db) : "f"(upper), "f"(start));
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(a) : "f"(da), "f"(rcp_dir));
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(b) : "f"(db), "f"(rcp_dir));
#else
    a = (lower - start) * rcp_dir;
    b = (upper - start) * rcp_dir;
#endif
}

// Order endpoints by direction sign instead of reducing them with min/max.
// The interval reductions then ignore an exact parallel face contact's NaN.
// Moving axes with exceptional reciprocals and stationary rays use cold paths.
CUDA_CALLABLE inline bool bvh_ray_intersect_signed_aabb(
    const vec3& start, const vec3& rcp_dir, const vec3i& sign_masks, const vec3& lower, const vec3& upper, float& t
)
{
    float a, b;
    bvh_ray_slab_times(start[0], rcp_dir[0], lower[0], upper[0], a, b);
    float lmin = bvh_ray_bit_select(a, b, sign_masks[0]);
    float lmax = bvh_ray_bit_select(b, a, sign_masks[0]);

    bvh_ray_slab_times(start[1], rcp_dir[1], lower[1], upper[1], a, b);
    lmin = max(bvh_ray_bit_select(a, b, sign_masks[1]), lmin);
    lmax = min(bvh_ray_bit_select(b, a, sign_masks[1]), lmax);

    bvh_ray_slab_times(start[2], rcp_dir[2], lower[2], upper[2], a, b);
    lmin = max(bvh_ray_bit_select(a, b, sign_masks[2]), lmin);
    lmax = min(bvh_ray_bit_select(b, a, sign_masks[2]), lmax);

    const bool hit = (lmax >= 0.0f) & (lmax >= lmin);
    if (hit)
        t = lmin;
    return hit;
}

CUDA_CALLABLE inline bool
bvh_ray_intersect_aabb(const vec3& start, const vec3& rcp_dir, const vec3& lower, const vec3& upper, float& t)
{
    float lmin = -INFINITY;
    float lmax = INFINITY;
    for (int i = 0; i < 3; ++i) {
#if !defined(__CUDA_ARCH__)
        if (bvh_ray_abs_bits(rcp_dir[i]) == 0x7f800000U) {
            if (start[i] < lower[i] || start[i] > upper[i])
                return false;
            continue;
        }
#endif
        float a, b;
#if defined(__CUDA_ARCH__)
        // For finite bounds and a true zero direction, NaN means an exact face contact.
        // Preserve subnormal differences through both operations under fast
        // math, so nearby points outside the slab cannot become contacts.
        float da, db;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(da) : "f"(lower[i]), "f"(start[i]));
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(db) : "f"(upper[i]), "f"(start[i]));
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(a) : "f"(da), "f"(rcp_dir[i]));
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(b) : "f"(db), "f"(rcp_dir[i]));
#else
        a = (lower[i] - start[i]) * rcp_dir[i];
        b = (upper[i] - start[i]) * rcp_dir[i];
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        // Propagate a boundary's 0 * infinity through both endpoint choices,
        // then let the interval reductions ignore that unconstrained axis.
        float near, far;
        asm("min.NaN.f32 %0, %1, %2;" : "=f"(near) : "f"(a), "f"(b));
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(far) : "f"(a), "f"(b));
        lmin = max(near, lmin);
        lmax = min(far, lmax);
#elif !defined(__CUDA_ARCH__)
        lmin = max(min(a, b), lmin);
        lmax = min(max(a, b), lmax);
#else
        if (bvh_ray_abs_bits(a) > 0x7f800000U || bvh_ray_abs_bits(b) > 0x7f800000U)
            continue;
        lmin = max(min(a, b), lmin);
        lmax = min(max(a, b), lmax);
#endif
    }
    const bool hit = (lmax >= 0.0f) & (lmax >= lmin);
    if (hit)
        t = lmin;
    return hit;
}

}  // namespace wp
