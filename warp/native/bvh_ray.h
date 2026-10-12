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

template <bool UNBOUNDED = false>
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
    // An outside parallel slab can leave [+inf, +inf]. Scalar queries reject
    // infinite entry at their max_dist check; unbounded CUDA tiles need this
    // check, while preserving moving-axis times that both overflow.
    if (UNBOUNDED && hit && lmin == INFINITY) {
        for (int i = 0; i < 3; ++i) {
            if (bvh_ray_abs_bits(rcp_dir[i]) == 0x7f800000U) {
#if defined(__CUDA_ARCH__)
                // Preserve subnormal coordinates under fast math here too.
                uint32_t outside;
                asm("{ .reg .pred lo, hi;\n"
                    "setp.lt.f32 lo, %1, %2;\n"
                    "setp.gt.f32 hi, %1, %3;\n"
                    "or.pred lo, lo, hi;\n"
                    "selp.u32 %0, 1, 0, lo; }"
                    : "=r"(outside)
                    : "f"(start[i]), "f"(lower[i]), "f"(upper[i]));
                if (outside)
                    return false;
#else
                if (start[i] < lower[i] || start[i] > upper[i])
                    return false;
#endif
            }
        }
    }
    if (hit)
        t = lmin;
    return hit;
}

}  // namespace wp
