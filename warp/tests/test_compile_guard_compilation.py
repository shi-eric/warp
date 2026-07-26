# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp
from warp._src.codegen import CompileFamily
from warp._src.context import ModuleBuilder
from warp.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel(module="compile_family_scalar", enable_backward=False)
def scalar_kernel(x: float, output: wp.array(dtype=float)):
    output[0] = x + x


@wp.kernel(module="compile_family_vector", enable_backward=False)
def vector_kernel(a: wp.vec3, b: wp.vec3, output: wp.array(dtype=wp.vec3)):
    output[0] = wp.cross(a, b)


@wp.kernel(module="compile_family_matrix", enable_backward=False)
def matrix_kernel(matrix: wp.mat33, vector: wp.vec3, output: wp.array(dtype=wp.vec3)):
    output[0] = matrix * vector


@wp.kernel(module="compile_family_quaternion", enable_backward=False)
def quaternion_kernel(rotation: wp.quat, vector: wp.vec3, output: wp.array(dtype=wp.vec3)):
    output[0] = wp.quat_rotate(rotation, vector)


@wp.kernel(module="compile_family_svd", enable_backward=False)
def svd_kernel(matrix: wp.mat33, output: wp.array(dtype=wp.vec3)):
    _, sigma, _ = wp.svd3(matrix)
    output[0] = sigma


@wp.kernel(module="compile_family_intersect", enable_backward=False)
def intersect_kernel(
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
    u0: wp.vec3,
    u1: wp.vec3,
    u2: wp.vec3,
    output: wp.array(dtype=wp.int32),
):
    output[0] = wp.intersect_tri_tri(v0, v1, v2, u0, u1, u2)


@wp.kernel(module="compile_family_bvh", enable_backward=False)
def bvh_kernel(
    bvh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    output: wp.array(dtype=wp.int32),
):
    query = wp.bvh_query_aabb(bvh_id, lower, upper)
    index = int(0)
    if wp.bvh_query_next(query, index):
        output[0] = index


@wp.kernel(module="compile_family_mesh", enable_backward=False)
def mesh_kernel(mesh_id: wp.uint64, output: wp.array(dtype=wp.vec3)):
    output[0] = wp.mesh_eval_position(mesh_id, 0, 0.25, 0.25)


@wp.kernel(module="compile_family_hashgrid", enable_backward=False)
def hashgrid_kernel(
    grid_id: wp.uint64,
    point: wp.vec3,
    radius: float,
    output: wp.array(dtype=wp.int32),
):
    query = wp.hash_grid_query(grid_id, point, radius)
    index = int(0)
    if wp.hash_grid_query_next(query, index):
        output[0] = index


@wp.kernel(module="compile_family_volume", enable_backward=False)
def volume_kernel(
    volume_id: wp.uint64,
    point: wp.vec3,
    output: wp.array(dtype=wp.float32),
):
    output[0] = wp.volume_sample_f(volume_id, point, wp.Volume.LINEAR)


@wp.kernel(module="compile_family_texture", enable_backward=False)
def texture_kernel(
    texture: wp.Texture2D,
    uv: wp.vec2,
    output: wp.array(dtype=wp.float32),
):
    output[0] = wp.texture_sample(texture, uv, dtype=float)


@wp.kernel(module="compile_family_random", enable_backward=False)
def random_kernel(seed: int, output: wp.array(dtype=float)):
    state = wp.rand_init(seed, wp.tid())
    output[0] = wp.randf(state)


@wp.kernel(module="compile_family_noise", enable_backward=False)
def noise_kernel(state: wp.uint32, output: wp.array(dtype=float)):
    output[0] = wp.noise(state, 0.5)


@wp.kernel(module="compile_family_float16", enable_backward=False)
def float16_kernel(x: wp.float16, output: wp.array(dtype=wp.float16)):
    output[0] = x + x


@wp.kernel(module="compile_family_float64", enable_backward=False)
def float64_kernel(x: wp.float64, output: wp.array(dtype=wp.float64)):
    output[0] = x + x


@wp.kernel(module="compile_family_tile", enable_backward=False)
def tile_kernel(input: wp.array(dtype=float), output: wp.array(dtype=float)):
    values = wp.tile_load(input, shape=16)
    wp.tile_store(output, values)


@wp.kernel(module="compile_family_backward_disabled", enable_backward=False)
def backward_disabled_kernel(x: float, output: wp.array(dtype=float)):
    output[0] = x + x


@wp.kernel(module="compile_family_backward_override", enable_backward=True)
def backward_override_kernel(x: float, output: wp.array(dtype=float)):
    output[0] = x + x


backward_disabled_kernel.module.options["enable_backward"] = False
backward_override_kernel.module.options["enable_backward"] = False


def assert_families_and_compile(test, kernel, device, expected):
    output_arch = device.arch if device.is_cuda else None
    options = kernel.module.options | {"output_arch": output_arch}
    builder = ModuleBuilder(kernel.module, options)
    source = builder.codegen("cuda" if device.is_cuda else "cpu")

    test.assertEqual(builder.required_families, expected)
    for family in CompileFamily:
        directive = f"#define {family.macro}"
        if family in expected:
            test.assertNotIn(directive, source)
        else:
            test.assertIn(directive, source)

    kernel.module.load(device)
    return source


def test_family_compilation(test, device):
    """Catch positive-family drift or a missing direct native include on real builders."""
    cases = [
        (scalar_kernel, set()),
        (vector_kernel, {CompileFamily.VECTOR}),
        (matrix_kernel, {CompileFamily.MATRIX, CompileFamily.VECTOR}),
        (quaternion_kernel, {CompileFamily.QUATERNION, CompileFamily.VECTOR}),
        (svd_kernel, {CompileFamily.SVD, CompileFamily.MATRIX, CompileFamily.VECTOR}),
        (intersect_kernel, {CompileFamily.INTERSECT, CompileFamily.VECTOR}),
        (bvh_kernel, {CompileFamily.BVH, CompileFamily.VECTOR}),
        (mesh_kernel, {CompileFamily.MESH, CompileFamily.VECTOR}),
        (hashgrid_kernel, {CompileFamily.HASHGRID, CompileFamily.VECTOR}),
        (volume_kernel, {CompileFamily.VOLUME, CompileFamily.VECTOR}),
        (random_kernel, {CompileFamily.STOCHASTIC}),
        (noise_kernel, {CompileFamily.STOCHASTIC}),
        (float16_kernel, {CompileFamily.FLOAT16}),
        (float64_kernel, set()),
    ]
    if device.is_cuda:
        cases.extend(
            [
                (texture_kernel, {CompileFamily.TEXTURE, CompileFamily.VECTOR}),
                (tile_kernel, {CompileFamily.TILE}),
            ]
        )

    use_precompiled_headers = wp.config.use_precompiled_headers
    if device.is_cpu:
        wp.config.use_precompiled_headers = False
    try:
        for kernel, expected in cases:
            with test.subTest(kernel=kernel.key):
                assert_families_and_compile(test, kernel, device, expected)
    finally:
        wp.config.use_precompiled_headers = use_precompiled_headers


def test_backward_compilation(test, device):
    """Catch Backward suppression ignoring module defaults or enabled overrides."""
    use_precompiled_headers = wp.config.use_precompiled_headers
    if device.is_cpu:
        wp.config.use_precompiled_headers = False
    try:
        disabled_source = assert_families_and_compile(test, backward_disabled_kernel, device, set())
        test.assertIn("#define WP_NO_BACKWARD", disabled_source)

        override_source = assert_families_and_compile(test, backward_override_kernel, device, set())
        test.assertNotIn("#define WP_NO_BACKWARD", override_source)
    finally:
        wp.config.use_precompiled_headers = use_precompiled_headers


class TestCompileGuardCompilation(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(
    TestCompileGuardCompilation,
    "test_family_compilation",
    test_family_compilation,
    devices=devices,
)
add_function_test(
    TestCompileGuardCompilation,
    "test_backward_compilation",
    test_backward_compilation,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
