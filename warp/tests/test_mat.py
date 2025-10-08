# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

np_signed_int_types = [np.int8, np.int16, np.int32, np.int64, np.byte]
np_float_types = [np.float16, np.float32, np.float64]


def randvals(rng, shape, dtype):
    if dtype in np_float_types:
        return rng.standard_normal(size=shape).astype(dtype)
    elif dtype in [np.int8, np.uint8, np.byte, np.ubyte]:
        return rng.integers(1, high=3, size=shape, dtype=dtype)
    return rng.integers(1, high=5, size=shape, dtype=dtype)


kernel_cache = {}


def getkernel(func, suffix=""):
    key = func.__name__ + "_" + suffix
    if key not in kernel_cache:
        kernel_cache[key] = wp.Kernel(func=func, key=key)
    return kernel_cache[key]


def get_select_kernel(dtype):
    def output_select_kernel_fn(input: wp.array(dtype=dtype), index: int, out: wp.array(dtype=dtype)):
        out[0] = input[index]

    return getkernel(output_select_kernel_fn, suffix=dtype.__name__)


def test_shape_mismatch(test, device):
    test.assertNotEqual(wp.mat33f(0.0), wp.mat22f(0.0))
    test.assertNotEqual(wp.mat22f(0.0), wp.mat33f(0.0))

    @wp.kernel(module="unique")
    def kernel():
        wp.expect_neq(wp.mat33f(0.0), wp.mat22f(0.0))
        wp.expect_neq(wp.mat22f(0.0), wp.mat33f(0.0))

    with test.assertRaisesRegex(
        RuntimeError,
        r"Can't test equality for objects with different types$",
    ):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_py_arithmetic_ops(test, device, dtype):
    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def make_mat(*args):
        if wptype in wp._src.types.int_types:
            # Cast to the correct integer type to simulate wrapping.
            return tuple(tuple(wptype._type_(x).value for x in row) for row in args)

        return args

    def make_vec(*args):
        if wptype in wp._src.types.int_types:
            # Cast to the correct integer type to simulate wrapping.
            return tuple(wptype._type_(x).value for x in args)

        return args

    mat_cls = wp.mat((3, 3), wptype)
    vec_cls = wp.vec(3, wptype)

    m = mat_cls(((-1, 2, 3), (4, -5, 6), (7, 8, -9)))
    test.assertSequenceEqual(+m, make_mat((-1, 2, 3), (4, -5, 6), (7, 8, -9)))
    test.assertSequenceEqual(-m, make_mat((1, -2, -3), (-4, 5, -6), (-7, -8, 9)))
    test.assertSequenceEqual(m + mat_cls((5, 5, 5) * 3), make_mat((4, 7, 8), (9, 0, 11), (12, 13, -4)))
    test.assertSequenceEqual(m - mat_cls((5, 5, 5) * 3), make_mat((-6, -3, -2), (-1, -10, 1), (2, 3, -14)))
    test.assertSequenceEqual(m * vec_cls(5, 5, 5), make_vec(20, 25, 30))
    test.assertSequenceEqual(m @ vec_cls(5, 5, 5), make_vec(20, 25, 30))
    test.assertSequenceEqual(vec_cls(5, 5, 5) * m, make_vec(50, 25, 0))
    test.assertSequenceEqual(vec_cls(5, 5, 5) @ m, make_vec(50, 25, 0))

    m = mat_cls(((2, 4, 6), (8, 10, 12), (14, 16, 18)))
    test.assertSequenceEqual(m * wptype(2), make_mat((4, 8, 12), (16, 20, 24), (28, 32, 36)))
    test.assertSequenceEqual(wptype(2) * m, make_mat((4, 8, 12), (16, 20, 24), (28, 32, 36)))
    test.assertSequenceEqual(m / wptype(2), make_mat((1, 2, 3), (4, 5, 6), (7, 8, 9)))
    test.assertSequenceEqual(wptype(5040) / m, make_mat((2520, 1260, 840), (630, 504, 420), (360, 315, 280)))
    test.assertSequenceEqual(m * vec_cls(5, 5, 5), make_vec(60, 150, 240))
    test.assertSequenceEqual(m @ vec_cls(5, 5, 5), make_vec(60, 150, 240))
    test.assertSequenceEqual(vec_cls(5, 5, 5) * m, make_vec(120, 150, 180))
    test.assertSequenceEqual(vec_cls(5, 5, 5) @ m, make_vec(120, 150, 180))


def test_negation(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp._src.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)
    mat55 = wp._src.types.matrix(shape=(5, 5), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_negation(
        m2: wp.array(dtype=mat22),
        m3: wp.array(dtype=mat33),
        m4: wp.array(dtype=mat44),
        m5: wp.array(dtype=mat55),
        outcomponents: wp.array(dtype=wptype),
    ):
        mat2 = -m2[0]
        mat3 = -m3[0]
        mat4 = -m4[0]
        mat5 = -m5[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = wptype(2) * mat2[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * mat3[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = wptype(2) * mat4[i, j]
                idx = idx + 1

        for i in range(5):
            for j in range(5):
                outcomponents[idx] = wptype(2) * mat5[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_negation, suffix=dtype.__name__)

    if register_kernels:
        return

    m2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    m3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    m4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    m5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * 2 + 3 * 3 + 4 * 4 + 5 * 5, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[m2, m3, m4, m5], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy()[:4], -2 * m2.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:13], -2 * m3.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[13:29], -2 * m4.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[29:54], -2 * m5.numpy().reshape(-1), tol=tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, input in [(2, m2), (3, m3), (4, m4), (5, m5)]:
            for i in range(dim):
                for j in range(dim):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[m2, m3, m4, m5], outputs=[outcomponents], device=device)
                        wp.launch(
                            output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device
                        )
                    tape.backward(loss=out)
                    expectedresult = np.zeros((dim, dim), dtype=dtype)
                    expectedresult[i, j] = -2
                    assert_np_equal(tape.gradients[input].numpy()[0], expectedresult)
                    tape.zero()
                    idx = idx + 1


def test_matmul(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-12,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp._src.types.matrix(shape=(3, 3), dtype=wptype)
    mat23 = wp._src.types.matrix(shape=(2, 3), dtype=wptype)
    mat32 = wp._src.types.matrix(shape=(3, 2), dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_mul(
        i23: wp.array(dtype=mat23),
        i32: wp.array(dtype=mat32),
        i44: wp.array(dtype=mat44),
        o22: wp.array(dtype=mat22),
        o33: wp.array(dtype=mat33),
        o44: wp.array(dtype=mat44),
    ):
        i = wp.tid()
        o22[i] = i23[i] @ i32[i]
        o33[i] = i32[i] @ i23[i]
        o44[i] = i44[i] @ i44[i]

    kernel = getkernel(check_mat_mul, suffix=dtype.__name__)

    if register_kernels:
        return

    test_adj = dtype in np_float_types

    i23 = wp.array(randvals(rng, [1, 2, 3], dtype), dtype=mat23, requires_grad=test_adj, device=device)
    i32 = wp.array(randvals(rng, [1, 3, 2], dtype), dtype=mat32, requires_grad=test_adj, device=device)
    i44 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=test_adj, device=device)
    o22 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=test_adj, device=device)
    o33 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=test_adj, device=device)
    o44 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=test_adj, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[i23, i32, i44],
            outputs=[o22, o33, o44],
            device=device,
        )

    assert_np_equal(o22.numpy(), i23.numpy() @ i32.numpy(), tol=tol)
    assert_np_equal(o33.numpy(), i32.numpy() @ i23.numpy(), tol=tol)
    assert_np_equal(o44.numpy(), i44.numpy() @ i44.numpy(), tol=tol)

    if test_adj:
        o22.grad.assign([np.eye(2)])
        o33.grad.assign([np.eye(3)])
        o44.grad.assign([np.eye(4)])

        tape.backward()

        assert_np_equal(i23.grad.numpy(), 2.0 * i32.numpy().T, tol=tol)
        assert_np_equal(i32.grad.numpy(), 2.0 * i23.numpy().T, tol=tol)
        assert_np_equal(i44.grad.numpy(), 2.0 * i44.numpy().T, tol=tol)


def test_subtraction(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp._src.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)
    mat55 = wp._src.types.matrix(shape=(5, 5), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_sub(
        s2: wp.array(dtype=mat22),
        s3: wp.array(dtype=mat33),
        s4: wp.array(dtype=mat44),
        s5: wp.array(dtype=mat55),
        v2: wp.array(dtype=mat22),
        v3: wp.array(dtype=mat33),
        v4: wp.array(dtype=mat44),
        v5: wp.array(dtype=mat55),
        outcomponents: wp.array(dtype=wptype),
    ):
        v2result = v2[0] - s2[0]
        v3result = v3[0] - s3[0]
        v4result = v4[0] - s4[0]
        v5result = v5[0] - s5[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = wptype(2) * v2result[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * v3result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = wptype(2) * v4result[i, j]
                idx = idx + 1

        for i in range(5):
            for j in range(5):
                outcomponents[idx] = wptype(2) * v5result[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_sub, suffix=dtype.__name__)

    if register_kernels:
        return

    s2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    s3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    s4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    s5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    v2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * 2 + 3 * 3 + 4 * 4 + 5 * 5, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[
            s2,
            s3,
            s4,
            s5,
            v2,
            v3,
            v4,
            v5,
        ],
        outputs=[outcomponents],
        device=device,
    )

    assert_np_equal(outcomponents.numpy()[:4], 2 * (v2.numpy() - s2.numpy()).reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:13], 2 * (v3.numpy() - s3.numpy()).reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[13:29], 2 * (v4.numpy() - s4.numpy()).reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[29:54], 2 * (v5.numpy() - s5.numpy()).reshape(-1), tol=10 * tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, in1, in2 in [(2, s2, v2), (3, s3, v3), (4, s4, v4), (5, s5, v5)]:
            for i in range(dim):
                for j in range(dim):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(
                            kernel,
                            dim=1,
                            inputs=[s2, s3, s4, s5, v2, v3, v4, v5],
                            outputs=[outcomponents],
                            device=device,
                        )
                        wp.launch(
                            output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device
                        )
                    tape.backward(loss=out)
                    expected_result = np.zeros((dim, dim), dtype=dtype)
                    expected_result[i, j] = 2
                    assert_np_equal(tape.gradients[in2].numpy()[0], expected_result, tol=10 * tol)
                    expected_result[i, j] = -2
                    assert_np_equal(tape.gradients[in1].numpy()[0], expected_result, tol=10 * tol)
                    tape.zero()

                    idx = idx + 1


def test_determinant(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp._src.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)

    def check_mat_det(
        v2: wp.array(dtype=mat22),
        v3: wp.array(dtype=mat33),
        v4: wp.array(dtype=mat44),
        det2: wp.array(dtype=wptype),
        det3: wp.array(dtype=wptype),
        det4: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        det2[0] = wptype(2) * wp.determinant(v2[0])
        det3[0] = wptype(2) * wp.determinant(v3[0])
        det4[0] = wptype(2) * wp.determinant(v4[0])

    kernel = getkernel(check_mat_det, suffix=dtype.__name__)
    if register_kernels:
        return

    v2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    det2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    det3 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    det4 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[v2, v3, v4], outputs=[det2, det3, det4], device=device)

    if dtype in np_float_types:
        assert_np_equal(det2.numpy()[0], 2 * np.linalg.det(v2.numpy()[0].astype(np.float64)), tol=100 * tol)
        assert_np_equal(det3.numpy()[0], 2 * np.linalg.det(v3.numpy()[0].astype(np.float64)), tol=100 * tol)
        assert_np_equal(det4.numpy()[0], 2 * np.linalg.det(v4.numpy()[0].astype(np.float64)), tol=420 * tol)
    else:
        assert_np_equal(det2.numpy()[0], 2 * np.around(np.linalg.det(v2.numpy()[0])).astype(int))
        assert_np_equal(det3.numpy()[0], 2 * np.around(np.linalg.det(v3.numpy()[0])).astype(int))
        assert_np_equal(det4.numpy()[0], 2 * np.around(np.linalg.det(v4.numpy()[0])).astype(int))

    if dtype in np_float_types:
        # determinant derivative formula is annoying so finite differences?
        tape.backward(loss=det2)
        v2grads = 1.0 * tape.gradients[v2].numpy()[0]
        tape.zero()

        tape.backward(loss=det3)
        v3grads = 1.0 * tape.gradients[v3].numpy()[0]
        tape.zero()

        tape.backward(loss=det4)
        v4grads = 1.0 * tape.gradients[v4].numpy()[0]
        tape.zero()

        # finite differences are also annoying hence the large tolerance...
        # absolute nightmare in float16 too innit...
        dx = 0.01 if dtype == np.float16 else 0.0001
        fdtol = 2.0e-1 if dtype == np.float16 else 2.0e-3
        for i in range(2):
            for j in range(2):
                v2test = v2.numpy()
                v2test[0, i, j] += dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[wp.array(v2test, dtype=v2.dtype, requires_grad=True, device=device), v3, v4],
                    outputs=[det2, det3, det4],
                    device=device,
                )
                dplus = det2.numpy()[0]
                v2test[0, i, j] -= 2.0 * dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[wp.array(v2test, dtype=v2.dtype, requires_grad=True, device=device), v3, v4],
                    outputs=[det2, det3, det4],
                    device=device,
                )
                dminus = det2.numpy()[0]
                assert_np_equal((dplus - dminus) / (2.0 * dx * dplus), v2grads[i, j] / dplus, tol=fdtol)

        for i in range(3):
            for j in range(3):
                v3test = v3.numpy()
                v3test[0, i, j] += dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[v2, wp.array(v3test, dtype=v3.dtype, requires_grad=True, device=device), v4],
                    outputs=[det2, det3, det4],
                    device=device,
                )
                dplus = det3.numpy()[0]
                v3test[0, i, j] -= 2.0 * dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[v2, wp.array(v3test, dtype=v3.dtype, requires_grad=True, device=device), v4],
                    outputs=[det2, det3, det4],
                    device=device,
                )
                dminus = det3.numpy()[0]
                assert_np_equal((dplus - dminus) / (2.0 * dx * dplus), v3grads[i, j] / dplus, tol=fdtol)

        for i in range(4):
            for j in range(4):
                v4test = v4.numpy()
                v4test[0, i, j] += dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[v2, v3, wp.array(v4test, dtype=v4.dtype, requires_grad=True, device=device)],
                    outputs=[det2, det3, det4],
                    device=device,
                )
                dplus = det4.numpy()[0]
                v4test[0, i, j] -= 2.0 * dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[v2, v3, wp.array(v4test, dtype=v4.dtype, requires_grad=True, device=device)],
                    outputs=[det2, det3, det4],
                    device=device,
                )
                dminus = det4.numpy()[0]
                assert_np_equal((dplus - dminus) / (2.0 * dx * dplus), v4grads[i, j] / dplus, tol=fdtol)


# Unused. Why?
# def test_get_diag(test, device, dtype, register_kernels=False):
#     tol = {
#         np.float16: 1.0e-3,
#         np.float32: 1.0e-6,
#         np.float64: 1.0e-8,
#     }.get(dtype, 0)
#
#     wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
#     mat55 = wp._src.types.vector(shape=(5, 5), dtype=wptype)
#
#     output_select_kernel = get_select_kernel(wptype)
#
#     def check_mat_diag(
#         m55: wp.array(dtype=mat55),
#         outcomponents: wp.array(dtype=wptype),
#     ):
#         # multiply outputs by 2 so we've got something to backpropagate:
#         vec5result = wptype(2) * wp.get_diag(m55[0])
#
#         idx = 0
#         for i in range(5):
#             outcomponents[idx] = vec5result[i]
#             idx = idx + 1
#
#     kernel = getkernel(check_mat_diag, suffix=dtype.__name__)
#
#     if register_kernels:
#         return
#
#     m55 = wp.array(randvals((1, 5, 5), dtype), dtype=mat55, requires_grad=True, device=device)
#     outcomponents = wp.zeros(5, dtype=wptype, requires_grad=True, device=device)
#     out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
#
#     wp.launch(kernel, dim=1, inputs=[m55], outputs=[outcomponents], device=device)
#
#     assert_np_equal(outcomponents.numpy(), 2 * np.diag(m55.numpy()[0]), tol=tol)
#
#     if dtype in np_float_types:
#         idx = 0
#         for i in range(5):
#             tape = wp.Tape()
#             with tape:
#                 wp.launch(kernel, dim=1, inputs=[m55], outputs=[outcomponents], device=device)
#                 wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
#             tape.backward(loss=out)
#             expectedresult = np.zeros((5, 5), dtype=dtype)
#             expectedresult[i, i] = 2
#             assert_np_equal(tape.gradients[m55].numpy()[0], expectedresult, tol=10 * tol)
#             tape.zero()
#
#             idx = idx + 1


def test_inverse(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-2,
        np.float32: 1.0e-5,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp._src.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_inverse(
        m2: wp.array(dtype=mat22),
        m3: wp.array(dtype=mat33),
        m4: wp.array(dtype=mat44),
        outcomponents: wp.array(dtype=wptype),
    ):
        m2result = wp.inverse(m2[0])
        m3result = wp.inverse(m3[0])
        m4result = wp.inverse(m4[0])

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = wptype(2) * m2result[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * m3result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = wptype(2) * m4result[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_inverse, suffix=dtype.__name__)

    if register_kernels:
        return

    m2 = wp.array(
        2 * (randvals(rng, [1, 2, 2], dtype) + 0.2 * np.eye(2)), dtype=mat22, requires_grad=True, device=device
    )
    m3 = wp.array(
        2 * (randvals(rng, [1, 3, 3], dtype) + 0.2 * np.eye(3)), dtype=mat33, requires_grad=True, device=device
    )
    m4 = wp.array(
        2 * (randvals(rng, [1, 4, 4], dtype) + 0.2 * np.eye(4)), dtype=mat44, requires_grad=True, device=device
    )

    outcomponents = wp.zeros(2 * 2 + 3 * 3 + 4 * 4, dtype=wptype, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[m2, m3, m4], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy()[:4], 2 * np.linalg.inv(m2.numpy()[0].astype(np.float64)), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:13], 2 * np.linalg.inv(m3.numpy()[0].astype(np.float64)), tol=5 * tol)
    assert_np_equal(outcomponents.numpy()[13:], 2 * np.linalg.inv(m4.numpy()[0].astype(np.float64)), tol=5 * tol)

    if dtype in np_float_types:
        # check gradients:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, input in [(2, m2), (3, m3), (4, m4)]:
            minv = np.linalg.inv(input.numpy()[0].astype(np.float64))
            for i in range(dim):
                for j in range(dim):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[m2, m3, m4], outputs=[outcomponents], device=device)
                        wp.launch(
                            output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device
                        )
                    tape.backward(loss=out)
                    d = np.zeros((dim, dim))
                    d[j, i] = 2
                    assert_np_equal(
                        tape.gradients[input].numpy()[0], -np.matmul(minv, np.matmul(d, minv)).T, tol=10 * tol
                    )
                    tape.zero()

                    idx = idx + 1

    # let's check 2x2 using different formulae just for (in)sanity's sake:
    m = m2.numpy()[0]

    det = m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]
    expected = 2 * np.array([[m[1, 1], -m[0, 1]], [-m[1, 0], m[0, 0]]], dtype=dtype) / det
    assert_np_equal(expected, outcomponents.numpy()[:4], tol=tol)

    # 0,0 component is this:
    # 2 * m[1,1] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])
    assert_np_equal(2 * m[1, 1] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]), outcomponents.numpy()[0], tol=tol)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[m2, m3, m4], outputs=[outcomponents], device=device)
        wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, 0], outputs=[out], device=device)

    if dtype in np_float_types:
        tape.backward(loss=out)
        g = tape.gradients[m2].numpy()[0]
        assert_np_equal(-2 * m[1, 1] * m[1, 1] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[0, 0], tol=tol)
        assert_np_equal(2 * m[1, 1] * m[0, 1] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[1, 0], tol=tol)
        assert_np_equal(-2 * m[0, 1] * m[1, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[1, 1], tol=tol)
        assert_np_equal(2 * m[1, 1] * m[1, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[0, 1], tol=tol)
        tape.zero()

    # 0,1 component is this:
    # -2 * m[0,1] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])
    assert_np_equal(-2 * m[0, 1] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]), outcomponents.numpy()[1], tol=tol)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[m2, m3, m4], outputs=[outcomponents], device=device)
        wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, 1], outputs=[out], device=device)
    if dtype in np_float_types:
        tape.backward(loss=out)
        g = tape.gradients[m2].numpy()[0]
        assert_np_equal(2 * m[0, 1] * m[1, 1] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[0, 0], tol=tol)
        assert_np_equal(-2 * m[0, 1] * m[0, 1] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[1, 0], tol=tol)
        assert_np_equal(2 * m[0, 0] * m[0, 1] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[1, 1], tol=tol)
        assert_np_equal(-2 * m[1, 1] * m[0, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[0, 1], tol=tol)
        tape.zero()

    # 1,0 component is this:
    # -2 * m[1,0] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])
    assert_np_equal(-2 * m[1, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]), outcomponents.numpy()[2], tol=tol)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[m2, m3, m4], outputs=[outcomponents], device=device)
        wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, 2], outputs=[out], device=device)

    if dtype in np_float_types:
        tape.backward(loss=out)
        g = tape.gradients[m2].numpy()[0]
        assert_np_equal(2 * m[1, 1] * m[1, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[0, 0], tol=tol)
        assert_np_equal(-2 * m[0, 0] * m[1, 1] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[1, 0], tol=tol)
        assert_np_equal(2 * m[0, 0] * m[1, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[1, 1], tol=tol)
        assert_np_equal(-2 * m[1, 0] * m[1, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[0, 1], tol=tol)
        tape.zero()

    # 1,1 component is this:
    # 2 * m[0,0] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])
    assert_np_equal(2 * m[0, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]), outcomponents.numpy()[3], tol=tol)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[m2, m3, m4], outputs=[outcomponents], device=device)
        wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, 3], outputs=[out], device=device)

    if dtype in np_float_types:
        tape.backward(loss=out)
        g = tape.gradients[m2].numpy()[0]
        assert_np_equal(-2 * m[0, 1] * m[1, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[0, 0], tol=tol)
        assert_np_equal(2 * m[0, 0] * m[0, 1] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[1, 0], tol=tol)
        assert_np_equal(2 * m[0, 0] * m[1, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[0, 1], tol=tol)
        assert_np_equal(-2 * m[0, 0] * m[0, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[1, 1], tol=tol)
        tape.zero()


def test_svd(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-12,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec3 = wp._src.types.vector(length=3, dtype=wptype)
    mat33 = wp._src.types.matrix(shape=(3, 3), dtype=wptype)

    def check_mat_svd(
        m3: wp.array(dtype=mat33),
        Uout: wp.array(dtype=mat33),
        sigmaout: wp.array(dtype=vec3),
        Vout: wp.array(dtype=mat33),
        outcomponents: wp.array(dtype=wptype),
    ):
        U = mat33()
        sigma = vec3()
        V = mat33()

        wp.svd3(m3[0], U, sigma, V)

        Uout[0] = U
        sigmaout[0] = sigma
        Vout[0] = V

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * U[i, j]
                idx = idx + 1

        for i in range(3):
            outcomponents[idx] = wptype(2) * sigma[i]
            idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * V[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_svd, suffix=dtype.__name__)

    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    m3 = wp.array(randvals(rng, [1, 3, 3], dtype) + np.eye(3), dtype=mat33, requires_grad=True, device=device)

    outcomponents = wp.zeros(2 * 3 * 3 + 3, dtype=wptype, requires_grad=True, device=device)
    Uout = wp.zeros(1, dtype=mat33, requires_grad=True, device=device)
    sigmaout = wp.zeros(1, dtype=vec3, requires_grad=True, device=device)
    Vout = wp.zeros(1, dtype=mat33, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[m3], outputs=[Uout, sigmaout, Vout, outcomponents], device=device)

    Uout_np = Uout.numpy()[0].astype(np.float64)
    sigmaout_np = np.diag(sigmaout.numpy()[0].astype(np.float64))
    Vout_np = Vout.numpy()[0].astype(np.float64)

    assert_np_equal(
        np.matmul(Uout_np, np.matmul(sigmaout_np, Vout_np.T)), m3.numpy()[0].astype(np.float64), tol=30 * tol
    )

    if dtype == np.float16:
        # I'm not even going to bother testing the gradients for float16
        # because the rounding errors are terrible...
        return

    # check gradients:
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    idx = 0
    for idx in range(3 * 3 + 3 + 3 * 3):
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[m3], outputs=[Uout, sigmaout, Vout, outcomponents], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
        tape.backward(out)
        m3grads = 1.0 * tape.gradients[m3].numpy()[0]

        tape.zero()

        dx = 0.0001
        fdtol = 5.0e-4 if dtype == np.float64 else 2.0e-2
        for ii in range(3):
            for jj in range(3):
                m3test = 1.0 * m3.numpy()
                m3test[0, ii, jj] += dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[wp.array(m3test, dtype=mat33, device=device)],
                    outputs=[Uout, sigmaout, Vout, outcomponents],
                    device=device,
                )
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
                plusval = out.numpy()[0]

                m3test = 1.0 * m3.numpy()
                m3test[0, ii, jj] -= dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[wp.array(m3test, dtype=mat33, device=device)],
                    outputs=[Uout, sigmaout, Vout, outcomponents],
                    device=device,
                )
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
                minusval = out.numpy()[0]

                assert_np_equal((plusval - minusval) / (2 * dx), m3grads[ii, jj], tol=fdtol)


def test_svd_2D(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-12,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp._src.types.vector(length=2, dtype=wptype)
    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)

    def check_mat_svd2(
        m2: wp.array(dtype=mat22),
        Uout: wp.array(dtype=mat22),
        sigmaout: wp.array(dtype=vec2),
        Vout: wp.array(dtype=mat22),
        outcomponents: wp.array(dtype=wptype),
    ):
        tid = wp.tid()

        U = mat22()
        sigma = vec2()
        V = mat22()

        wp.svd2(m2[tid], U, sigma, V)  # Assuming there's a 2D SVD kernel

        Uout[tid] = U
        sigmaout[tid] = sigma
        Vout[tid] = V

        # backprop test only for first input
        if tid > 0:
            return

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = wptype(2) * U[i, j]
                idx = idx + 1

        for i in range(2):
            outcomponents[idx] = wptype(2) * sigma[i]
            idx = idx + 1

        for i in range(2):
            for j in range(2):
                outcomponents[idx] = wptype(2) * V[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_svd2, suffix=dtype.__name__)

    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    mats = np.concatenate(
        (
            randvals(rng, [24, 2, 2], dtype) + np.eye(2),
            # rng unlikely to hit edge cases, build them manually
            [
                np.zeros((2, 2)),
                np.eye(2),
                5.0 * np.eye(2),
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                np.array([[0.0, 0.0], [0.0, 2.0]]),
                np.array([[1.0, 1.0], [-1.0, -1.0]]),
                np.array([[3.0, 0.0], [4.0, 5.0]]),
                np.eye(2) + tol * np.array([[1.0, 1.0], [-1.0, -1.0]]),
            ],
        ),
        axis=0,
    )
    M = len(mats)
    m2 = wp.array(mats, dtype=mat22, requires_grad=True, device=device)

    outcomponents = wp.zeros(2 * 2 * 2 + 2, dtype=wptype, requires_grad=True, device=device)
    Uout = wp.zeros(M, dtype=mat22, requires_grad=True, device=device)
    sigmaout = wp.zeros(M, dtype=vec2, requires_grad=True, device=device)
    Vout = wp.zeros(M, dtype=mat22, requires_grad=True, device=device)

    wp.launch(kernel, dim=M, inputs=[m2], outputs=[Uout, sigmaout, Vout, outcomponents], device=device)

    Uout_np = Uout.numpy().astype(np.float64)
    sigmaout_np = sigmaout.numpy().astype(np.float64)
    Vout_np = Vout.numpy().astype(np.float64)

    USVt_np = Uout_np @ (sigmaout_np[..., None] * np.transpose(Vout_np, axes=(0, 2, 1)))

    assert_np_equal(
        Uout_np @ np.transpose(Uout_np, axes=(0, 2, 1)), np.broadcast_to(np.eye(2), shape=(M, 2, 2)), tol=30 * tol
    )
    assert_np_equal(
        Vout_np @ np.transpose(Vout_np, axes=(0, 2, 1)), np.broadcast_to(np.eye(2), shape=(M, 2, 2)), tol=30 * tol
    )
    assert_np_equal(USVt_np, m2.numpy().astype(np.float64), tol=30 * tol)

    if dtype == np.float16:
        # Skip gradient check for float16 due to rounding errors
        return

    # Check gradients:
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    idx = 0
    for idx in range(2 * 2 + 2 + 2 * 2):
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[m2], outputs=[Uout, sigmaout, Vout, outcomponents], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
        tape.backward(out)
        m2grads = 1.0 * tape.gradients[m2].numpy()[0]

        tape.zero()

        dx = 0.001
        fdtol = 5.0e-4 if dtype == np.float64 else 2.0e-2
        for ii in range(2):
            for jj in range(2):
                m2test = 1.0 * m2.numpy()
                m2test[0, ii, jj] += dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[wp.array(m2test, dtype=mat22, device=device)],
                    outputs=[Uout, sigmaout, Vout, outcomponents],
                    device=device,
                )
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
                plusval = out.numpy()[0]

                m2test = 1.0 * m2.numpy()
                m2test[0, ii, jj] -= dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[wp.array(m2test, dtype=mat22, device=device)],
                    outputs=[Uout, sigmaout, Vout, outcomponents],
                    device=device,
                )
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
                minusval = out.numpy()[0]

                assert_np_equal((plusval - minusval) / (2 * dx), m2grads[ii, jj], tol=fdtol)


def test_qr(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 2.5e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-12,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat33 = wp._src.types.matrix(shape=(3, 3), dtype=wptype)

    def check_mat_qr(
        m3: wp.array(dtype=mat33),
        Qout: wp.array(dtype=mat33),
        Rout: wp.array(dtype=mat33),
        outcomponents: wp.array(dtype=wptype),
    ):
        Q = mat33()
        R = mat33()

        wp.qr3(m3[0], Q, R)

        Qout[0] = Q
        Rout[0] = R

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * Q[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * R[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_qr, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    m3 = wp.array(0.5 * (randvals(rng, [1, 3, 3], dtype) + np.eye(3)), dtype=mat33, requires_grad=True, device=device)

    outcomponents = wp.zeros(2 * 3 * 3, dtype=wptype, requires_grad=True, device=device)
    Qout = wp.zeros(1, dtype=mat33, requires_grad=True, device=device)
    Rout = wp.zeros(1, dtype=mat33, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[m3], outputs=[Qout, Rout, outcomponents], device=device)

    Qout_np = Qout.numpy()[0].astype(np.float64)
    Rout_np = Rout.numpy()[0].astype(np.float64)

    # check it's actually a q and an r:
    assert_np_equal(np.matmul(Qout_np.T, Qout_np), np.eye(3, dtype=np.float64), tol=tol)
    assert_np_equal(Rout_np[1, [0]], np.zeros(1, dtype=np.float64), tol=tol)
    assert_np_equal(Rout_np[2, [0, 1]], np.zeros(2, dtype=np.float64), tol=tol)

    # check it's a factorization:
    assert_np_equal(np.matmul(Qout_np, Rout_np), m3.numpy()[0].astype(np.float64), tol=30 * tol)

    if dtype == np.float16:
        # I'm not even going to bother testing the gradients for float16
        # because the rounding errors are terrible...
        return

    # check gradients:
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    idx = 0
    for idx in range(len(outcomponents)):
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[m3], outputs=[Qout, Rout, outcomponents], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
        tape.backward(out)
        m3grads = 1.0 * tape.gradients[m3].numpy()[0]

        tape.zero()

        dx = 0.0001
        fdtol = 5.0e-4 if dtype == np.float64 else 2.0e-2
        for ii in range(3):
            for jj in range(3):
                m3test = 1.0 * m3.numpy()
                m3test[0, ii, jj] += dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[wp.array(m3test, dtype=mat33, device=device)],
                    outputs=[Qout, Rout, outcomponents],
                    device=device,
                )
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
                plusval = out.numpy()[0]

                m3test = 1.0 * m3.numpy()
                m3test[0, ii, jj] -= dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[wp.array(m3test, dtype=mat33, device=device)],
                    outputs=[Qout, Rout, outcomponents],
                    device=device,
                )
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
                minusval = out.numpy()[0]

                assert_np_equal((plusval - minusval) / (2 * dx), m3grads[ii, jj], tol=fdtol)


def test_eig(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 4.0e-2,
        np.float32: 1.0e-5,
        np.float64: 1.0e-5,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec3 = wp._src.types.vector(length=3, dtype=wptype)
    mat33 = wp._src.types.matrix(shape=(3, 3), dtype=wptype)

    def check_mat_eig(
        m3: wp.array(dtype=mat33),
        Qout: wp.array(dtype=mat33),
        dout: wp.array(dtype=vec3),
        outcomponents: wp.array(dtype=wptype),
    ):
        Q = mat33()
        d = vec3()

        wp.eig3(m3[0] + wp.transpose(m3[0]), Q, d)

        Qout[0] = Q
        dout[0] = d

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * Q[i, j]
                idx = idx + 1

        for i in range(3):
            outcomponents[idx] = wptype(2) * d[i]
            idx = idx + 1

    kernel = getkernel(check_mat_eig, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    m3_np = randvals(rng, [1, 3, 3], dtype) + np.eye(3, dtype=dtype)
    m3 = wp.array(m3_np, dtype=mat33, requires_grad=True, device=device)

    outcomponents = wp.zeros(3 * 3 + 3, dtype=wptype, requires_grad=True, device=device)
    Qout = wp.zeros(1, dtype=mat33, requires_grad=True, device=device)
    dout = wp.zeros(1, dtype=vec3, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[m3], outputs=[Qout, dout, outcomponents], device=device)

    Qout_np = Qout.numpy()[0].astype(np.float64)
    dout_np = dout.numpy()[0].astype(np.float64)
    Dout_np = np.diag(dout_np)

    # check Q is orthogonal:
    assert_np_equal(np.matmul(Qout_np.T, Qout_np), np.eye(3), tol=tol)

    # check Q contains eigenvectors:
    assert_np_equal(np.matmul(Qout_np, np.matmul(Dout_np, Qout_np.T)), (m3_np[0] + m3_np[0].transpose()), tol=tol)

    if dtype == np.float16:
        # I'm not even going to bother testing the gradients for float16
        # because the rounding errors are terrible...
        return

    # check gradients:
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    idx = 0
    for idx in range(len(outcomponents)):
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[m3], outputs=[Qout, dout, outcomponents], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
        tape.backward(out)
        m3grads = 1.0 * tape.gradients[m3].numpy()[0]

        tape.zero()

        dx = 0.0001
        fdtol = 5.0e-4 if dtype == np.float64 else 2.0e-2
        for ii in range(3):
            for jj in range(3):
                m3test = 1.0 * m3.numpy()
                m3test[0, ii, jj] += dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[wp.array(m3test, dtype=mat33, device=device)],
                    outputs=[Qout, dout, outcomponents],
                    device=device,
                )
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
                plusval = out.numpy()[0]

                m3test = 1.0 * m3.numpy()
                m3test[0, ii, jj] -= dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[wp.array(m3test, dtype=mat33, device=device)],
                    outputs=[Qout, dout, outcomponents],
                    device=device,
                )
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
                minusval = out.numpy()[0]

                assert_np_equal((plusval - minusval) / (2 * dx), m3grads[ii, jj], tol=fdtol)


def test_skew(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec3 = wp._src.types.vector(length=3, dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_skew(
        v3: wp.array(dtype=vec3),
        outcomponents: wp.array(dtype=wptype),
    ):
        m3result = wp.skew(v3[0])

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * m3result[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_skew, suffix=dtype.__name__)

    if register_kernels:
        return

    v3 = wp.array(randvals(rng, [1, 3], dtype), dtype=vec3, requires_grad=True, device=device)

    outcomponents = wp.zeros(3 * 3, dtype=wptype, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[v3], outputs=[outcomponents], device=device)

    # make sure it gives you a cross product matrix:
    crossprodmat = outcomponents.numpy().reshape(3, 3)
    assert_np_equal(
        np.matmul(crossprodmat, np.array([1, 0, 0])).reshape(-1),
        2 * np.cross(v3.numpy()[0], np.array([1, 0, 0])),
        tol=tol,
    )
    assert_np_equal(
        np.matmul(crossprodmat, np.array([0, 1, 0])).reshape(-1),
        2 * np.cross(v3.numpy()[0], np.array([0, 1, 0])),
        tol=tol,
    )
    assert_np_equal(
        np.matmul(crossprodmat, np.array([0, 0, 1])).reshape(-1),
        2 * np.cross(v3.numpy()[0], np.array([0, 0, 1])),
        tol=tol,
    )

    # check it another way:
    x0 = v3.numpy()[0, 0]
    x1 = v3.numpy()[0, 1]
    x2 = v3.numpy()[0, 2]
    crossprodmat_expected = np.array(
        [
            [0, -x2, x1],
            [x2, 0, -x0],
            [-x1, x0, 0],
        ],
        dtype=dtype,
    )
    assert_np_equal(crossprodmat, 2 * crossprodmat_expected, tol=tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

        for i in range(3):
            for j in range(3):
                tape = wp.Tape()
                with tape:
                    wp.launch(kernel, dim=1, inputs=[v3], outputs=[outcomponents], device=device)
                    wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
                tape.backward(loss=out)
                if i == j:
                    assert_np_equal(tape.gradients[v3].numpy()[0], np.zeros(3))
                elif [i, j] == [0, 1]:
                    assert_np_equal(tape.gradients[v3].numpy()[0], np.array([0, 0, -2]))
                elif [i, j] == [1, 0]:
                    assert_np_equal(tape.gradients[v3].numpy()[0], np.array([0, 0, 2]))
                elif [i, j] == [0, 2]:
                    assert_np_equal(tape.gradients[v3].numpy()[0], np.array([0, 2, 0]))
                elif [i, j] == [2, 0]:
                    assert_np_equal(tape.gradients[v3].numpy()[0], np.array([0, -2, 0]))
                elif [i, j] == [1, 2]:
                    assert_np_equal(tape.gradients[v3].numpy()[0], np.array([-2, 0, 0]))
                elif [i, j] == [2, 1]:
                    assert_np_equal(tape.gradients[v3].numpy()[0], np.array([2, 0, 0]))
                tape.zero()

                idx = idx + 1


def test_transform_point(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec3 = wp._src.types.vector(length=3, dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_transform_point(
        v3: wp.array(dtype=vec3),
        m4: wp.array(dtype=mat44),
        outcomponents: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        presult = wptype(2) * wp.transform_point(m4[0], v3[0])

        outcomponents[0] = presult[0]
        outcomponents[1] = presult[1]
        outcomponents[2] = presult[2]

    kernel = getkernel(check_mat_transform_point, suffix=dtype.__name__)

    if register_kernels:
        return

    v3 = wp.array(randvals(rng, [1, 3], dtype), dtype=vec3, requires_grad=True, device=device)
    m4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)

    outcomponents = wp.zeros(3, dtype=wptype, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[v3, m4], outputs=[outcomponents], device=device)

    v3homog = np.ones(4, dtype=dtype)
    v3homog[:3] = v3.numpy()[0]
    assert_np_equal(outcomponents.numpy(), 2 * np.matmul(m4.numpy()[0], v3homog)[:3], tol=10 * tol)

    if dtype in np_float_types:
        for j in range(3):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[v3, m4], outputs=[outcomponents], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, j], outputs=[out], device=device)
            tape.backward(loss=out)

            assert_np_equal(2 * m4.numpy()[0, j, :3], tape.gradients[v3].numpy(), tol=tol)
            expected = np.zeros((4, 4), dtype=dtype)
            expected[j, :3] = 2 * v3.numpy()
            expected[j, 3] = 2
            assert_np_equal(tape.gradients[m4].numpy(), expected, tol=tol)

            tape.zero()


def test_transform_vector(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec3 = wp._src.types.vector(length=3, dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_transform_vector(
        v3: wp.array(dtype=vec3),
        m4: wp.array(dtype=mat44),
        outcomponents: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        presult = wptype(2) * wp.transform_vector(m4[0], v3[0])

        outcomponents[0] = presult[0]
        outcomponents[1] = presult[1]
        outcomponents[2] = presult[2]

    kernel = getkernel(check_mat_transform_vector, suffix=dtype.__name__)

    if register_kernels:
        return

    v3 = wp.array(randvals(rng, [1, 3], dtype), dtype=vec3, requires_grad=True, device=device)
    m4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)

    outcomponents = wp.zeros(3, dtype=wptype, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[v3, m4], outputs=[outcomponents], device=device)

    v3homog = np.zeros(4, dtype=dtype)
    v3homog[:3] = v3.numpy()[0]
    assert_np_equal(outcomponents.numpy(), 2 * np.matmul(m4.numpy()[0], v3homog)[:3], tol=10 * tol)

    if dtype in np_float_types:
        for j in range(3):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[v3, m4], outputs=[outcomponents], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, j], outputs=[out], device=device)
            tape.backward(loss=out)

            assert_np_equal(2 * m4.numpy()[0, j, :3], tape.gradients[v3].numpy(), tol=tol)
            expected = np.zeros((4, 4), dtype=dtype)
            expected[j, :3] = 2 * v3.numpy()
            assert_np_equal(tape.gradients[m4].numpy(), expected, tol=tol)

            tape.zero()


@wp.kernel
def test_matrix_mutation(expected: wp._src.types.matrix(shape=(10, 3), dtype=float)):
    m = wp.matrix(shape=(10, 3), dtype=float)

    # test direct element indexing
    m[0, 0] = 1.0
    m[0, 1] = 2.0
    m[0, 2] = 3.0

    # The nested indexing (matrix->vector->scalar) below does not
    # currently modify m because m[0] returns row vector by
    # value rather than reference, this is different from NumPy
    # which always returns by ref. Not clear how we can support
    # this as well as auto-diff.

    # m[0][1] = 2.0
    # m[0][2] = 3.0

    # test setting rows
    for i in range(1, 10):
        m[i] = m[i - 1] + wp.vec3(1.0, 2.0, 3.0)

    wp.expect_eq(m, expected)


Mat23 = wp.mat((2, 3), dtype=wp.float16)


@wp.kernel(module="unique")
def matrix_len_kernel(
    m1: wp.mat22, m2: wp.mat((3, 3), float), m3: wp.mat((Any, Any), float), m4: Mat23, out: wp.array(dtype=int)
):
    length = wp.static(len(m1))
    wp.expect_eq(len(m1), 2)
    out[0] = len(m1)

    length = len(m2)
    wp.expect_eq(wp.static(len(m2)), 3)
    out[1] = len(m2)

    length = len(m3)
    wp.expect_eq(len(m3), 4)
    out[2] = wp.static(len(m3))

    length = wp.static(len(m4))
    wp.expect_eq(wp.static(len(m4)), 2)
    out[3] = wp.static(len(m4))

    foo = wp.mat22()
    length = len(foo)
    wp.expect_eq(len(foo), 2)
    out[4] = len(foo)


def test_matrix_len(test, device):
    m1 = wp.mat22()
    m2 = wp.mat33()
    m3 = wp.mat44()
    m4 = Mat23()
    out = wp.empty(5, dtype=int, device=device)
    wp.launch(matrix_len_kernel, dim=(1,), inputs=(m1, m2, m3, m4), outputs=(out,), device=device)

    test.assertEqual(out.numpy()[0], 2)
    test.assertEqual(out.numpy()[1], 3)
    test.assertEqual(out.numpy()[2], 4)
    test.assertEqual(out.numpy()[3], 2)
    test.assertEqual(out.numpy()[4], 2)

    test.assertEqual(len(m1), 2)
    test.assertEqual(len(m2), 3)
    test.assertEqual(len(m3), 4)
    test.assertEqual(len(m4), 2)


@wp.kernel
def mat_extract_element(x: wp.array(dtype=wp.mat22), y: wp.array(dtype=float)):
    tid = wp.tid()

    a = x[tid]
    b = a[0, 0] + 2.0 * a[0, 1] + 3.0 * a[1, 0] + 4.0 * a[1, 1]
    y[tid] = b


@wp.kernel
def mat_extract_row(x: wp.array(dtype=wp.mat22), y: wp.array(dtype=wp.vec2)):
    tid = wp.tid()

    a = x[tid]
    b = a[0] + 2.0 * a[1]
    y[tid] = b


def test_mat_extract(test, device):
    # matrix element
    x = wp.ones(1, dtype=wp.mat22, requires_grad=True, device=device)
    y = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mat_extract_element, 1, inputs=[x], outputs=[y], device=device)

    y.grad = wp.ones_like(y)
    tape.backward()

    assert_np_equal(y.numpy(), np.array([10.0], dtype=float))
    assert_np_equal(x.grad.numpy(), np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=float))

    # matrix row
    x = wp.ones(1, dtype=wp.mat22, requires_grad=True, device=device)
    y = wp.zeros(1, dtype=wp.vec2, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mat_extract_row, 1, inputs=[x], outputs=[y], device=device)

    y.grad = wp.ones_like(y)
    tape.backward()

    assert_np_equal(y.numpy(), np.array([[3.0, 3.0]], dtype=float))
    assert_np_equal(x.grad.numpy(), np.array([[[1.0, 1.0], [2.0, 2.0]]], dtype=float))


@wp.kernel
def mat_assign_element(x: wp.array(dtype=float), y: wp.array(dtype=wp.mat22)):
    i = wp.tid()

    a = wp.mat22()
    a[0, 0] = 1.0 * x[i]
    a[0, 1] = 2.0 * x[i]
    a[1, 0] = 3.0 * x[i]
    a[1, 1] = 4.0 * x[i]

    y[i] = a


@wp.kernel
def mat_assign_row(x: wp.array(dtype=wp.vec2), y: wp.array(dtype=wp.mat22)):
    i = wp.tid()

    a = wp.mat22()
    a[0] = 1.0 * x[i]
    a[1] = 2.0 * x[i]

    y[i] = a


def test_mat_assign(test, device):
    # matrix element
    x = wp.ones(1, dtype=float, requires_grad=True, device=device)
    y = wp.zeros(1, dtype=wp.mat22, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mat_assign_element, 1, inputs=[x], outputs=[y], device=device)

    y.grad = wp.ones_like(y)
    tape.backward()

    assert_np_equal(y.numpy(), np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=float))
    assert_np_equal(x.grad.numpy(), np.array([10.0], dtype=float))

    # matrix row
    x = wp.ones(1, dtype=wp.vec2, requires_grad=True, device=device)
    y = wp.zeros(1, dtype=wp.mat22, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mat_assign_row, 1, inputs=[x], outputs=[y], device=device)

    y.grad = wp.ones_like(y)
    tape.backward()

    assert_np_equal(y.numpy(), np.array([[[1.0, 1.0], [2.0, 2.0]]], dtype=float))
    assert_np_equal(x.grad.numpy(), np.array([[3.0, 3.0]], dtype=float))


@wp.kernel
def mat_array_extract_element(x: wp.array2d(dtype=wp.mat22), y: wp.array2d(dtype=float)):
    i, j = wp.tid()
    a = x[i, j][0, 0]
    b = x[i, j][0, 1]
    c = x[i, j][1, 0]
    d = x[i, j][1, 1]
    y[i, j] = 1.0 * a + 2.0 * b + 3.0 * c + 4.0 * d


@wp.kernel
def mat_array_extract_row(x: wp.array2d(dtype=wp.mat22), y: wp.array2d(dtype=wp.vec2)):
    i, j = wp.tid()
    a = x[i, j][0]
    b = x[i, j][1]
    y[i, j] = 1.0 * a + 2.0 * b


def test_mat_array_extract(test, device):
    # matrix element
    x = wp.ones((1, 1), dtype=wp.mat22, requires_grad=True, device=device)
    y = wp.zeros((1, 1), dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mat_array_extract_element, (1, 1), inputs=[x], outputs=[y], device=device)

    y.grad = wp.ones_like(y)
    tape.backward()

    assert_np_equal(y.numpy(), np.array([[10.0]], dtype=float))
    assert_np_equal(x.grad.numpy(), np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=float))

    # matrix row
    x = wp.ones((1, 1), dtype=wp.mat22, requires_grad=True, device=device)
    y = wp.zeros((1, 1), dtype=wp.vec2, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mat_array_extract_row, (1, 1), inputs=[x], outputs=[y], device=device)

    y.grad = wp.ones_like(y)
    tape.backward()

    assert_np_equal(y.numpy(), np.array([[[3.0, 3.0]]], dtype=float))
    assert_np_equal(x.grad.numpy(), np.array([[[[1.0, 1.0], [2.0, 2.0]]]], dtype=float))


""" TODO: gradient propagation for in-place array assignment
@wp.kernel
def mat_array_assign_element(x: wp.array2d(dtype=float), y: wp.array2d(dtype=wp.mat22)):
    i, j = wp.tid()

    y[i, j][0, 0] = 1.0 * x[i, j]
    y[i, j][0, 1] = 2.0 * x[i, j]
    y[i, j][1, 0] = 3.0 * x[i, j]
    y[i, j][1, 1] = 4.0 * x[i, j]


@wp.kernel
def mat_array_assign_row(x: wp.array2d(dtype=wp.vec3), y: wp.array2d(dtype=wp.mat(shape=(2, 3), dtype=float))):
    i, j = wp.tid()

    y[i, j][0] = 1.0 * x[i, j]
    y[i, j][1] = 2.0 * x[i, j]


def test_mat_array_assign(test, device):
    # matrix element
    x = wp.ones((1, 1), dtype=float, requires_grad=True, device=device)
    y = wp.zeros((1, 1), dtype=wp.mat22, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mat_array_assign_element, (1, 1), inputs=[x], outputs=[y], device=device)

    y.grad = wp.ones_like(y)
    tape.backward()

    assert_np_equal(y.numpy(), np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=float))
    assert_np_equal(x.grad.numpy(), np.array([[10.0]], dtype=float))

    # matrix row
    x = wp.ones((1, 1), dtype=wp.vec3, requires_grad=True, device=device)
    y = wp.zeros((1, 1), dtype=wp.mat(shape=(2, 3), dtype=float), requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mat_array_assign_row, (1, 1), inputs=[x], outputs=[y], device=device)

    y.grad = wp.ones_like(y)
    tape.backward()

    assert_np_equal(y.numpy(), np.array([[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]], dtype=float))
    assert_np_equal(x.grad.numpy(), np.array([[[3.0, 3.0, 3.0]]], dtype=float))
"""


@wp.kernel
def mat_add_inplace_element(x: wp.array(dtype=wp.mat22), y: wp.array(dtype=wp.mat22)):
    i = wp.tid()

    a = wp.mat22()
    b = x[i]

    a[0, 0] += 1.0 * b[0, 0]
    a[0, 1] += 2.0 * b[0, 1]
    a[1, 0] += 3.0 * b[1, 0]
    a[1, 1] += 4.0 * b[1, 1]

    y[i] = a


@wp.kernel
def mat_add_inplace_row(x: wp.array(dtype=wp.mat22), y: wp.array(dtype=wp.mat22)):
    i = wp.tid()

    a = wp.mat22()
    b = x[i]

    a[0] += 1.0 * b[0]
    a[1] += 2.0 * b[1]

    y[i] = a


def test_mat_add_inplace(test, device):
    x = wp.ones(1, dtype=wp.mat22, requires_grad=True, device=device)
    y = wp.zeros(1, dtype=wp.mat22, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mat_add_inplace_element, 1, inputs=[x], outputs=[y], device=device)

    y.grad = wp.ones_like(y)
    tape.backward()

    assert_np_equal(y.numpy(), np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=float))
    assert_np_equal(x.grad.numpy(), np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=float))

    x = wp.ones(1, dtype=wp.mat22, requires_grad=True, device=device)
    y = wp.zeros(1, dtype=wp.mat22, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mat_add_inplace_row, 1, inputs=[x], outputs=[y], device=device)

    y.grad = wp.ones_like(y)
    tape.backward()

    assert_np_equal(y.numpy(), np.array([[[1.0, 1.0], [2.0, 2.0]]], dtype=float))
    assert_np_equal(x.grad.numpy(), np.array([[[1.0, 1.0], [2.0, 2.0]]], dtype=float))


@wp.kernel
def mat_sub_inplace_element(x: wp.array(dtype=wp.mat22), y: wp.array(dtype=wp.mat22)):
    i = wp.tid()

    a = wp.mat22()
    b = x[i]

    a[0, 0] -= 1.0 * b[0, 0]
    a[0, 1] -= 2.0 * b[0, 1]
    a[1, 0] -= 3.0 * b[1, 0]
    a[1, 1] -= 4.0 * b[1, 1]

    y[i] = a


@wp.kernel
def mat_sub_inplace_row(x: wp.array(dtype=wp.mat22), y: wp.array(dtype=wp.mat22)):
    i = wp.tid()

    a = wp.mat22()
    b = x[i]

    a[0] -= 1.0 * b[0]
    a[1] -= 2.0 * b[1]

    y[i] = a


def test_mat_sub_inplace(test, device):
    x = wp.ones(1, dtype=wp.mat22, requires_grad=True, device=device)
    y = wp.zeros(1, dtype=wp.mat22, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mat_sub_inplace_element, 1, inputs=[x], outputs=[y], device=device)

    y.grad = wp.ones_like(y)
    tape.backward()

    assert_np_equal(y.numpy(), np.array([[[-1.0, -2.0], [-3.0, -4.0]]], dtype=float))
    assert_np_equal(x.grad.numpy(), np.array([[[-1.0, -2.0], [-3.0, -4.0]]], dtype=float))

    x = wp.ones(1, dtype=wp.mat22, requires_grad=True, device=device)
    y = wp.zeros(1, dtype=wp.mat22, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mat_sub_inplace_row, 1, inputs=[x], outputs=[y], device=device)

    y.grad = wp.ones_like(y)
    tape.backward()

    assert_np_equal(y.numpy(), np.array([[[-1.0, -1.0], [-2.0, -2.0]]], dtype=float))
    assert_np_equal(x.grad.numpy(), np.array([[[-1.0, -1.0], [-2.0, -2.0]]], dtype=float))


@wp.kernel
def mat_array_add_inplace(x: wp.array(dtype=wp.mat22), y: wp.array(dtype=wp.mat22)):
    i = wp.tid()

    y[i] += x[i]


def test_mat_array_add_inplace(test, device):
    x = wp.ones(1, dtype=wp.mat22, requires_grad=True, device=device)
    y = wp.zeros(1, dtype=wp.mat22, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mat_array_add_inplace, 1, inputs=[x], outputs=[y], device=device)

    y.grad = wp.ones_like(y)
    tape.backward()

    assert_np_equal(y.numpy(), np.array([[[1.0, 1.0], [1.0, 1.0]]], dtype=float))
    assert_np_equal(x.grad.numpy(), np.array([[[1.0, 1.0], [1.0, 1.0]]], dtype=float))


@wp.kernel
def mat_array_sub_inplace(x: wp.array(dtype=wp.mat22), y: wp.array(dtype=wp.mat22)):
    i = wp.tid()

    y[i] -= x[i]


def test_mat_array_sub_inplace(test, device):
    x = wp.ones(1, dtype=wp.mat22, requires_grad=True, device=device)
    y = wp.zeros(1, dtype=wp.mat22, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mat_array_sub_inplace, 1, inputs=[x], outputs=[y], device=device)

    y.grad = wp.ones_like(y)
    tape.backward()

    assert_np_equal(y.numpy(), np.array([[[-1.0, -1.0], [-1.0, -1.0]]], dtype=float))
    assert_np_equal(x.grad.numpy(), np.array([[[-1.0, -1.0], [-1.0, -1.0]]], dtype=float))


@wp.kernel
def scalar_mat_div(x: wp.array(dtype=wp.mat22), y: wp.array(dtype=wp.mat22)):
    i = wp.tid()
    y[i] = 1.0 / x[i]


def test_scalar_mat_div(test, device):
    x = wp.array((wp.mat22(1.0, 2.0, 4.0, 8.0),), dtype=wp.mat22, requires_grad=True, device=device)
    y = wp.ones(1, dtype=wp.mat22, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(scalar_mat_div, 1, inputs=(x,), outputs=(y,), device=device)

    y.grad = wp.ones_like(y)
    tape.backward()

    assert_np_equal(y.numpy(), np.array((((1.0, 0.5), (0.25, 0.125)),), dtype=float))
    assert_np_equal(x.grad.numpy(), np.array((((-1.0, -0.25), (-0.0625, -0.015625)),), dtype=float))


def test_mat_from_rows_indexing_assign(test, device):
    @wp.func
    def fn():
        m = wp.matrix_from_rows(
            wp.vec2(1.0, 2.0),
            wp.vec2(3.0, 4.0),
            wp.vec2(5.0, 6.0),
        )

        m[0] = wp.vec2(123.0, 234.0)
        m[1] *= 2.0

        wp.expect_eq(m[0], wp.vec2(123.0, 234.0))
        wp.expect_eq(m[1], wp.vec2(6.0, 8.0))
        wp.expect_eq(m[2], wp.vec2(5.0, 6.0))

        m[-1] = wp.vec2(123.0, 234.0)
        m[-2] *= 2.0

        wp.expect_eq(m[-1], wp.vec2(123.0, 234.0))
        wp.expect_eq(m[-2], wp.vec2(12.0, 16.0))
        wp.expect_eq(m[-3], wp.vec2(123.0, 234.0))

        m[0, 0] = 345.0
        m[1, 0] *= 2.0

        wp.expect_eq(m[0, 0], 345.0)
        wp.expect_eq(m[0, 1], 234.0)
        wp.expect_eq(m[1, 0], 24.0)
        wp.expect_eq(m[1, 1], 16.0)
        wp.expect_eq(m[2, 0], 123.0)
        wp.expect_eq(m[2, 1], 234.0)

        m[-1, -1] = 345.0
        m[-2, -1] *= 2.0

        wp.expect_eq(m[-1, -1], 345.0)
        wp.expect_eq(m[-1, -2], 123.0)
        wp.expect_eq(m[-2, -1], 32.0)
        wp.expect_eq(m[-2, -2], 24.0)
        wp.expect_eq(m[-3, -1], 234.0)
        wp.expect_eq(m[-3, -2], 345.0)

        m[0, 1] = 456.0
        m[1, 1] *= 2.0

        wp.expect_eq(m[0][0], 345.0)
        wp.expect_eq(m[0][1], 456.0)
        wp.expect_eq(m[1][0], 24.0)
        wp.expect_eq(m[1][1], 64.0)
        wp.expect_eq(m[2][0], 123.0)
        wp.expect_eq(m[2][1], 345.0)

        m[-1, -2] = 456.0
        m[-2, -2] *= 2.0

        wp.expect_eq(m[-1][-1], 345.0)
        wp.expect_eq(m[-1][-2], 456.0)
        wp.expect_eq(m[-2][-1], 64.0)
        wp.expect_eq(m[-2][-2], 48.0)
        wp.expect_eq(m[-3][-1], 456.0)
        wp.expect_eq(m[-3][-2], 345.0)

    @wp.kernel(module="unique")
    def kernel():
        fn()

    wp.launch(kernel, 1, device=device)
    wp.synchronize()
    fn()


def test_mat_from_cols_indexing_assign(test, device):
    @wp.func
    def fn():
        m = wp.matrix_from_cols(
            wp.vec2(1.0, 2.0),
            wp.vec2(3.0, 4.0),
            wp.vec2(5.0, 6.0),
        )

        m[0] = wp.vec3(123.0, 234.0, 345.0)
        m[1] *= 2.0

        wp.expect_eq(m[0], wp.vec3(123.0, 234.0, 345.0))
        wp.expect_eq(m[1], wp.vec3(4.0, 8.0, 12.0))

        m[-1] = wp.vec3(123.0, 234.0, 345.0)
        m[-2] *= 2.0

        wp.expect_eq(m[-1], wp.vec3(123.0, 234.0, 345.0))
        wp.expect_eq(m[-2], wp.vec3(246.0, 468.0, 690.0))

        m[0, 0] = 456.0
        m[1, 0] *= 2.0

        wp.expect_eq(m[0, 0], 456.0)
        wp.expect_eq(m[0, 1], 468.0)
        wp.expect_eq(m[0, 2], 690.0)
        wp.expect_eq(m[1, 0], 246.0)
        wp.expect_eq(m[1, 1], 234.0)
        wp.expect_eq(m[1, 2], 345.0)

        m[-1, -1] = 456.0
        m[-2, -1] *= 2.0

        wp.expect_eq(m[-1, -1], 456.0)
        wp.expect_eq(m[-1, -2], 234.0)
        wp.expect_eq(m[-1, -3], 246.0)
        wp.expect_eq(m[-2, -1], 1380.0)
        wp.expect_eq(m[-2, -2], 468.0)
        wp.expect_eq(m[-2, -3], 456.0)

        m[0, 1] = 567.0
        m[1, 1] *= 2.0

        wp.expect_eq(m[0][0], 456.0)
        wp.expect_eq(m[0][1], 567.0)
        wp.expect_eq(m[0][2], 1380.0)
        wp.expect_eq(m[1][0], 246.0)
        wp.expect_eq(m[1][1], 468.0)
        wp.expect_eq(m[1][2], 456.0)

        m[-1, -2] = 567.0
        m[-2, -2] *= 2.0

        wp.expect_eq(m[-1][-1], 456.0)
        wp.expect_eq(m[-1][-2], 567.0)
        wp.expect_eq(m[-1][-3], 246.0)
        wp.expect_eq(m[-2][-1], 1380.0)
        wp.expect_eq(m[-2][-2], 1134.0)
        wp.expect_eq(m[-2][-3], 456.0)

    @wp.kernel(module="unique")
    def kernel():
        fn()

    wp.launch(kernel, 1, device=device)
    wp.synchronize()
    fn()


def test_mat_from_rows_slicing_assign(test, device):
    mat00 = wp.mat((0, 0), float)
    vec1 = wp.vec(1, float)
    vec2 = wp.vec(2, float)
    vec3 = wp.vec(3, float)
    vec4 = wp.vec(4, float)

    @wp.func
    def fn():
        m = wp.matrix_from_rows(
            vec4(1.0, 2.0, 3.0, 4.0),
            vec4(5.0, 6.0, 7.0, 8.0),
            vec4(9.0, 10.0, 11.0, 12.0),
            vec4(13.0, 14.0, 15.0, 16.0),
        )

        wp.expect_eq(
            m[:]
            == wp.matrix_from_rows(
                vec4(1.0, 2.0, 3.0, 4.0),
                vec4(5.0, 6.0, 7.0, 8.0),
                vec4(9.0, 10.0, 11.0, 12.0),
                vec4(13.0, 14.0, 15.0, 16.0),
            ),
            True,
        )
        wp.expect_eq(
            m[-123:123]
            == wp.matrix_from_rows(
                vec4(1.0, 2.0, 3.0, 4.0),
                vec4(5.0, 6.0, 7.0, 8.0),
                vec4(9.0, 10.0, 11.0, 12.0),
                vec4(13.0, 14.0, 15.0, 16.0),
            ),
            True,
        )
        wp.expect_eq(m[123:] == mat00(), True)
        wp.expect_eq(m[:-123] == mat00(), True)
        wp.expect_eq(
            m[::123]
            == wp.matrix_from_rows(
                vec4(1.0, 2.0, 3.0, 4.0),
            ),
            True,
        )

        wp.expect_eq(
            m[1:]
            == wp.matrix_from_rows(
                vec4(5.0, 6.0, 7.0, 8.0),
                vec4(9.0, 10.0, 11.0, 12.0),
                vec4(13.0, 14.0, 15.0, 16.0),
            ),
            True,
        )
        wp.expect_eq(
            m[-2:]
            == wp.matrix_from_rows(
                vec4(9.0, 10.0, 11.0, 12.0),
                vec4(13.0, 14.0, 15.0, 16.0),
            ),
            True,
        )
        wp.expect_eq(
            m[:2]
            == wp.matrix_from_rows(
                vec4(1.0, 2.0, 3.0, 4.0),
                vec4(5.0, 6.0, 7.0, 8.0),
            ),
            True,
        )
        wp.expect_eq(
            m[:-1]
            == wp.matrix_from_rows(
                vec4(1.0, 2.0, 3.0, 4.0),
                vec4(5.0, 6.0, 7.0, 8.0),
                vec4(9.0, 10.0, 11.0, 12.0),
            ),
            True,
        )
        wp.expect_eq(
            m[::2]
            == wp.matrix_from_rows(
                vec4(1.0, 2.0, 3.0, 4.0),
                vec4(9.0, 10.0, 11.0, 12.0),
            ),
            True,
        )
        wp.expect_eq(
            m[1::2]
            == wp.matrix_from_rows(
                vec4(5.0, 6.0, 7.0, 8.0),
                vec4(13.0, 14.0, 15.0, 16.0),
            ),
            True,
        )
        wp.expect_eq(
            m[::-1]
            == wp.matrix_from_rows(
                vec4(13.0, 14.0, 15.0, 16.0),
                vec4(9.0, 10.0, 11.0, 12.0),
                vec4(5.0, 6.0, 7.0, 8.0),
                vec4(1.0, 2.0, 3.0, 4.0),
            ),
            True,
        )
        wp.expect_eq(
            m[::-2]
            == wp.matrix_from_rows(
                vec4(13.0, 14.0, 15.0, 16.0),
                vec4(5.0, 6.0, 7.0, 8.0),
            ),
            True,
        )
        wp.expect_eq(
            m[1::-2]
            == wp.matrix_from_rows(
                vec4(5.0, 6.0, 7.0, 8.0),
            ),
            True,
        )

        wp.expect_eq(
            m[:, :]
            == wp.matrix_from_rows(
                vec4(1.0, 2.0, 3.0, 4.0),
                vec4(5.0, 6.0, 7.0, 8.0),
                vec4(9.0, 10.0, 11.0, 12.0),
                vec4(13.0, 14.0, 15.0, 16.0),
            ),
            True,
        )
        wp.expect_eq(
            m[:, 2:]
            == wp.matrix_from_rows(
                vec2(3.0, 4.0),
                vec2(7.0, 8.0),
                vec2(11.0, 12.0),
                vec2(15.0, 16.0),
            ),
            True,
        )
        wp.expect_eq(
            m[1:, 2:]
            == wp.matrix_from_rows(
                vec2(7.0, 8.0),
                vec2(11.0, 12.0),
                vec2(15.0, 16.0),
            ),
            True,
        )
        wp.expect_eq(
            m[-2:, 2:]
            == wp.matrix_from_rows(
                vec2(11.0, 12.0),
                vec2(15.0, 16.0),
            ),
            True,
        )
        wp.expect_eq(
            m[2:, -2:]
            == wp.matrix_from_rows(
                vec2(11.0, 12.0),
                vec2(15.0, 16.0),
            ),
            True,
        )
        wp.expect_eq(
            m[1:, :2]
            == wp.matrix_from_rows(
                vec2(5.0, 6.0),
                vec2(9.0, 10.0),
                vec2(13.0, 14.0),
            ),
            True,
        )
        wp.expect_eq(
            m[:1, 2:]
            == wp.matrix_from_rows(
                vec2(3.0, 4.0),
            ),
            True,
        )
        wp.expect_eq(
            m[::-1, :1]
            == wp.matrix_from_rows(
                vec1(13.0),
                vec1(9.0),
                vec1(5.0),
                vec1(1.0),
            ),
            True,
        )
        wp.expect_eq(
            m[:1, ::-1]
            == wp.matrix_from_rows(
                vec4(4.0, 3.0, 2.0, 1.0),
            ),
            True,
        )
        wp.expect_eq(
            m[:1:-1, 2::-1]
            == wp.matrix_from_rows(
                vec3(15.0, 14.0, 13.0),
                vec3(11.0, 10.0, 9.0),
            ),
            True,
        )

        wp.expect_eq(m[:2, 0] == vec2(1.0, 5.0), True)
        wp.expect_eq(m[2:, 1] == vec2(10.0, 14.0), True)
        wp.expect_eq(m[0, :3] == vec3(1.0, 2.0, 3.0), True)
        wp.expect_eq(m[1, 1:] == vec3(6.0, 7.0, 8.0), True)

        m[1:] = wp.matrix_from_rows(
            vec4(17.0, 18.0, 19.0, 20.0),
            vec4(21.0, 22.0, 23.0, 24.0),
            vec4(25.0, 26.0, 27.0, 28.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(1.0, 2.0, 3.0, 4.0),
                vec4(17.0, 18.0, 19.0, 20.0),
                vec4(21.0, 22.0, 23.0, 24.0),
                vec4(25.0, 26.0, 27.0, 28.0),
            ),
            True,
        )

        m[-2:] = wp.matrix_from_rows(
            vec4(29.0, 30.0, 31.0, 32.0),
            vec4(33.0, 34.0, 35.0, 36.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(1.0, 2.0, 3.0, 4.0),
                vec4(17.0, 18.0, 19.0, 20.0),
                vec4(29.0, 30.0, 31.0, 32.0),
                vec4(33.0, 34.0, 35.0, 36.0),
            ),
            True,
        )

        m[:2] = wp.matrix_from_rows(
            vec4(37.0, 38.0, 39.0, 40.0),
            vec4(41.0, 42.0, 43.0, 44.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(37.0, 38.0, 39.0, 40.0),
                vec4(41.0, 42.0, 43.0, 44.0),
                vec4(29.0, 30.0, 31.0, 32.0),
                vec4(33.0, 34.0, 35.0, 36.0),
            ),
            True,
        )

        m[:-1] = wp.matrix_from_rows(
            vec4(45.0, 46.0, 47.0, 48.0),
            vec4(49.0, 50.0, 51.0, 52.0),
            vec4(53.0, 54.0, 55.0, 56.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(45.0, 46.0, 47.0, 48.0),
                vec4(49.0, 50.0, 51.0, 52.0),
                vec4(53.0, 54.0, 55.0, 56.0),
                vec4(33.0, 34.0, 35.0, 36.0),
            ),
            True,
        )

        m[::2] = wp.matrix_from_rows(
            vec4(57.0, 58.0, 59.0, 60.0),
            vec4(61.0, 62.0, 63.0, 64.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(57.0, 58.0, 59.0, 60.0),
                vec4(49.0, 50.0, 51.0, 52.0),
                vec4(61.0, 62.0, 63.0, 64.0),
                vec4(33.0, 34.0, 35.0, 36.0),
            ),
            True,
        )

        m[1::2] = wp.matrix_from_rows(
            vec4(65.0, 66.0, 67.0, 68.0),
            vec4(69.0, 70.0, 71.0, 72.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(57.0, 58.0, 59.0, 60.0),
                vec4(65.0, 66.0, 67.0, 68.0),
                vec4(61.0, 62.0, 63.0, 64.0),
                vec4(69.0, 70.0, 71.0, 72.0),
            ),
            True,
        )

        m[::-1] = wp.matrix_from_rows(
            vec4(73.0, 74.0, 75.0, 76.0),
            vec4(77.0, 78.0, 79.0, 80.0),
            vec4(81.0, 82.0, 83.0, 84.0),
            vec4(85.0, 86.0, 87.0, 88.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(85.0, 86.0, 87.0, 88.0),
                vec4(81.0, 82.0, 83.0, 84.0),
                vec4(77.0, 78.0, 79.0, 80.0),
                vec4(73.0, 74.0, 75.0, 76.0),
            ),
            True,
        )

        m[::-2] = wp.matrix_from_rows(
            vec4(89.0, 90.0, 91.0, 92.0),
            vec4(93.0, 94.0, 95.0, 96.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(85.0, 86.0, 87.0, 88.0),
                vec4(93.0, 94.0, 95.0, 96.0),
                vec4(77.0, 78.0, 79.0, 80.0),
                vec4(89.0, 90.0, 91.0, 92.0),
            ),
            True,
        )

        m[1::-2] = wp.matrix_from_rows(
            vec4(97.0, 98.0, 99.0, 100.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(85.0, 86.0, 87.0, 88.0),
                vec4(97.0, 98.0, 99.0, 100.0),
                vec4(77.0, 78.0, 79.0, 80.0),
                vec4(89.0, 90.0, 91.0, 92.0),
            ),
            True,
        )

        m[:, :] = wp.matrix_from_rows(
            vec4(101.0, 102.0, 103.0, 104.0),
            vec4(105.0, 106.0, 107.0, 108.0),
            vec4(109.0, 110.0, 111.0, 112.0),
            vec4(113.0, 114.0, 115.0, 116.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(101.0, 102.0, 103.0, 104.0),
                vec4(105.0, 106.0, 107.0, 108.0),
                vec4(109.0, 110.0, 111.0, 112.0),
                vec4(113.0, 114.0, 115.0, 116.0),
            ),
            True,
        )

        m[:, 2:] = wp.matrix_from_rows(
            vec2(117.0, 118.0),
            vec2(119.0, 120.0),
            vec2(121.0, 122.0),
            vec2(123.0, 124.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(101.0, 102.0, 117.0, 118.0),
                vec4(105.0, 106.0, 119.0, 120.0),
                vec4(109.0, 110.0, 121.0, 122.0),
                vec4(113.0, 114.0, 123.0, 124.0),
            ),
            True,
        )

        m[1:, 2:] = wp.matrix_from_rows(
            vec2(125.0, 126.0),
            vec2(127.0, 128.0),
            vec2(129.0, 130.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(101.0, 102.0, 117.0, 118.0),
                vec4(105.0, 106.0, 125.0, 126.0),
                vec4(109.0, 110.0, 127.0, 128.0),
                vec4(113.0, 114.0, 129.0, 130.0),
            ),
            True,
        )

        m[-2:, 2:] = wp.matrix_from_rows(
            vec2(131.0, 132.0),
            vec2(133.0, 134.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(101.0, 102.0, 117.0, 118.0),
                vec4(105.0, 106.0, 125.0, 126.0),
                vec4(109.0, 110.0, 131.0, 132.0),
                vec4(113.0, 114.0, 133.0, 134.0),
            ),
            True,
        )

        m[2:, -2:] = wp.matrix_from_rows(
            vec2(135.0, 136.0),
            vec2(137.0, 138.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(101.0, 102.0, 117.0, 118.0),
                vec4(105.0, 106.0, 125.0, 126.0),
                vec4(109.0, 110.0, 135.0, 136.0),
                vec4(113.0, 114.0, 137.0, 138.0),
            ),
            True,
        )

        m[1:, :2] = wp.matrix_from_rows(
            vec2(139.0, 140.0),
            vec2(141.0, 142.0),
            vec2(143.0, 144.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(101.0, 102.0, 117.0, 118.0),
                vec4(139.0, 140.0, 125.0, 126.0),
                vec4(141.0, 142.0, 135.0, 136.0),
                vec4(143.0, 144.0, 137.0, 138.0),
            ),
            True,
        )

        m[:1, 2:] = wp.matrix_from_rows(
            vec2(145.0, 146.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(101.0, 102.0, 145.0, 146.0),
                vec4(139.0, 140.0, 125.0, 126.0),
                vec4(141.0, 142.0, 135.0, 136.0),
                vec4(143.0, 144.0, 137.0, 138.0),
            ),
            True,
        )

        m[:2, 0] = vec2(147.0, 148.0)
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(147.0, 102.0, 145.0, 146.0),
                vec4(148.0, 140.0, 125.0, 126.0),
                vec4(141.0, 142.0, 135.0, 136.0),
                vec4(143.0, 144.0, 137.0, 138.0),
            ),
            True,
        )

        m[2:, 1] = vec2(149.0, 150.0)
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(147.0, 102.0, 145.0, 146.0),
                vec4(148.0, 140.0, 125.0, 126.0),
                vec4(141.0, 149.0, 135.0, 136.0),
                vec4(143.0, 150.0, 137.0, 138.0),
            ),
            True,
        )

        m[0, :3] = vec3(151.0, 152.0, 153.0)
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(151.0, 152.0, 153.0, 146.0),
                vec4(148.0, 140.0, 125.0, 126.0),
                vec4(141.0, 149.0, 135.0, 136.0),
                vec4(143.0, 150.0, 137.0, 138.0),
            ),
            True,
        )

        m[1, 1:] = vec3(154.0, 155.0, 156.0)
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(151.0, 152.0, 153.0, 146.0),
                vec4(148.0, 154.0, 155.0, 156.0),
                vec4(141.0, 149.0, 135.0, 136.0),
                vec4(143.0, 150.0, 137.0, 138.0),
            ),
            True,
        )

        m[0, 2] = 157.0
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(151.0, 152.0, 157.0, 146.0),
                vec4(148.0, 154.0, 155.0, 156.0),
                vec4(141.0, 149.0, 135.0, 136.0),
                vec4(143.0, 150.0, 137.0, 138.0),
            ),
            True,
        )

        m[3, 1:] += vec3(158.0, 159.0, 160.0)
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(151.0, 152.0, 157.0, 146.0),
                vec4(148.0, 154.0, 155.0, 156.0),
                vec4(141.0, 149.0, 135.0, 136.0),
                vec4(143.0, 308.0, 296.0, 298.0),
            ),
            True,
        )

        m[2:, 1] += vec2(161.0, 162.0)
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(151.0, 152.0, 157.0, 146.0),
                vec4(148.0, 154.0, 155.0, 156.0),
                vec4(141.0, 310.0, 135.0, 136.0),
                vec4(143.0, 470.0, 296.0, 298.0),
            ),
            True,
        )

        m[2:, 3] -= vec2(163.0, 164.0)
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(151.0, 152.0, 157.0, 146.0),
                vec4(148.0, 154.0, 155.0, 156.0),
                vec4(141.0, 310.0, 135.0, -27.0),
                vec4(143.0, 470.0, 296.0, 134.0),
            ),
            True,
        )

        m[1, :3] -= vec3(165.0, 166.0, 167.0)
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(151.0, 152.0, 157.0, 146.0),
                vec4(-17.0, -12.0, -12.0, 156.0),
                vec4(141.0, 310.0, 135.0, -27.0),
                vec4(143.0, 470.0, 296.0, 134.0),
            ),
            True,
        )

        m[:-2, 2:] *= 3.0
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(151.0, 152.0, 471.0, 438.0),
                vec4(-17.0, -12.0, -36.0, 468.0),
                vec4(141.0, 310.0, 135.0, -27.0),
                vec4(143.0, 470.0, 296.0, 134.0),
            ),
            True,
        )

        m[-2:, 1] *= 4.0
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(151.0, 152.0, 471.0, 438.0),
                vec4(-17.0, -12.0, -36.0, 468.0),
                vec4(141.0, 1240.0, 135.0, -27.0),
                vec4(143.0, 1880.0, 296.0, 134.0),
            ),
            True,
        )

        m[3, :1] *= 5.0
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(151.0, 152.0, 471.0, 438.0),
                vec4(-17.0, -12.0, -36.0, 468.0),
                vec4(141.0, 1240.0, 135.0, -27.0),
                vec4(715.0, 1880.0, 296.0, 134.0),
            ),
            True,
        )

        m[:2, :2] /= 2.0
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(75.5, 76.0, 471.0, 438.0),
                vec4(-8.5, -6.0, -36.0, 468.0),
                vec4(141.0, 1240.0, 135.0, -27.0),
                vec4(715.0, 1880.0, 296.0, 134.0),
            ),
            True,
        )

        m[3:, 3] /= 4.0
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(75.5, 76.0, 471.0, 438.0),
                vec4(-8.5, -6.0, -36.0, 468.0),
                vec4(141.0, 1240.0, 135.0, -27.0),
                vec4(715.0, 1880.0, 296.0, 33.5),
            ),
            True,
        )

        m[0, :2] /= 4.0
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(18.875, 19.0, 471.0, 438.0),
                vec4(-8.5, -6.0, -36.0, 468.0),
                vec4(141.0, 1240.0, 135.0, -27.0),
                vec4(715.0, 1880.0, 296.0, 33.5),
            ),
            True,
        )

    @wp.kernel(module="unique")
    def kernel():
        fn()

    wp.launch(kernel, 1, device=device)
    wp.synchronize()
    fn()


def test_mat_from_cols_slicing_assign(test, device):
    mat00 = wp.mat((0, 0), float)
    vec1 = wp.vec(1, float)
    vec2 = wp.vec(2, float)
    vec3 = wp.vec(3, float)
    vec4 = wp.vec(4, float)

    @wp.func
    def fn():
        m = wp.matrix_from_cols(
            vec4(1.0, 2.0, 3.0, 4.0),
            vec4(5.0, 6.0, 7.0, 8.0),
            vec4(9.0, 10.0, 11.0, 12.0),
            vec4(13.0, 14.0, 15.0, 16.0),
        )

        wp.expect_eq(
            m[:]
            == wp.matrix_from_rows(
                vec4(1.0, 5.0, 9.0, 13.0),
                vec4(2.0, 6.0, 10.0, 14.0),
                vec4(3.0, 7.0, 11.0, 15.0),
                vec4(4.0, 8.0, 12.0, 16.0),
            ),
            True,
        )
        wp.expect_eq(
            m[-123:123]
            == wp.matrix_from_rows(
                vec4(1.0, 5.0, 9.0, 13.0),
                vec4(2.0, 6.0, 10.0, 14.0),
                vec4(3.0, 7.0, 11.0, 15.0),
                vec4(4.0, 8.0, 12.0, 16.0),
            ),
            True,
        )
        wp.expect_eq(m[123:] == mat00(), True)
        wp.expect_eq(m[:-123] == mat00(), True)
        wp.expect_eq(
            m[::123]
            == wp.matrix_from_rows(
                vec4(1.0, 5.0, 9.0, 13.0),
            ),
            True,
        )

        wp.expect_eq(
            m[1:]
            == wp.matrix_from_rows(
                vec4(2.0, 6.0, 10.0, 14.0),
                vec4(3.0, 7.0, 11.0, 15.0),
                vec4(4.0, 8.0, 12.0, 16.0),
            ),
            True,
        )
        wp.expect_eq(
            m[-2:]
            == wp.matrix_from_rows(
                vec4(3.0, 7.0, 11.0, 15.0),
                vec4(4.0, 8.0, 12.0, 16.0),
            ),
            True,
        )
        wp.expect_eq(
            m[:2]
            == wp.matrix_from_rows(
                vec4(1.0, 5.0, 9.0, 13.0),
                vec4(2.0, 6.0, 10.0, 14.0),
            ),
            True,
        )
        wp.expect_eq(
            m[:-1]
            == wp.matrix_from_rows(
                vec4(1.0, 5.0, 9.0, 13.0),
                vec4(2.0, 6.0, 10.0, 14.0),
                vec4(3.0, 7.0, 11.0, 15.0),
            ),
            True,
        )
        wp.expect_eq(
            m[::2]
            == wp.matrix_from_rows(
                vec4(1.0, 5.0, 9.0, 13.0),
                vec4(3.0, 7.0, 11.0, 15.0),
            ),
            True,
        )
        wp.expect_eq(
            m[1::2]
            == wp.matrix_from_rows(
                vec4(2.0, 6.0, 10.0, 14.0),
                vec4(4.0, 8.0, 12.0, 16.0),
            ),
            True,
        )
        wp.expect_eq(
            m[::-1]
            == wp.matrix_from_rows(
                vec4(4.0, 8.0, 12.0, 16.0),
                vec4(3.0, 7.0, 11.0, 15.0),
                vec4(2.0, 6.0, 10.0, 14.0),
                vec4(1.0, 5.0, 9.0, 13.0),
            ),
            True,
        )
        wp.expect_eq(
            m[::-2]
            == wp.matrix_from_rows(
                vec4(4.0, 8.0, 12.0, 16.0),
                vec4(2.0, 6.0, 10.0, 14.0),
            ),
            True,
        )
        wp.expect_eq(
            m[1::-2]
            == wp.matrix_from_rows(
                vec4(2.0, 6.0, 10.0, 14.0),
            ),
            True,
        )

        wp.expect_eq(
            m[:, :]
            == wp.matrix_from_rows(
                vec4(1.0, 5.0, 9.0, 13.0),
                vec4(2.0, 6.0, 10.0, 14.0),
                vec4(3.0, 7.0, 11.0, 15.0),
                vec4(4.0, 8.0, 12.0, 16.0),
            ),
            True,
        )
        wp.expect_eq(
            m[:, 2:]
            == wp.matrix_from_rows(
                vec2(9.0, 13.0),
                vec2(10.0, 14.0),
                vec2(11.0, 15.0),
                vec2(12.0, 16.0),
            ),
            True,
        )
        wp.expect_eq(
            m[1:, 2:]
            == wp.matrix_from_rows(
                vec2(10.0, 14.0),
                vec2(11.0, 15.0),
                vec2(12.0, 16.0),
            ),
            True,
        )
        wp.expect_eq(
            m[-2:, 2:]
            == wp.matrix_from_rows(
                vec2(11.0, 15.0),
                vec2(12.0, 16.0),
            ),
            True,
        )
        wp.expect_eq(
            m[2:, -2:]
            == wp.matrix_from_rows(
                vec2(11.0, 15.0),
                vec2(12.0, 16.0),
            ),
            True,
        )
        wp.expect_eq(
            m[1:, :2]
            == wp.matrix_from_rows(
                vec2(2.0, 6.0),
                vec2(3.0, 7.0),
                vec2(4.0, 8.0),
            ),
            True,
        )
        wp.expect_eq(
            m[:1, 2:]
            == wp.matrix_from_rows(
                vec2(9.0, 13.0),
            ),
            True,
        )
        wp.expect_eq(
            m[::-1, :1]
            == wp.matrix_from_rows(
                vec1(4.0),
                vec1(3.0),
                vec1(2.0),
                vec1(1.0),
            ),
            True,
        )
        wp.expect_eq(
            m[:1, ::-1]
            == wp.matrix_from_rows(
                vec4(13.0, 9.0, 5.0, 1.0),
            ),
            True,
        )
        wp.expect_eq(
            m[:1:-1, 2::-1]
            == wp.matrix_from_rows(
                vec3(12.0, 8.0, 4.0),
                vec3(11.0, 7.0, 3.0),
            ),
            True,
        )

        wp.expect_eq(m[:2, 0] == vec2(1.0, 2.0), True)
        wp.expect_eq(m[2:, 1] == vec2(7.0, 8.0), True)
        wp.expect_eq(m[0, :3] == vec3(1.0, 5.0, 9.0), True)
        wp.expect_eq(m[1, 1:] == vec3(6.0, 10.0, 14.0), True)

        m[1:] = wp.matrix_from_cols(
            vec3(17.0, 18.0, 19.0),
            vec3(20.0, 21.0, 22.0),
            vec3(23.0, 24.0, 25.0),
            vec3(26.0, 27.0, 28.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(1.0, 5.0, 9.0, 13.0),
                vec4(17.0, 20.0, 23.0, 26.0),
                vec4(18.0, 21.0, 24.0, 27.0),
                vec4(19.0, 22.0, 25.0, 28.0),
            ),
            True,
        )

        m[-2:] = wp.matrix_from_cols(
            vec2(29.0, 30.0),
            vec2(31.0, 32.0),
            vec2(33.0, 34.0),
            vec2(35.0, 36.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(1.0, 5.0, 9.0, 13.0),
                vec4(17.0, 20.0, 23.0, 26.0),
                vec4(29.0, 31.0, 33.0, 35.0),
                vec4(30.0, 32.0, 34.0, 36.0),
            ),
            True,
        )

        m[:2] = wp.matrix_from_cols(
            vec2(37.0, 38.0),
            vec2(39.0, 40.0),
            vec2(41.0, 42.0),
            vec2(43.0, 44.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(37.0, 39.0, 41.0, 43.0),
                vec4(38.0, 40.0, 42.0, 44.0),
                vec4(29.0, 31.0, 33.0, 35.0),
                vec4(30.0, 32.0, 34.0, 36.0),
            ),
            True,
        )

        m[:-1] = wp.matrix_from_cols(
            vec3(45.0, 46.0, 47.0),
            vec3(48.0, 49.0, 50.0),
            vec3(51.0, 52.0, 53.0),
            vec3(54.0, 55.0, 56.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(45.0, 48.0, 51.0, 54.0),
                vec4(46.0, 49.0, 52.0, 55.0),
                vec4(47.0, 50.0, 53.0, 56.0),
                vec4(30.0, 32.0, 34.0, 36.0),
            ),
            True,
        )

        m[::2] = wp.matrix_from_cols(
            vec2(57.0, 58.0),
            vec2(59.0, 60.0),
            vec2(61.0, 62.0),
            vec2(63.0, 64.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(57.0, 59.0, 61.0, 63.0),
                vec4(46.0, 49.0, 52.0, 55.0),
                vec4(58.0, 60.0, 62.0, 64.0),
                vec4(30.0, 32.0, 34.0, 36.0),
            ),
            True,
        )

        m[1::2] = wp.matrix_from_cols(
            vec2(65.0, 66.0),
            vec2(67.0, 68.0),
            vec2(69.0, 70.0),
            vec2(71.0, 72.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(57.0, 59.0, 61.0, 63.0),
                vec4(65.0, 67.0, 69.0, 71.0),
                vec4(58.0, 60.0, 62.0, 64.0),
                vec4(66.0, 68.0, 70.0, 72.0),
            ),
            True,
        )

        m[::-1] = wp.matrix_from_cols(
            vec4(73.0, 74.0, 75.0, 76.0),
            vec4(77.0, 78.0, 79.0, 80.0),
            vec4(81.0, 82.0, 83.0, 84.0),
            vec4(85.0, 86.0, 87.0, 88.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(76.0, 80.0, 84.0, 88.0),
                vec4(75.0, 79.0, 83.0, 87.0),
                vec4(74.0, 78.0, 82.0, 86.0),
                vec4(73.0, 77.0, 81.0, 85.0),
            ),
            True,
        )

        m[::-2] = wp.matrix_from_cols(
            vec2(89.0, 90.0),
            vec2(91.0, 92.0),
            vec2(93.0, 94.0),
            vec2(95.0, 96.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(76.0, 80.0, 84.0, 88.0),
                vec4(90.0, 92.0, 94.0, 96.0),
                vec4(74.0, 78.0, 82.0, 86.0),
                vec4(89.0, 91.0, 93.0, 95.0),
            ),
            True,
        )

        m[1::-2] = wp.matrix_from_cols(
            vec1(97.0),
            vec1(98.0),
            vec1(99.0),
            vec1(100.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(76.0, 80.0, 84.0, 88.0),
                vec4(97.0, 98.0, 99.0, 100.0),
                vec4(74.0, 78.0, 82.0, 86.0),
                vec4(89.0, 91.0, 93.0, 95.0),
            ),
            True,
        )

        m[:, :] = wp.matrix_from_cols(
            vec4(101.0, 102.0, 103.0, 104.0),
            vec4(105.0, 106.0, 107.0, 108.0),
            vec4(109.0, 110.0, 111.0, 112.0),
            vec4(113.0, 114.0, 115.0, 116.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(101.0, 105.0, 109.0, 113.0),
                vec4(102.0, 106.0, 110.0, 114.0),
                vec4(103.0, 107.0, 111.0, 115.0),
                vec4(104.0, 108.0, 112.0, 116.0),
            ),
            True,
        )

        m[:, 2:] = wp.matrix_from_cols(
            vec4(117.0, 118.0, 119.0, 120.0),
            vec4(121.0, 122.0, 123.0, 124.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(101.0, 105.0, 117.0, 121.0),
                vec4(102.0, 106.0, 118.0, 122.0),
                vec4(103.0, 107.0, 119.0, 123.0),
                vec4(104.0, 108.0, 120.0, 124.0),
            ),
            True,
        )

        m[1:, 2:] = wp.matrix_from_cols(
            vec3(125.0, 126.0, 127.0),
            vec3(128.0, 129.0, 130.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(101.0, 105.0, 117.0, 121.0),
                vec4(102.0, 106.0, 125.0, 128.0),
                vec4(103.0, 107.0, 126.0, 129.0),
                vec4(104.0, 108.0, 127.0, 130.0),
            ),
            True,
        )

        m[-2:, 2:] = wp.matrix_from_cols(
            vec2(131.0, 132.0),
            vec2(133.0, 134.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(101.0, 105.0, 117.0, 121.0),
                vec4(102.0, 106.0, 125.0, 128.0),
                vec4(103.0, 107.0, 131.0, 133.0),
                vec4(104.0, 108.0, 132.0, 134.0),
            ),
            True,
        )

        m[2:, -2:] = wp.matrix_from_cols(
            vec2(135.0, 136.0),
            vec2(137.0, 138.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(101.0, 105.0, 117.0, 121.0),
                vec4(102.0, 106.0, 125.0, 128.0),
                vec4(103.0, 107.0, 135.0, 137.0),
                vec4(104.0, 108.0, 136.0, 138.0),
            ),
            True,
        )

        m[1:, :2] = wp.matrix_from_cols(
            vec3(139.0, 140.0, 141.0),
            vec3(142.0, 143.0, 144.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(101.0, 105.0, 117.0, 121.0),
                vec4(139.0, 142.0, 125.0, 128.0),
                vec4(140.0, 143.0, 135.0, 137.0),
                vec4(141.0, 144.0, 136.0, 138.0),
            ),
            True,
        )

        m[:1, 2:] = wp.matrix_from_cols(
            vec1(145.0),
            vec1(146.0),
        )
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(101.0, 105.0, 145.0, 146.0),
                vec4(139.0, 142.0, 125.0, 128.0),
                vec4(140.0, 143.0, 135.0, 137.0),
                vec4(141.0, 144.0, 136.0, 138.0),
            ),
            True,
        )

        m[:2, 0] = vec2(147.0, 148.0)
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(147.0, 105.0, 145.0, 146.0),
                vec4(148.0, 142.0, 125.0, 128.0),
                vec4(140.0, 143.0, 135.0, 137.0),
                vec4(141.0, 144.0, 136.0, 138.0),
            ),
            True,
        )

        m[2:, 1] = vec2(149.0, 150.0)
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(147.0, 105.0, 145.0, 146.0),
                vec4(148.0, 142.0, 125.0, 128.0),
                vec4(140.0, 149.0, 135.0, 137.0),
                vec4(141.0, 150.0, 136.0, 138.0),
            ),
            True,
        )

        m[0, :3] = vec3(151.0, 152.0, 153.0)
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(151.0, 152.0, 153.0, 146.0),
                vec4(148.0, 142.0, 125.0, 128.0),
                vec4(140.0, 149.0, 135.0, 137.0),
                vec4(141.0, 150.0, 136.0, 138.0),
            ),
            True,
        )

        m[1, 1:] = vec3(154.0, 155.0, 156.0)
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(151.0, 152.0, 153.0, 146.0),
                vec4(148.0, 154.0, 155.0, 156.0),
                vec4(140.0, 149.0, 135.0, 137.0),
                vec4(141.0, 150.0, 136.0, 138.0),
            ),
            True,
        )

        m[0, 2] = 157.0
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(151.0, 152.0, 157.0, 146.0),
                vec4(148.0, 154.0, 155.0, 156.0),
                vec4(140.0, 149.0, 135.0, 137.0),
                vec4(141.0, 150.0, 136.0, 138.0),
            ),
            True,
        )

        m[3, 1:] += vec3(158.0, 159.0, 160.0)
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(151.0, 152.0, 157.0, 146.0),
                vec4(148.0, 154.0, 155.0, 156.0),
                vec4(140.0, 149.0, 135.0, 137.0),
                vec4(141.0, 308.0, 295.0, 298.0),
            ),
            True,
        )

        m[2:, 1] += vec2(161.0, 162.0)
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(151.0, 152.0, 157.0, 146.0),
                vec4(148.0, 154.0, 155.0, 156.0),
                vec4(140.0, 310.0, 135.0, 137.0),
                vec4(141.0, 470.0, 295.0, 298.0),
            ),
            True,
        )

        m[2:, 3] -= vec2(163.0, 164.0)
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(151.0, 152.0, 157.0, 146.0),
                vec4(148.0, 154.0, 155.0, 156.0),
                vec4(140.0, 310.0, 135.0, -26.0),
                vec4(141.0, 470.0, 295.0, 134.0),
            ),
            True,
        )

        m[1, :3] -= vec3(165.0, 166.0, 167.0)
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(151.0, 152.0, 157.0, 146.0),
                vec4(-17.0, -12.0, -12.0, 156.0),
                vec4(140.0, 310.0, 135.0, -26.0),
                vec4(141.0, 470.0, 295.0, 134.0),
            ),
            True,
        )

        m[:-2, 2:] *= 3.0
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(151.0, 152.0, 471.0, 438.0),
                vec4(-17.0, -12.0, -36.0, 468.0),
                vec4(140.0, 310.0, 135.0, -26.0),
                vec4(141.0, 470.0, 295.0, 134.0),
            ),
            True,
        )

        m[-2:, 1] *= 4.0
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(151.0, 152.0, 471.0, 438.0),
                vec4(-17.0, -12.0, -36.0, 468.0),
                vec4(140.0, 1240.0, 135.0, -26.0),
                vec4(141.0, 1880.0, 295.0, 134.0),
            ),
            True,
        )

        m[3, :1] *= 5.0
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(151.0, 152.0, 471.0, 438.0),
                vec4(-17.0, -12.0, -36.0, 468.0),
                vec4(140.0, 1240.0, 135.0, -26.0),
                vec4(705.0, 1880.0, 295.0, 134.0),
            ),
            True,
        )

        m[:2, :2] /= 2.0
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(75.5, 76.0, 471.0, 438.0),
                vec4(-8.5, -6.0, -36.0, 468.0),
                vec4(140.0, 1240.0, 135.0, -26.0),
                vec4(705.0, 1880.0, 295.0, 134.0),
            ),
            True,
        )

        m[3:, 3] /= 4.0
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(75.5, 76.0, 471.0, 438.0),
                vec4(-8.5, -6.0, -36.0, 468.0),
                vec4(140.0, 1240.0, 135.0, -26.0),
                vec4(705.0, 1880.0, 295.0, 33.5),
            ),
            True,
        )

        m[0, :2] /= 4.0
        wp.expect_eq(
            m
            == wp.matrix_from_rows(
                vec4(18.875, 19.0, 471.0, 438.0),
                vec4(-8.5, -6.0, -36.0, 468.0),
                vec4(140.0, 1240.0, 135.0, -26.0),
                vec4(705.0, 1880.0, 295.0, 33.5),
            ),
            True,
        )

    @wp.kernel(module="unique")
    def kernel():
        fn()

    wp.launch(kernel, 1, device=device)
    wp.synchronize()
    fn()


def test_mat_slicing_assign_backward(test, device):
    mat23 = wp.mat((2, 3), float)

    @wp.kernel(module="unique")
    def kernel(
        arr_x: wp.array(dtype=wp.vec2),
        arr_y: wp.array(dtype=mat23),
        arr_z: wp.array(dtype=wp.mat44),
    ):
        i = wp.tid()

        z = arr_z[i]

        z[0, :2] = arr_x[i]
        z[:2, 1:] = arr_y[i]

        z[:2, 3] += arr_x[i][:2]
        z[1:-1, :2] += arr_y[i][::-1, :-1]

        z[2:, 3] -= arr_x[i][0:]
        z[3:, -1:] -= arr_y[i][:1, :1]

        arr_z[i] = z

    x = wp.ones(1, dtype=wp.vec2, requires_grad=True, device=device)
    y = wp.ones(1, dtype=mat23, requires_grad=True, device=device)
    z = wp.zeros(1, dtype=wp.mat44, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, 1, inputs=(x, y), outputs=(z,), device=device)

    z.grad = wp.ones_like(z)
    tape.backward()

    assert_np_equal(
        z.numpy(),
        np.array(
            (
                (
                    (1.0, 1.0, 1.0, 2.0),
                    (1.0, 2.0, 1.0, 2.0),
                    (1.0, 1.0, 0.0, -1.0),
                    (0.0, 0.0, 0.0, -2.0),
                ),
            ),
            dtype=float,
        ),
    )
    assert_np_equal(x.grad.numpy(), np.array(((1.0, 1.0),), dtype=float))
    assert_np_equal(y.grad.numpy(), np.array((((1.0, 2.0, 1.0), (2.0, 2.0, 1.0)),), dtype=float))


devices = get_test_devices()


class TestMat(unittest.TestCase):
    def test_tpl_ops_with_anon(self):
        mat22f = wp.mat((2, 2), dtype=float)

        m = wp.mat22f(1.0, 2.0, 3.0, 4.0)
        m += mat22f(2.0, 3.0, 4.0, 5.0)
        m -= mat22f(3.0, 4.0, 5.0, 6.0)
        self.assertSequenceEqual(m, ((0.0, 1.0), (2.0, 3.0)))

        m = mat22f(1.0, 2.0, 3.0, 4.0)
        m += wp.mat22f(2.0, 3.0, 4.0, 5.0)
        m -= wp.mat22f(3.0, 4.0, 5.0, 6.0)
        self.assertSequenceEqual(m, ((0.0, 1.0), (2.0, 3.0)))


mat103 = wp._src.types.matrix(shape=(10, 3), dtype=float)
add_kernel_test(
    TestMat,
    test_matrix_mutation,
    dim=1,
    inputs=[
        mat103(
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0,
            3.0, 6.0, 9.0,
            4.0, 8.0, 12.0,
            5.0, 10.0, 15.0,
            6.0, 12.0, 18.0,
            7.0, 14.0, 21.0,
            8.0, 16.0, 24.0,
            9.0, 18.0, 27.0,
            10.0, 20.0, 30.0,
        )
    ],
    devices=devices,
)  # fmt: skip

for dtype in np_signed_int_types + np_float_types:
    add_function_test_register_kernel(
        TestMat, f"test_negation_{dtype.__name__}", test_negation, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMat, f"test_subtraction_{dtype.__name__}", test_subtraction, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMat, f"test_matmul_{dtype.__name__}", test_matmul, devices=devices, dtype=dtype
    )

add_function_test(TestMat, "test_shape_mismatch", test_shape_mismatch, devices=devices)


for dtype in np_float_types:
    add_function_test(
        TestMat, f"test_py_arithmetic_ops_{dtype.__name__}", test_py_arithmetic_ops, devices=None, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMat, f"test_inverse_{dtype.__name__}", test_inverse, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(TestMat, f"test_svd_{dtype.__name__}", test_svd, devices=devices, dtype=dtype)
    add_function_test_register_kernel(
        TestMat, f"test_svd_2D{dtype.__name__}", test_svd_2D, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(TestMat, f"test_qr_{dtype.__name__}", test_qr, devices=devices, dtype=dtype)
    add_function_test_register_kernel(TestMat, f"test_eig_{dtype.__name__}", test_eig, devices=devices, dtype=dtype)
    add_function_test_register_kernel(
        TestMat, f"test_transform_point_{dtype.__name__}", test_transform_point, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMat, f"test_transform_vector_{dtype.__name__}", test_transform_vector, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMat, f"test_determinant_{dtype.__name__}", test_determinant, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(TestMat, f"test_skew_{dtype.__name__}", test_skew, devices=devices, dtype=dtype)

add_function_test(TestMat, "test_matrix_len", test_matrix_len, devices=devices)
add_function_test(TestMat, "test_mat_extract", test_mat_extract, devices=devices)
add_function_test(TestMat, "test_mat_assign", test_mat_assign, devices=devices)
add_function_test(TestMat, "test_mat_array_extract", test_mat_array_extract, devices=devices)
# add_function_test(TestMat, "test_mat_array_assign", test_mat_array_assign, devices=devices)
add_function_test(TestMat, "test_mat_add_inplace", test_mat_add_inplace, devices=devices)
add_function_test(TestMat, "test_mat_sub_inplace", test_mat_sub_inplace, devices=devices)
add_function_test(TestMat, "test_mat_array_add_inplace", test_mat_array_add_inplace, devices=devices)
add_function_test(TestMat, "test_mat_array_sub_inplace", test_mat_array_sub_inplace, devices=devices)
add_function_test(TestMat, "test_scalar_mat_div", test_scalar_mat_div, devices=devices)
add_function_test(TestMat, "test_mat_from_rows_indexing_assign", test_mat_from_rows_indexing_assign, devices=devices)
add_function_test(TestMat, "test_mat_from_cols_indexing_assign", test_mat_from_cols_indexing_assign, devices=devices)
add_function_test(TestMat, "test_mat_from_rows_slicing_assign", test_mat_from_rows_slicing_assign, devices=devices)
add_function_test(TestMat, "test_mat_from_cols_slicing_assign", test_mat_from_cols_slicing_assign, devices=devices)
add_function_test(TestMat, "test_mat_slicing_assign_backward", test_mat_slicing_assign_backward, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
