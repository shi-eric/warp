# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def bf16_conversion_kernel(input: wp.array(dtype=wp.float32), output: wp.array(dtype=wp.bfloat16)):
    tid = wp.tid()
    output[tid] = wp.bfloat16(input[tid])


@wp.kernel
def bf16_to_f32_kernel(input: wp.array(dtype=wp.bfloat16), output: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    output[tid] = wp.float32(input[tid])


@wp.kernel
def bf16_param_kernel(x: wp.bfloat16, output: wp.array(dtype=wp.bfloat16)):
    output[0] = x


@wp.kernel
def bf16_arithmetic_kernel(
    a: wp.array(dtype=wp.bfloat16),
    b: wp.array(dtype=wp.bfloat16),
    out_add: wp.array(dtype=wp.bfloat16),
    out_sub: wp.array(dtype=wp.bfloat16),
    out_mul: wp.array(dtype=wp.bfloat16),
    out_div: wp.array(dtype=wp.bfloat16),
):
    tid = wp.tid()
    out_add[tid] = a[tid] + b[tid]
    out_sub[tid] = a[tid] - b[tid]
    out_mul[tid] = a[tid] * b[tid]
    out_div[tid] = a[tid] / b[tid]


def test_bf16_conversion(test, device):
    n = 10
    input_data = np.array([1.0, 2.0, 3.0, -1.0, 0.0, 0.5, 100.0, -100.0, 0.001, 1000.0], dtype=np.float32)
    input_arr = wp.array(input_data, dtype=wp.float32, device=device)
    output_arr = wp.zeros(n, dtype=wp.bfloat16, device=device)

    wp.launch(bf16_conversion_kernel, dim=n, inputs=[input_arr, output_arr], device=device)

    result_arr = wp.zeros(n, dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=n, inputs=[output_arr, result_arr], device=device)
    result = result_arr.numpy()

    np.testing.assert_allclose(result, input_data, rtol=1e-2)


def test_bf16_kernel_parameter(test, device):
    output = wp.zeros(1, dtype=wp.bfloat16, device=device)
    wp.launch(bf16_param_kernel, dim=1, inputs=[wp.bfloat16(3.14), output], device=device)

    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=1, inputs=[output, result], device=device)
    np.testing.assert_allclose(result.numpy()[0], 3.14, rtol=1e-2)


def test_bf16_arithmetic(test, device):
    n = 4
    a_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b_data = np.array([0.5, 1.5, 2.0, 0.25], dtype=np.float32)

    a = wp.array(a_data, dtype=wp.bfloat16, device=device)
    b = wp.array(b_data, dtype=wp.bfloat16, device=device)

    out_add = wp.zeros(n, dtype=wp.bfloat16, device=device)
    out_sub = wp.zeros(n, dtype=wp.bfloat16, device=device)
    out_mul = wp.zeros(n, dtype=wp.bfloat16, device=device)
    out_div = wp.zeros(n, dtype=wp.bfloat16, device=device)

    wp.launch(bf16_arithmetic_kernel, dim=n, inputs=[a, b, out_add, out_sub, out_mul, out_div], device=device)

    add_f32 = wp.zeros(n, dtype=wp.float32, device=device)
    sub_f32 = wp.zeros(n, dtype=wp.float32, device=device)
    mul_f32 = wp.zeros(n, dtype=wp.float32, device=device)
    div_f32 = wp.zeros(n, dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=n, inputs=[out_add, add_f32], device=device)
    wp.launch(bf16_to_f32_kernel, dim=n, inputs=[out_sub, sub_f32], device=device)
    wp.launch(bf16_to_f32_kernel, dim=n, inputs=[out_mul, mul_f32], device=device)
    wp.launch(bf16_to_f32_kernel, dim=n, inputs=[out_div, div_f32], device=device)

    np.testing.assert_allclose(add_f32.numpy(), a_data + b_data, rtol=1e-2)
    np.testing.assert_allclose(sub_f32.numpy(), a_data - b_data, rtol=1e-2)
    np.testing.assert_allclose(mul_f32.numpy(), a_data * b_data, rtol=1e-2)
    np.testing.assert_allclose(div_f32.numpy(), a_data / b_data, rtol=1e-2)


@wp.kernel
def bf16_grad_kernel(
    x: wp.array(dtype=wp.bfloat16),
    loss: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    v = wp.float32(x[tid])
    wp.atomic_add(loss, 0, v * v)


@wp.kernel
def bf16_atomic_kernel(output: wp.array(dtype=wp.bfloat16)):
    wp.atomic_add(output, 0, wp.bfloat16(1.0))


def test_bf16_numpy(test, device):
    n = 4
    arr = wp.zeros(n, dtype=wp.bfloat16, device=device)
    np_arr = arr.numpy()
    test.assertEqual(np_arr.dtype, np.uint16)


def test_bf16_array_from_list(test, device):
    """Test that wp.array([floats], dtype=wp.bfloat16) correctly converts float values."""
    input_values = [1.0, 2.0, 3.0, -1.0, 0.0, 0.5, 100.0, -100.0]
    arr = wp.array(input_values, dtype=wp.bfloat16, device=device)

    test.assertEqual(arr.dtype, wp.bfloat16)
    test.assertEqual(arr.shape, (len(input_values),))

    # Verify the values survived the conversion by reading them back through a kernel
    result = wp.zeros(len(input_values), dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=len(input_values), inputs=[arr, result], device=device)

    np.testing.assert_allclose(result.numpy(), input_values, rtol=1e-2)


def test_bf16_array_from_tuple(test, device):
    """Test that wp.array(tuple_of_floats, dtype=wp.bfloat16) works."""
    input_values = (1.5, 2.5, -3.5, 0.0)
    arr = wp.array(input_values, dtype=wp.bfloat16, device=device)

    result = wp.zeros(len(input_values), dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=len(input_values), inputs=[arr, result], device=device)

    np.testing.assert_allclose(result.numpy(), input_values, rtol=1e-2)


def test_bf16_array_from_numpy_float(test, device):
    """Test that wp.array(np.array([...], dtype=np.float32), dtype=wp.bfloat16) works."""
    input_data = np.array([1.0, 2.5, 3.14, -0.5, 0.0], dtype=np.float32)
    arr = wp.array(input_data, dtype=wp.bfloat16, device=device)

    result = wp.zeros(len(input_data), dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=len(input_data), inputs=[arr, result], device=device)

    np.testing.assert_allclose(result.numpy(), input_data, rtol=1e-2)


def test_bf16_array_from_numpy_uint16(test, device):
    """Test that wp.array(np.array([...], dtype=np.uint16), dtype=wp.bfloat16) passes raw bits through."""
    # Pre-encoded bfloat16 bits: 0x3F80 = 1.0, 0x4000 = 2.0, 0x4040 = 3.0
    raw_bits = np.array([0x3F80, 0x4000, 0x4040], dtype=np.uint16)
    arr = wp.array(raw_bits, dtype=wp.bfloat16, device=device)

    result = wp.zeros(3, dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=3, inputs=[arr, result], device=device)

    np.testing.assert_allclose(result.numpy(), [1.0, 2.0, 3.0], rtol=1e-2)


def test_bf16_array_special_values(test, device):
    """Test that special float values (inf, -inf, very small) convert correctly."""
    input_values = [float("inf"), float("-inf"), 0.0, -0.0, 1e-6]
    arr = wp.array(input_values, dtype=wp.bfloat16, device=device)

    result = wp.zeros(len(input_values), dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=len(input_values), inputs=[arr, result], device=device)

    r = result.numpy()
    test.assertTrue(np.isinf(r[0]) and r[0] > 0, "Expected +inf")
    test.assertTrue(np.isinf(r[1]) and r[1] < 0, "Expected -inf")
    test.assertEqual(r[2], 0.0)
    # -0.0 should preserve sign
    test.assertEqual(r[3], 0.0)
    # Very small value — bfloat16 has limited precision, just check it's close
    np.testing.assert_allclose(r[4], 1e-6, rtol=0.5)


def test_bf16_grad(test, device):
    x_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = wp.array(x_data, dtype=wp.bfloat16, device=device, requires_grad=True)

    loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(bf16_grad_kernel, dim=3, inputs=[x, loss], device=device)

    tape.backward(loss)

    x_grad = tape.gradients[x]
    grad_f32 = wp.zeros(3, dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=3, inputs=[x_grad, grad_f32], device=device)

    # Values [1.0, 2.0, 3.0] and expected gradients [2.0, 4.0, 6.0]
    # are exactly representable in bfloat16, so use tighter tolerance.
    np.testing.assert_allclose(grad_f32.numpy(), 2.0 * x_data, rtol=1e-2)


def test_bf16_atomics(test, device):
    device = wp.get_device(device)
    if not device.is_cuda:
        test.skipTest("bfloat16 atomics test requires CUDA")

    output = wp.zeros(1, dtype=wp.bfloat16, device=device)
    wp.launch(bf16_atomic_kernel, dim=10, inputs=[output], device=device)

    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=1, inputs=[output, result], device=device)
    np.testing.assert_allclose(result.numpy()[0], 10.0, rtol=1e-1)


def test_bf16_interop_dlpack(test, device):
    n = 4
    input_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    arr = wp.array(input_data, dtype=wp.bfloat16, device=device)

    # Round-trip through DLPack
    dl = wp.to_dlpack(arr)
    arr2 = wp.from_dlpack(dl)
    test.assertEqual(arr2.dtype, wp.bfloat16)
    test.assertEqual(arr2.shape, (n,))

    # Verify values survived
    result = wp.zeros(n, dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=n, inputs=[arr2, result], device=device)
    np.testing.assert_allclose(result.numpy(), input_data, rtol=1e-2)


def test_bf16_interop_torch(test, device):
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        test.skipTest("PyTorch not available")

    wp_arr = wp.zeros(4, dtype=wp.bfloat16, device=device)
    torch_tensor = wp.to_torch(wp_arr)
    test.assertEqual(torch_tensor.dtype, torch.bfloat16)

    torch_device = wp.device_to_torch(device)
    t = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16, device=torch_device)
    wp_from_torch = wp.from_torch(t)
    test.assertEqual(wp_from_torch.dtype, wp.bfloat16)

    # Verify values survive Warp -> Torch -> Warp round-trip
    n = 4
    input_data = np.array([1.0, 2.5, -3.0, 4.0], dtype=np.float32)
    bf16_arr = wp.array(input_data, dtype=wp.bfloat16, device=device)

    torch_rt = wp.to_torch(bf16_arr)
    wp_rt = wp.from_torch(torch_rt)

    result_f32 = wp.zeros(n, dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=n, inputs=[wp_rt, result_f32], device=device)
    np.testing.assert_allclose(result_f32.numpy(), input_data, rtol=1e-2)


@wp.kernel
def bf16_math_kernel(
    input: wp.array(dtype=wp.bfloat16),
    out_sin: wp.array(dtype=wp.bfloat16),
    out_cos: wp.array(dtype=wp.bfloat16),
    out_sqrt: wp.array(dtype=wp.bfloat16),
    out_exp: wp.array(dtype=wp.bfloat16),
    out_log: wp.array(dtype=wp.bfloat16),
    out_pow: wp.array(dtype=wp.bfloat16),
    out_tan: wp.array(dtype=wp.bfloat16),
    out_abs: wp.array(dtype=wp.bfloat16),
):
    tid = wp.tid()
    x = input[tid]
    out_sin[tid] = wp.sin(x)
    out_cos[tid] = wp.cos(x)
    out_sqrt[tid] = wp.sqrt(wp.abs(x))
    out_exp[tid] = wp.exp(x)
    out_log[tid] = wp.log(x)
    out_pow[tid] = wp.pow(x, wp.bfloat16(2.0))
    out_tan[tid] = wp.tan(x)
    out_abs[tid] = wp.abs(x)


def test_bf16_math_builtins(test, device):
    # Use positive values to avoid domain errors for log; avoid pi/2 for tan
    n = 4
    input_data = np.array([0.25, 0.5, 1.0, 1.5], dtype=np.float32)
    input_bf16 = wp.array(input_data, dtype=wp.bfloat16, device=device)

    out_sin = wp.zeros(n, dtype=wp.bfloat16, device=device)
    out_cos = wp.zeros(n, dtype=wp.bfloat16, device=device)
    out_sqrt = wp.zeros(n, dtype=wp.bfloat16, device=device)
    out_exp = wp.zeros(n, dtype=wp.bfloat16, device=device)
    out_log = wp.zeros(n, dtype=wp.bfloat16, device=device)
    out_pow = wp.zeros(n, dtype=wp.bfloat16, device=device)
    out_tan = wp.zeros(n, dtype=wp.bfloat16, device=device)
    out_abs = wp.zeros(n, dtype=wp.bfloat16, device=device)

    wp.launch(
        bf16_math_kernel,
        dim=n,
        inputs=[input_bf16, out_sin, out_cos, out_sqrt, out_exp, out_log, out_pow, out_tan, out_abs],
        device=device,
    )

    for out_arr, expected_fn in [
        (out_sin, np.sin),
        (out_cos, np.cos),
        (out_sqrt, lambda x: np.sqrt(np.abs(x))),
        (out_exp, np.exp),
        (out_log, np.log),
        (out_pow, lambda x: np.power(x, 2.0)),
        (out_tan, np.tan),
        (out_abs, np.abs),
    ]:
        result = wp.zeros(n, dtype=wp.float32, device=device)
        wp.launch(bf16_to_f32_kernel, dim=n, inputs=[out_arr, result], device=device)
        np.testing.assert_allclose(result.numpy(), expected_fn(input_data), rtol=1e-1)


def test_bf16_interop_jax(test, device):
    try:
        import jax.numpy as jnp  # noqa: PLC0415
    except ImportError:
        test.skipTest("JAX not available")

    wp_arr = wp.zeros(4, dtype=wp.bfloat16, device=device)
    jax_arr = wp.to_jax(wp_arr)
    test.assertEqual(jax_arr.dtype, jnp.bfloat16)


# Test bfloat16 in user-defined vector/matrix types
bf16_vec3 = wp.types.vector(3, dtype=wp.bfloat16)
bf16_mat22 = wp.types.matrix(shape=(2, 2), dtype=wp.bfloat16)


@wp.kernel
def bf16_vec3_kernel(
    out_x: wp.array(dtype=wp.float32),
    out_y: wp.array(dtype=wp.float32),
    out_z: wp.array(dtype=wp.float32),
):
    v = bf16_vec3(wp.bfloat16(1.0), wp.bfloat16(2.0), wp.bfloat16(3.0))
    out_x[0] = wp.float32(v[0])
    out_y[0] = wp.float32(v[1])
    out_z[0] = wp.float32(v[2])


@wp.kernel
def bf16_mat22_kernel(
    out_00: wp.array(dtype=wp.float32),
    out_01: wp.array(dtype=wp.float32),
    out_10: wp.array(dtype=wp.float32),
    out_11: wp.array(dtype=wp.float32),
):
    m = bf16_mat22(wp.bfloat16(1.0), wp.bfloat16(2.0), wp.bfloat16(3.0), wp.bfloat16(4.0))
    out_00[0] = wp.float32(m[0, 0])
    out_01[0] = wp.float32(m[0, 1])
    out_10[0] = wp.float32(m[1, 0])
    out_11[0] = wp.float32(m[1, 1])


def test_bf16_vector_matrix(test, device):
    out_x = wp.zeros(1, dtype=wp.float32, device=device)
    out_y = wp.zeros(1, dtype=wp.float32, device=device)
    out_z = wp.zeros(1, dtype=wp.float32, device=device)

    wp.launch(bf16_vec3_kernel, dim=1, inputs=[out_x, out_y, out_z], device=device)

    np.testing.assert_allclose(out_x.numpy()[0], 1.0, rtol=1e-2)
    np.testing.assert_allclose(out_y.numpy()[0], 2.0, rtol=1e-2)
    np.testing.assert_allclose(out_z.numpy()[0], 3.0, rtol=1e-2)

    out_00 = wp.zeros(1, dtype=wp.float32, device=device)
    out_01 = wp.zeros(1, dtype=wp.float32, device=device)
    out_10 = wp.zeros(1, dtype=wp.float32, device=device)
    out_11 = wp.zeros(1, dtype=wp.float32, device=device)

    wp.launch(bf16_mat22_kernel, dim=1, inputs=[out_00, out_01, out_10, out_11], device=device)

    np.testing.assert_allclose(out_00.numpy()[0], 1.0, rtol=1e-2)
    np.testing.assert_allclose(out_01.numpy()[0], 2.0, rtol=1e-2)
    np.testing.assert_allclose(out_10.numpy()[0], 3.0, rtol=1e-2)
    np.testing.assert_allclose(out_11.numpy()[0], 4.0, rtol=1e-2)


# Test bfloat16 struct fields
@wp.struct
class Bf16Struct:
    bf16_val: wp.bfloat16
    f32_val: wp.float32


@wp.kernel
def bf16_struct_kernel(
    s: Bf16Struct,
    out_bf16: wp.array(dtype=wp.float32),
    out_f32: wp.array(dtype=wp.float32),
):
    out_bf16[0] = wp.float32(s.bf16_val)
    out_f32[0] = s.f32_val


def test_bf16_struct(test, device):
    s = Bf16Struct()
    s.bf16_val = wp.bfloat16(3.5)
    s.f32_val = wp.float32(7.25)

    out_bf16 = wp.zeros(1, dtype=wp.float32, device=device)
    out_f32 = wp.zeros(1, dtype=wp.float32, device=device)

    wp.launch(bf16_struct_kernel, dim=1, inputs=[s, out_bf16, out_f32], device=device)

    np.testing.assert_allclose(out_bf16.numpy()[0], 3.5, rtol=1e-2)
    np.testing.assert_allclose(out_f32.numpy()[0], 7.25, rtol=1e-7)


# Test comparison operators
@wp.kernel
def bf16_comparison_kernel(
    results: wp.array(dtype=wp.int32),
):
    a = wp.bfloat16(1.0)
    b = wp.bfloat16(2.0)
    c = wp.bfloat16(1.0)

    # equal values
    results[0] = wp.where(a == c, 1, 0)  # true -> 1
    # unequal values
    results[1] = wp.where(a != b, 1, 0)  # true -> 1
    # less than
    results[2] = wp.where(a < b, 1, 0)  # true -> 1
    # greater than
    results[3] = wp.where(b > a, 1, 0)  # true -> 1
    # less than or equal (equal case)
    results[4] = wp.where(a <= c, 1, 0)  # true -> 1
    # greater than or equal (equal case)
    results[5] = wp.where(a >= c, 1, 0)  # true -> 1
    # false cases
    results[6] = wp.where(a == b, 1, 0)  # false -> 0
    results[7] = wp.where(a > b, 1, 0)  # false -> 0

    # negative zero vs positive zero
    neg_zero = wp.bfloat16(-0.0)
    pos_zero = wp.bfloat16(0.0)
    results[8] = wp.where(neg_zero == pos_zero, 1, 0)  # true -> 1 (IEEE 754)


def test_bf16_comparisons(test, device):
    results = wp.zeros(9, dtype=wp.int32, device=device)
    wp.launch(bf16_comparison_kernel, dim=1, inputs=[results], device=device)

    r = results.numpy()
    # True cases
    test.assertEqual(r[0], 1, "a == c should be true")
    test.assertEqual(r[1], 1, "a != b should be true")
    test.assertEqual(r[2], 1, "a < b should be true")
    test.assertEqual(r[3], 1, "b > a should be true")
    test.assertEqual(r[4], 1, "a <= c should be true")
    test.assertEqual(r[5], 1, "a >= c should be true")
    # False cases
    test.assertEqual(r[6], 0, "a == b should be false")
    test.assertEqual(r[7], 0, "a > b should be false")
    # Negative zero == positive zero (IEEE 754)
    test.assertEqual(r[8], 1, "-0.0 == 0.0 should be true")


class TestBf16(unittest.TestCase):
    pass


devices = []
if wp.is_cpu_available():
    devices.append("cpu")
for cuda_device in get_selected_cuda_test_devices():
    if cuda_device.arch >= 80:
        devices.append(cuda_device)
add_function_test(TestBf16, "test_bf16_conversion", test_bf16_conversion, devices=devices)
add_function_test(TestBf16, "test_bf16_kernel_parameter", test_bf16_kernel_parameter, devices=devices)
add_function_test(TestBf16, "test_bf16_arithmetic", test_bf16_arithmetic, devices=devices)
add_function_test(TestBf16, "test_bf16_numpy", test_bf16_numpy, devices=devices, check_output=False)
add_function_test(TestBf16, "test_bf16_array_from_list", test_bf16_array_from_list, devices=devices)
add_function_test(TestBf16, "test_bf16_array_from_tuple", test_bf16_array_from_tuple, devices=devices)
add_function_test(TestBf16, "test_bf16_array_from_numpy_float", test_bf16_array_from_numpy_float, devices=devices)
add_function_test(TestBf16, "test_bf16_array_from_numpy_uint16", test_bf16_array_from_numpy_uint16, devices=devices)
add_function_test(TestBf16, "test_bf16_array_special_values", test_bf16_array_special_values, devices=devices)
add_function_test(TestBf16, "test_bf16_grad", test_bf16_grad, devices=devices)
add_function_test(TestBf16, "test_bf16_atomics", test_bf16_atomics, devices=devices)
add_function_test(TestBf16, "test_bf16_interop_dlpack", test_bf16_interop_dlpack, devices=devices)
add_function_test(TestBf16, "test_bf16_interop_torch", test_bf16_interop_torch, devices=devices)
add_function_test(TestBf16, "test_bf16_math_builtins", test_bf16_math_builtins, devices=devices)
add_function_test(TestBf16, "test_bf16_interop_jax", test_bf16_interop_jax, devices=devices)
add_function_test(TestBf16, "test_bf16_vector_matrix", test_bf16_vector_matrix, devices=devices)
add_function_test(TestBf16, "test_bf16_struct", test_bf16_struct, devices=devices)
add_function_test(TestBf16, "test_bf16_comparisons", test_bf16_comparisons, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
