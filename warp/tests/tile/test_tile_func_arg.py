# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for passing tiles of various storage types to @wp.func."""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

BLOCK_DIM = 64
TILE_M = 4
TILE_N = 4


# ---- Helper functions used by tests ----


@wp.func
def scale_tile(t: wp.tile[float, TILE_M, TILE_N]):
    """Multiply every element of a tile by 2."""
    return t * 2.0


@wp.func
def pass_through_tile(t: wp.tile[float, TILE_M, TILE_N]):
    """Return a tile unchanged (tests direct return of parameter)."""
    return t


@wp.func
def add_tiles(
    a: wp.tile[float, TILE_M, TILE_N],
    b: wp.tile[float, TILE_M, TILE_N],
):
    """Add two tiles element-wise."""
    return a + b


@wp.func
def scale_then_pass_through(t: wp.tile[float, TILE_M, TILE_N]):
    """Chain two @wp.func calls: scale then pass through."""
    s = scale_tile(t)
    return pass_through_tile(s)


@wp.func
def inplace_add_func(
    a: wp.tile[float, TILE_M, TILE_N],
    b: wp.tile[float, TILE_M, TILE_N],
):
    """Modify a in place (a += b) and return a."""
    a += b
    return a


@wp.func
def modify_tile_no_return(t: wp.tile[float, TILE_M, TILE_N]):
    """Write into a tile in place without returning it. Used for both register and shared tiles."""
    t += wp.tile_ones(dtype=float, shape=(TILE_M, TILE_N), storage="register") * 5.0


@wp.func
def matmul_func(a: wp.tile[float, TILE_M, TILE_N], b: wp.tile[float, TILE_N, TILE_M]):
    """Multiply two tiles; ``tile_matmul()`` needs its operands in shared memory."""
    return wp.tile_matmul(a, b)


@wp.func
def matmul_outer_func(a: wp.tile[float, TILE_M, TILE_N], b: wp.tile[float, TILE_N, TILE_M]):
    """Forward the tile parameters to another function that needs them in shared memory."""
    return matmul_func(a, b)


@wp.func
def read_elements_func(t: wp.tile[float, TILE_M, TILE_N]):
    """Combine two elements of a tile; reading elements needs the tile in shared memory."""
    return t[1, 2] * 2.0 + t[3, 0]


# ---- Forward tests ----


def test_shared_tile_func_arg(test, device):
    """Pass a shared tile into a @wp.func, get a scaled copy back."""

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d[float], out: wp.array2d[float]):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        r = scale_tile(t)
        wp.tile_store(out, r)

    inp = wp.array(np.ones((TILE_M, TILE_N), dtype=np.float32) * 3.0, device=device)
    out = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=BLOCK_DIM, device=device)

    expected = np.ones((TILE_M, TILE_N), dtype=np.float32) * 6.0
    np.testing.assert_array_equal(out.numpy(), expected)


def test_register_tile_func_arg(test, device):
    """Pass a register tile into a @wp.func, get a scaled copy back."""

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d[float], out: wp.array2d[float]):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="register")
        r = scale_tile(t)
        wp.tile_store(out, r)

    inp = wp.array(np.ones((TILE_M, TILE_N), dtype=np.float32) * 5.0, device=device)
    out = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=BLOCK_DIM, device=device)

    expected = np.ones((TILE_M, TILE_N), dtype=np.float32) * 10.0
    np.testing.assert_array_equal(out.numpy(), expected)


def test_pass_through_shared_tile(test, device):
    """Return a shared tile directly from a @wp.func (tests cross-storage assignment)."""

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d[float], out: wp.array2d[float]):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        r = pass_through_tile(t)
        wp.tile_store(out, r)

    inp_data = np.arange(TILE_M * TILE_N, dtype=np.float32).reshape(TILE_M, TILE_N)
    inp = wp.array(inp_data, device=device)
    out = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=BLOCK_DIM, device=device)

    np.testing.assert_array_equal(out.numpy(), inp_data)


def test_mixed_storage_func_args(test, device):
    """Pass one shared tile and one register tile to the same @wp.func."""

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        input_a: wp.array2d[float],
        input_b: wp.array2d[float],
        out: wp.array2d[float],
    ):
        i = wp.tid()
        a = wp.tile_load(input_a, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        b = wp.tile_load(input_b, shape=(TILE_M, TILE_N), offset=(0, 0), storage="register")
        r = add_tiles(a, b)
        wp.tile_store(out, r)

    inp_a = wp.array(np.ones((TILE_M, TILE_N), dtype=np.float32) * 2.0, device=device)
    inp_b = wp.array(np.ones((TILE_M, TILE_N), dtype=np.float32) * 3.0, device=device)
    out = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp_a, inp_b, out], block_dim=BLOCK_DIM, device=device)

    expected = np.ones((TILE_M, TILE_N), dtype=np.float32) * 5.0
    np.testing.assert_array_equal(out.numpy(), expected)


def test_nested_func_calls(test, device):
    """Chain two @wp.func calls with tile args (tests template propagation)."""

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d[float], out: wp.array2d[float]):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        r = scale_then_pass_through(t)
        wp.tile_store(out, r)

    inp = wp.array(np.ones((TILE_M, TILE_N), dtype=np.float32) * 4.0, device=device)
    out = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=BLOCK_DIM, device=device)

    expected = np.ones((TILE_M, TILE_N), dtype=np.float32) * 8.0
    np.testing.assert_array_equal(out.numpy(), expected)


def test_shared_tile_inplace_no_return(test, device):
    """Modify a shared tile inside a @wp.func without returning it.

    The caller should see the mutation because shared tiles live in shared memory.
    """

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d[float], out: wp.array2d[float]):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        modify_tile_no_return(t)
        wp.tile_store(out, t)

    inp = wp.array(np.ones((TILE_M, TILE_N), dtype=np.float32) * 3.0, device=device)
    out = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=BLOCK_DIM, device=device)

    expected = np.ones((TILE_M, TILE_N), dtype=np.float32) * 8.0  # 3.0 + 5.0
    np.testing.assert_array_equal(out.numpy(), expected)


def test_register_tile_inplace_no_return(test, device):
    """Modify a register tile inside a @wp.func without returning it.

    Register tiles are now passed by reference (previously by value), so the
    caller should see the in-place mutation.
    """

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d[float], out: wp.array2d[float]):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="register")
        modify_tile_no_return(t)
        wp.tile_store(out, t)

    inp = wp.array(np.ones((TILE_M, TILE_N), dtype=np.float32) * 3.0, device=device)
    out = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=BLOCK_DIM, device=device)

    expected = np.ones((TILE_M, TILE_N), dtype=np.float32) * 8.0  # 3.0 + 5.0
    np.testing.assert_array_equal(out.numpy(), expected)


# ---- Gradient tests ----


def test_nested_func_calls_grad(test, device):
    """Test gradient propagation through chained tile functions.

    f(x) = sum(2*x), df/dx = 2.0 for every element.
    """

    @wp.kernel(module="unique")
    def compute(input: wp.array2d[float], loss: wp.array1d[float]):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        r = scale_then_pass_through(t)
        s = wp.tile_sum(r)
        wp.tile_store(loss, s)

    inp = wp.array(np.full((TILE_M, TILE_N), 4.0, dtype=np.float32), device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[inp, loss], block_dim=BLOCK_DIM, device=device)

    tape.backward(loss=loss)
    np.testing.assert_allclose(inp.grad.numpy(), np.full((TILE_M, TILE_N), 2.0, dtype=np.float32), rtol=1e-5)


def test_shared_tile_func_grad(test, device):
    """Test gradient propagation through a function with a shared tile argument.

    f(x) = sum(2*x), df/dx = 2.0 for every element.
    """

    @wp.kernel(module="unique")
    def compute(input: wp.array2d[float], loss: wp.array1d[float]):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        r = scale_tile(t)
        s = wp.tile_sum(r)
        wp.tile_store(loss, s)

    inp = wp.array(np.full((TILE_M, TILE_N), 3.0, dtype=np.float32), device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[inp, loss], block_dim=BLOCK_DIM, device=device)

    tape.backward(loss=loss)
    np.testing.assert_allclose(inp.grad.numpy(), np.full((TILE_M, TILE_N), 2.0, dtype=np.float32), rtol=1e-5)


def test_register_tile_func_grad(test, device):
    """Test gradient propagation through a function with a register tile argument.

    f(x) = sum(2*x), df/dx = 2.0 for every element.
    """

    @wp.kernel(module="unique")
    def compute(input: wp.array2d[float], loss: wp.array1d[float]):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="register")
        r = scale_tile(t)
        s = wp.tile_sum(r)
        wp.tile_store(loss, s)

    inp = wp.array(np.full((TILE_M, TILE_N), 3.0, dtype=np.float32), device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[inp, loss], block_dim=BLOCK_DIM, device=device)

    tape.backward(loss=loss)
    np.testing.assert_allclose(inp.grad.numpy(), np.full((TILE_M, TILE_N), 2.0, dtype=np.float32), rtol=1e-5)


def test_mixed_storage_func_grad(test, device):
    """Test gradient propagation through mixed tile storage.

    f(a, b) = sum(a + b), df/da = df/db = 1.0 for every element.
    """

    @wp.kernel(module="unique")
    def compute(
        input_a: wp.array2d[float],
        input_b: wp.array2d[float],
        loss: wp.array1d[float],
    ):
        i = wp.tid()
        a = wp.tile_load(input_a, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        b = wp.tile_load(input_b, shape=(TILE_M, TILE_N), offset=(0, 0), storage="register")
        r = add_tiles(a, b)
        s = wp.tile_sum(r)
        wp.tile_store(loss, s)

    inp_a = wp.array(np.full((TILE_M, TILE_N), 2.0, dtype=np.float32), device=device, requires_grad=True)
    inp_b = wp.array(np.full((TILE_M, TILE_N), 3.0, dtype=np.float32), device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[inp_a, inp_b, loss], block_dim=BLOCK_DIM, device=device)

    tape.backward(loss=loss)
    expected_grad = np.ones((TILE_M, TILE_N), dtype=np.float32)
    np.testing.assert_allclose(inp_a.grad.numpy(), expected_grad, rtol=1e-5)
    np.testing.assert_allclose(inp_b.grad.numpy(), expected_grad, rtol=1e-5)


def test_inplace_modify_func_grad(test, device):
    """Test gradient propagation through an in-place tile update.

    f(a, b) = sum(a + b) computed via a += b; return a.
    df/da = df/db = 1.0 for every element.
    """

    @wp.kernel(module="unique")
    def compute(
        input_a: wp.array2d[float],
        input_b: wp.array2d[float],
        loss: wp.array1d[float],
    ):
        i = wp.tid()
        a = wp.tile_load(input_a, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        b = wp.tile_load(input_b, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        r = inplace_add_func(a, b)
        s = wp.tile_sum(r)
        wp.tile_store(loss, s)

    inp_a = wp.array(np.full((TILE_M, TILE_N), 2.0, dtype=np.float32), device=device, requires_grad=True)
    inp_b = wp.array(np.full((TILE_M, TILE_N), 3.0, dtype=np.float32), device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[inp_a, inp_b, loss], block_dim=BLOCK_DIM, device=device)

    tape.backward(loss=loss)
    expected_grad = np.ones((TILE_M, TILE_N), dtype=np.float32)
    np.testing.assert_allclose(inp_a.grad.numpy(), expected_grad, rtol=1e-5)
    np.testing.assert_allclose(inp_b.grad.numpy(), expected_grad, rtol=1e-5)


# ---- @wp.func_native tests ----


def test_native_func_shared_tile(test, device):
    """Pass a shared tile to a @wp.func_native and read from it."""

    snippet = """
    // Extract element [0,0] of the tile and write to out[0]
    auto val = tile_extract(t, 0, 0);
    if (threadIdx.x == 0) {
        out[0] = val;
    }
    """

    @wp.func_native(snippet)
    def read_tile_native(t: wp.tile[float, TILE_M, TILE_N], out: wp.array[float]): ...

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d[float], out: wp.array[float]):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        read_tile_native(t, out)

    inp = wp.array(np.ones((TILE_M, TILE_N), dtype=np.float32) * 7.0, device=device)
    out = wp.zeros(1, dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=BLOCK_DIM, device=device)

    np.testing.assert_allclose(out.numpy()[0], 7.0)


def test_native_func_register_tile(test, device):
    """Pass a register tile to a @wp.func_native and read from it."""

    snippet = """
    auto val = tile_extract(t, 0, 0);
    if (threadIdx.x == 0) {
        out[0] = val;
    }
    """

    @wp.func_native(snippet)
    def read_tile_native(t: wp.tile[float, TILE_M, TILE_N], out: wp.array[float]): ...

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d[float], out: wp.array[float]):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="register")
        read_tile_native(t, out)

    inp = wp.array(np.ones((TILE_M, TILE_N), dtype=np.float32) * 3.0, device=device)
    out = wp.zeros(1, dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=BLOCK_DIM, device=device)

    np.testing.assert_allclose(out.numpy()[0], 3.0)


def test_native_func_inplace_modify_tile(test, device):
    """Modify a shared tile in-place inside a @wp.func_native via reference."""

    snippet = """
    // Add 10.0 to element [0,0] of the tile via shared memory pointer
    auto& data = t.data;
    if (threadIdx.x == 0) {
        data.ptr[0] += wp::float32(10.0);
    }
    __syncthreads();
    """

    @wp.func_native(snippet)
    def modify_tile_native(t: wp.tile[float, TILE_M, TILE_N]): ...

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d[float], out: wp.array2d[float]):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        modify_tile_native(t)
        wp.tile_store(out, t)

    inp = wp.array(np.ones((TILE_M, TILE_N), dtype=np.float32) * 5.0, device=device)
    out = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=BLOCK_DIM, device=device)

    result = out.numpy()
    test.assertAlmostEqual(result[0, 0], 15.0)  # 5.0 + 10.0
    test.assertAlmostEqual(result[0, 1], 5.0)  # unchanged


# ---- Builtins that need shared memory tiles, called on register tile parameters ----


def test_register_tile_func_matmul(test, device):
    """Multiply register tiles in a user function, called directly and through another user function."""

    @wp.kernel(module="unique")
    def compute(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float], d: wp.array2d[float]):
        # Load separate register tiles for each call so that each one is checked on its own
        a0 = wp.tile_load(a, shape=(TILE_M, TILE_N), storage="register")
        b0 = wp.tile_load(b, shape=(TILE_N, TILE_M), storage="register")
        wp.tile_store(c, matmul_func(a0, b0))

        a1 = wp.tile_load(a, shape=(TILE_M, TILE_N), storage="register")
        b1 = wp.tile_load(b, shape=(TILE_N, TILE_M), storage="register")
        wp.tile_store(d, matmul_outer_func(a1, b1))

    rng = np.random.default_rng(42)
    a_np = rng.standard_normal((TILE_M, TILE_N)).astype(np.float32)
    b_np = rng.standard_normal((TILE_N, TILE_M)).astype(np.float32)
    adj_c = rng.standard_normal((TILE_M, TILE_M)).astype(np.float32)
    adj_d = rng.standard_normal((TILE_M, TILE_M)).astype(np.float32)

    a = wp.array(a_np, requires_grad=True, device=device)
    b = wp.array(b_np, requires_grad=True, device=device)
    c = wp.zeros((TILE_M, TILE_M), dtype=float, requires_grad=True, device=device)
    d = wp.zeros((TILE_M, TILE_M), dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[a, b], outputs=[c, d], block_dim=BLOCK_DIM, device=device)

    np.testing.assert_allclose(c.numpy(), a_np @ b_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(d.numpy(), a_np @ b_np, rtol=1e-5, atol=1e-5)

    tape.backward(grads={c: wp.array(adj_c, device=device), d: wp.array(adj_d, device=device)})
    adj_out = adj_c + adj_d
    np.testing.assert_allclose(a.grad.numpy(), adj_out @ b_np.T, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(b.grad.numpy(), a_np.T @ adj_out, rtol=1e-5, atol=1e-5)


def test_register_tile_func_element_read(test, device):
    """Read elements of a register tile in a user function and check the gradient."""

    @wp.kernel(module="unique")
    def compute(x: wp.array2d[float], out: wp.array1d[float]):
        _, lane = wp.tid()
        t = wp.tile_load(x, shape=(TILE_M, TILE_N), storage="register")
        s = read_elements_func(t)
        # Every thread holds the result; store it from one thread so that only one passes a gradient back
        if lane == 0:
            out[0] = s

    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((TILE_M, TILE_N)).astype(np.float32)

    x = wp.array(x_np, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[x], outputs=[out], block_dim=BLOCK_DIM, device=device)

    np.testing.assert_allclose(out.numpy(), [x_np[1, 2] * 2.0 + x_np[3, 0]], rtol=1e-6)

    tape.backward(loss=out)
    expected_grad = np.zeros_like(x_np)
    expected_grad[1, 2] = 2.0
    expected_grad[3, 0] = 1.0
    np.testing.assert_allclose(x.grad.numpy(), expected_grad)


devices = get_test_devices()


class TestTileFuncArg(unittest.TestCase):
    pass


add_function_test(TestTileFuncArg, "test_shared_tile_func_arg", test_shared_tile_func_arg, devices=devices)
add_function_test(TestTileFuncArg, "test_register_tile_func_arg", test_register_tile_func_arg, devices=devices)
add_function_test(TestTileFuncArg, "test_pass_through_shared_tile", test_pass_through_shared_tile, devices=devices)
add_function_test(TestTileFuncArg, "test_mixed_storage_func_args", test_mixed_storage_func_args, devices=devices)
add_function_test(TestTileFuncArg, "test_nested_func_calls", test_nested_func_calls, devices=devices)
add_function_test(
    TestTileFuncArg, "test_shared_tile_inplace_no_return", test_shared_tile_inplace_no_return, devices=devices
)
add_function_test(
    TestTileFuncArg, "test_register_tile_inplace_no_return", test_register_tile_inplace_no_return, devices=devices
)
add_function_test(TestTileFuncArg, "test_shared_tile_func_grad", test_shared_tile_func_grad, devices=devices)
add_function_test(TestTileFuncArg, "test_register_tile_func_grad", test_register_tile_func_grad, devices=devices)
add_function_test(TestTileFuncArg, "test_mixed_storage_func_grad", test_mixed_storage_func_grad, devices=devices)
add_function_test(TestTileFuncArg, "test_nested_func_calls_grad", test_nested_func_calls_grad, devices=devices)
add_function_test(TestTileFuncArg, "test_inplace_modify_func_grad", test_inplace_modify_func_grad, devices=devices)
add_function_test(TestTileFuncArg, "test_register_tile_func_matmul", test_register_tile_func_matmul, devices=devices)
add_function_test(
    TestTileFuncArg,
    "test_register_tile_func_element_read",
    test_register_tile_func_element_read,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestTileFuncArg, "test_native_func_shared_tile", test_native_func_shared_tile, devices=get_cuda_test_devices()
)
add_function_test(
    TestTileFuncArg, "test_native_func_register_tile", test_native_func_register_tile, devices=get_cuda_test_devices()
)
add_function_test(
    TestTileFuncArg,
    "test_native_func_inplace_modify_tile",
    test_native_func_inplace_modify_tile,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
