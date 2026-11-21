# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import ctypes
import io
import unittest
from typing import Any

import numpy as np

import warp as wp
import warp.sparse
from warp._src.types import array_t, check_array_shape, largearray_t
from warp.tests.unittest_utils import *


def _fake_large(device, n=2**31 + 16, dtype=wp.int8):
    """Build a large 1-D array over a small buffer.

    Only safe for code paths that never dereference past the buffer, such as metadata checks
    or calls that are expected to raise. The buffer holds at least one element of the largest
    dtype used by the tests.
    """
    backing = wp.zeros(16, dtype=wp.int8, device=device)
    return backing, wp.array(ptr=backing.ptr, shape=(n,), dtype=dtype, device=device)


def test_large_shape_accepted(test, device):
    backing, a = _fake_large(device)
    test.assertEqual(a.shape, (2**31 + 16,))
    test.assertEqual(a.size, 2**31 + 16)
    test.assertTrue(a._is_large)
    test.assertFalse(backing._is_large)


def test_shape_limits(test, device):
    backing = wp.zeros(16, dtype=wp.int8, device=device)
    with test.assertRaisesRegex(ValueError, "Array shapes"):
        wp.array(ptr=backing.ptr, shape=(2**63,), dtype=wp.int8, device=device)
    with test.assertRaisesRegex(ValueError, "Array shapes"):
        wp.array(ptr=backing.ptr, shape=(2**31, 1), dtype=wp.int8, device=device)


def _check_shape_boundaries(test):
    device = "cpu"
    backing = wp.zeros(16, dtype=wp.int8, device=device)

    # largest 1-D length is accepted
    a = wp.array(ptr=backing.ptr, shape=(2**63 - 1,), dtype=wp.int8, device=device)
    test.assertEqual(a.shape, (2**63 - 1,))
    test.assertTrue(a._is_large)

    # largest length addressable by the 32-bit descriptor is not "large"
    b = wp.array(ptr=backing.ptr, shape=(2**31 - 1,), dtype=wp.int8, device=device)
    test.assertEqual(b.shape, (2**31 - 1,))
    test.assertFalse(b._is_large)

    # multi-dimensional shapes keep the 32-bit per-dimension limit
    check_array_shape((2**31 - 1, 1))


def test_large_rejected_by_32bit_paths(test, device):
    _backing, a = _fake_large(device)
    for call in (a.__ctype__, lambda: a[0:4], lambda: a.reshape((2**30 + 8, 2)), lambda: a.view(wp.uint8)):
        with test.assertRaisesRegex(ValueError, r"more than 2\*\*31 - 1 elements \(got 2147483664\)"):
            call()
    # integer indexing reports what it reports for any 1-D array
    with test.assertRaisesRegex(RuntimeError, "Item indexing is not supported"):
        a[0]


def test_large_rejected_after_descriptor_reset(test, device):
    # Changing the gradient discards the cached descriptor; the large-array guard must survive it.
    _backing, a = _fake_large(device)
    _grad_backing, g = _fake_large(device)
    a.grad = g
    a.retain_grad = True
    for call in (a.__ctype__, a.grad.__ctype__, lambda: a.reshape((2**30 + 8, 2)), lambda: a.view(wp.uint8)):
        with test.assertRaisesRegex(ValueError, r"more than 2\*\*31 - 1 elements \(got 2147483664\)"):
            call()
    test.assertEqual(a.__ctype64__().shape[0], 2**31 + 16)


class _SuperCtypeSubclass(wp.array):
    def __ctype__(self):
        return super().__ctype__()


def test_large_rejected_through_class_method(test, device):
    # The check lives in the class method, so it holds however the method is reached.
    _backing, a = _fake_large(device)
    for _ in range(2):
        with test.assertRaisesRegex(ValueError, r"more than 2\*\*31 - 1 elements \(got 2147483664\)"):
            wp.array.__ctype__(a)

    sub = _SuperCtypeSubclass(ptr=_backing.ptr, shape=(2**31 + 16,), dtype=wp.int8, device=device)
    with test.assertRaisesRegex(ValueError, r"more than 2\*\*31 - 1 elements"):
        sub.__ctype__()


class _OwnDescriptorSubclass(wp.array):
    """Builds its own descriptor from the current shape, without a size check."""

    def __ctype__(self):
        return array_t(self.ptr, 0, self.ndim, self.shape, self.strides)


def test_large_overriding_subclass_rejected(test, device):
    backing = wp.zeros(16, dtype=wp.int8, device=device)
    big = _OwnDescriptorSubclass(ptr=backing.ptr, shape=(2**31 + 16,), dtype=wp.int8, device=device)
    pattern = r"more than 2\*\*31 - 1 elements \(got 2147483664\)"

    with test.assertRaisesRegex(RuntimeError, r"kernel '_noop32'.*argument 'x'.*" + pattern):
        wp.launch(_noop32, dim=1, inputs=[big], device=device)

    s = _HasArray()
    with test.assertRaisesRegex(TypeError, r"Struct field 'a'.*" + pattern):
        s.a = big

    # non-contiguous copy takes the descriptors of both arrays
    strided = _OwnDescriptorSubclass(ptr=backing.ptr, shape=(2**31 + 16,), strides=(2,), dtype=wp.int8, device=device)
    with test.assertRaisesRegex(ValueError, pattern):
        wp.copy(strided, strided)

    # a small instance of the same subclass keeps working
    small = _OwnDescriptorSubclass(ptr=backing.ptr, shape=(8,), dtype=wp.int8, device=device)
    wp.launch(_noop32, dim=1, inputs=[small], device=device)
    s.a = small
    test.assertEqual(s.a.shape, (8,))
    wp.synchronize_device(device)


def test_large_subclass_still_accepted_by_largearray(test, device):
    # A large subclass that does not override __ctype__ is fine for wp.largearray; one that does is rejected.
    backing = wp.zeros(16, dtype=wp.int8, device=device)
    plain = _PlainSubclass(ptr=backing.ptr, shape=(2**31 + 16,), dtype=wp.int8, device=device)
    wp.launch(_noop_large_int8, dim=1, inputs=[plain], device=device)
    own = _OwnDescriptorSubclass(ptr=backing.ptr, shape=(2**31 + 16,), dtype=wp.int8, device=device)
    with test.assertRaisesRegex(RuntimeError, "custom descriptor are not supported for wp.largearray"):
        wp.launch(_noop_large_int8, dim=1, inputs=[own], device=device)
    wp.synchronize_device(device)


def test_stale_descriptor_after_shape_reassignment(test, device):
    # Documented limit, not behavior to rely on: shape is a plain attribute, so a descriptor cached
    # while the array was small is still returned after shape is reassigned to a large value.
    backing = wp.zeros(16, dtype=wp.int8, device=device)
    m = wp.array(ptr=backing.ptr, shape=(64,), dtype=wp.int8, device=device)
    c = m.__ctype__()
    m.shape = (2**31 + 16,)
    test.assertIs(m.__ctype__(), c)
    test.assertEqual(c.shape[0], 64)


def test_large_rejected_after_shape_reassignment(test, device):
    # The check reads the current shape, so it sees a shape assigned after construction.
    backing = wp.zeros(16, dtype=wp.int8, device=device)
    m = wp.array(ptr=backing.ptr, shape=(64,), dtype=wp.int8, device=device)
    m.shape = (2**31 + 16,)
    with test.assertRaisesRegex(ValueError, r"more than 2\*\*31 - 1 elements \(got 2147483664\)"):
        m.__ctype__()


def test_small_descriptor_cached_and_rebuilt(test, device):
    a = wp.zeros(8, dtype=float, requires_grad=True, device=device)
    c = a.__ctype__()
    test.assertIs(a.__ctype__(), c)
    test.assertEqual(c.shape[0], 8)
    test.assertEqual(c.grad, a.grad.ptr)

    a.grad = wp.zeros(8, dtype=float, device=device)
    c2 = a.__ctype__()
    test.assertIsNot(c2, c)
    test.assertEqual(c2.grad, a.grad.ptr)
    test.assertIs(a.__ctype__(), c2)
    wp.synchronize_device(device)


@wp.struct
class _HasArray:
    a: wp.array[wp.int8]


@wp.struct
class _HasIndexedArray:
    a: wp.indexedarray[wp.int8]


def test_large_rejected_elsewhere(test, device):
    backing, a = _fake_large(device)
    _fbacking, f = _fake_large(device, dtype=wp.float32)
    _ibacking, big_indices = _fake_large(device, dtype=wp.int32)
    small = wp.zeros(4, dtype=wp.int8, device=device)
    small_indices = wp.zeros(4, dtype=int, device=device)
    pattern = r"more than 2\*\*31 - 1 elements"

    # struct field assignment
    s = _HasArray()
    with test.assertRaisesRegex(ValueError, pattern):
        s.a = a

    # indexed arrays: large data array and large index array
    with test.assertRaisesRegex(ValueError, pattern):
        wp.indexedarray(a, [small_indices])
    with test.assertRaisesRegex(ValueError, pattern):
        wp.indexedarray(small, [big_indices])
    si = _HasIndexedArray()
    with test.assertRaisesRegex(ValueError, pattern):
        si.a = wp.indexedarray(a, [small_indices])

    # array descriptors built from interface or tensor metadata
    interface = {"data": (backing.ptr, False), "shape": (2**31 + 16,), "typestr": "|i1", "strides": None}
    with test.assertRaisesRegex(ValueError, pattern):
        wp._src.types.array_ctype_from_interface(interface)
    with test.assertRaisesRegex(ValueError, pattern):
        wp._src.types.check_descriptor_shape((2**31 + 16,))
    with test.assertRaisesRegex(ValueError, "Array shapes must not exceed"):
        wp._src.types.check_descriptor_shape((2**31, 2))
    wp._src.types.check_descriptor_shape((2**31 - 1, 2))

    # a large array raises on every call, never serving a cached descriptor
    for _ in range(2):
        with test.assertRaisesRegex(ValueError, pattern):
            a.__ctype__()

    # utilities that pass sizes to native code as C ints
    with test.assertRaisesRegex(ValueError, pattern):
        wp.utils.array_sum(f)
    with test.assertRaisesRegex(ValueError, pattern):
        wp.utils.array_inner(f, f)
    with test.assertRaisesRegex(ValueError, pattern):
        wp.utils.array_scan(f, f)
    with test.assertRaisesRegex(ValueError, pattern):
        wp.utils.array_cast(f, a)
    with test.assertRaisesRegex(ValueError, pattern):
        wp.utils.radix_sort_pairs(f, f, 4)
    with test.assertRaisesRegex(ValueError, pattern):
        wp.utils.segmented_sort_pairs(f, f, 4, big_indices)
    with test.assertRaisesRegex(ValueError, pattern):
        wp.utils.runlength_encode(f, f, big_indices)

    # BVH construction passes the array length as a C int
    with test.assertRaisesRegex(ValueError, pattern):
        wp.Bvh(_fake_large(device, dtype=wp.vec3)[1], _fake_large(device, dtype=wp.vec3)[1])

    # sparse triplet assembly passes the triplet count as a C int
    bsr = wp.sparse.bsr_zeros(4, 4, wp.float32, device=device)
    with test.assertRaisesRegex(ValueError, pattern):
        wp.sparse.bsr_set_from_triplets(bsr, big_indices, big_indices)

    wp.synchronize_device(device)


def test_large_rejected_by_volume(test, device):
    # The native Volume builders take the point count as a C int; each entry point raises first.
    _backing, big = _fake_large(device, dtype=wp.vec3i)
    small = wp.array([[0, 0, 0], [8, 8, 8]], dtype=wp.vec3i, device=device)
    _mbacking, big_mask = _fake_large(device, dtype=wp.int32)
    pattern = r"more than 2\*\*31 - 1 elements \(got 2147483664\)"

    with test.assertRaisesRegex(ValueError, r"wp\.Volume\.allocate_by_tiles\(\) .*" + pattern):
        wp.Volume.allocate_by_tiles(big, voxel_size=0.1, device=device)
    with test.assertRaisesRegex(ValueError, r"wp\.Volume\.allocate_by_voxels\(\) .*" + pattern):
        wp.Volume.allocate_by_voxels(big, voxel_size=0.1, device=device)
    with test.assertRaisesRegex(ValueError, r"wp\.Volume point_mask .*" + pattern):
        wp.Volume.allocate_by_tiles(small, voxel_size=0.1, point_mask=big_mask, device=device)

    for allocate in (wp.Volume.allocate_by_tiles, wp.Volume.allocate_by_voxels):
        volume = allocate(small, voxel_size=0.1, rebuildable=True, device=device)
        with test.assertRaisesRegex(ValueError, r"wp\.Volume\.rebuild\(\) .*" + pattern):
            volume.rebuild(big)
    wp.synchronize_device(device)


def test_large_rejected_by_mesh_and_hash_grid(test, device):
    backing = wp.zeros(16, dtype=wp.int8, device=device)
    pattern = r"more than 2\*\*31 - 1 elements \(got 2147483664\)"
    indices = wp.array([0, 0, 0], dtype=wp.int32, device=device)
    grid = wp.HashGrid(4, 4, 4, device=device)
    # A plain array, and a subclass whose own __ctype__() would wrap the length without a check.
    for cls in (wp.array, _OwnDescriptorSubclass):
        big = cls(ptr=backing.ptr, shape=(2**31 + 16,), dtype=wp.vec3, device=device)
        with test.subTest(cls=cls.__name__):
            with test.assertRaisesRegex(ValueError, r"wp\.Mesh .*" + pattern):
                wp.Mesh(points=big, indices=indices)
            with test.assertRaisesRegex(ValueError, r"wp\.HashGrid\.build\(\) .*" + pattern):
                grid.build(big, radius=1.0)
    with test.assertRaisesRegex(ValueError, r"supports at most 2\*\*31 - 1 points, got 2147483664"):
        grid.reserve(2**31 + 16)
    wp.synchronize_device(device)


def test_large_noncontiguous_fill_and_copy(test, device):
    backing = wp.zeros(16, dtype=wp.int8, device=device)
    pattern = r"more than 2\*\*31 - 1 elements \(got 2147483664\)"
    for cls in (wp.array, _OwnDescriptorSubclass):
        strided = cls(ptr=backing.ptr, shape=(2**31 + 16,), strides=(2,), dtype=wp.int8, device=device)
        test.assertFalse(strided.is_contiguous)
        with test.subTest(cls=cls.__name__):
            with test.assertRaisesRegex(ValueError, pattern):
                strided.fill_(1)
            with test.assertRaisesRegex(ValueError, pattern):
                strided.zero_()
            with test.assertRaisesRegex(ValueError, pattern):
                wp.copy(strided, strided)
    np.testing.assert_array_equal(backing.numpy(), np.zeros(16, dtype=np.int8))


def test_large_copy_on_tape_rejected(test, device):
    # The decision runs where wp.copy() records onto a tape, after the copy itself, so it is
    # tested directly: a real copy of a fake-large array would read past its buffer.
    _backing, big = _fake_large(device, dtype=wp.float32)
    _gbacking, big_grad = _fake_large(device, dtype=wp.float32)
    big.grad = big_grad
    small = wp.zeros(4, dtype=float, device=device, requires_grad=True)
    pattern = r"wp\.copy\(\) of an array with more than 2\*\*31 - 1 elements \(got 2147483664\) cannot be recorded"
    for dest, src in ((small, big), (big, small)):
        with test.assertRaisesRegex(ValueError, pattern):
            wp._src.context._reject_large_tape_copy(dest, src)
    wp._src.context._reject_large_tape_copy(small, small)

    # copies of small arrays are still recorded and differentiated
    src = wp.array([1.0, 2.0, 3.0, 4.0], dtype=float, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.copy(small, src)
    tape.backward(grads={small: wp.ones_like(small)})
    np.testing.assert_array_equal(src.grad.numpy(), np.ones(4))


def test_bsr_transpose_nnz_limit(test, device):
    # The transpose uses a scratch map of 2 * nnz C ints. ``nnz`` is a host-side upper bound,
    # so setting it is enough to reach the check, which runs before anything is allocated.
    for topology in ("compact", "padded"):
        src = wp.sparse.bsr_diag(wp.ones(4, dtype=float, device=device))
        dest = wp.sparse.bsr_diag(wp.ones(4, dtype=float, device=device))
        src.nnz = 2**30
        with test.subTest(topology=topology):
            with test.assertRaisesRegex(ValueError, r"at most 1073741823 non-zero blocks, got 1073741824"):
                wp.sparse.bsr_set_transpose(dest, src, topology=topology)
    wp.synchronize_device(device)


def test_large_rejected_by_texture(test, device):
    _backing, big = _fake_large(device, dtype=wp.float32)
    with test.assertRaisesRegex(ValueError, r"Texture dimensions must not exceed 2\*\*31 - 1, got width=2147483664"):
        wp.Texture1D(big, device=device)


def _check_large_rejected_by_graph_coloring(test):
    device = "cpu"  # graph coloring runs on the CPU only
    _backing, big = _fake_large(device, dtype=wp.int32)
    edges = wp.array([[0, 1]], dtype=wp.int32, device=device)
    pattern = r"more than 2\*\*31 - 1 elements \(got 2147483664\)"
    with test.assertRaisesRegex(ValueError, r"graph_coloring_assign\(\) .*" + pattern):
        wp.utils.graph_coloring_assign(edges, big)
    with test.assertRaisesRegex(ValueError, r"graph_coloring_balance\(\) .*" + pattern):
        wp.utils.graph_coloring_balance(edges, big, 2, 1.1)


@wp.kernel
def _noop_large(x: wp.largearray[float]):
    pass


@wp.kernel
def _noop32(x: wp.array[wp.int8]):
    pass


@wp.kernel
def _mixed_large(x: wp.largearray[float], y: wp.array[float], n: int):
    y[0] = float(n)


@wp.kernel
def _generic_noop(x: Any):
    pass


_generic_noop_large = wp.overload(_generic_noop, {"x": wp.largearray[float]})
_generic_noop32 = wp.overload(_generic_noop, {"x": wp.array[float]})


def _check_annotation(test):
    test.assertIn("largearray", repr(wp.largearray[float]))
    test.assertEqual(wp.largearray[float].ndim, 1)
    test.assertEqual(wp.largearray[float].dtype, wp.float32)
    test.assertNotEqual(wp.largearray[float], wp.array[float])
    with test.assertRaises(TypeError):
        wp.largearray[float, 2]
    with test.assertRaisesRegex(TypeError, "cannot be instantiated"):
        wp.largearray(shape=4, dtype=float)
    with test.assertRaisesRegex(TypeError, r"write wp\.largearray\[float\] instead of wp\.largearray\(dtype=float\)"):
        wp.largearray(dtype=float)

    # the bare family descriptor used to register built-ins
    generic = wp.largearray(dtype=Any)
    test.assertIs(generic.dtype, Any)
    test.assertEqual(generic.ndim, 1)


def _check_descriptor_layout(test):
    test.assertEqual(ctypes.sizeof(largearray_t), 32)
    test.assertEqual(ctypes.sizeof(array_t), 56)
    test.assertIsInstance(wp.largearray[float].__ctype__(), largearray_t)


def test_ctype64(test, device):
    a = wp.zeros(5, dtype=float, device=device, requires_grad=True)
    c = a.__ctype64__()
    test.assertIsInstance(c, largearray_t)
    test.assertEqual(c.data, a.ptr)
    test.assertEqual(c.grad, a.grad.ptr)
    test.assertEqual(c.shape[0], 5)
    test.assertEqual(c.stride, 4)
    test.assertIs(a.__ctype64__(), c)

    # replacing the gradient invalidates the cached descriptor
    a.grad = wp.zeros_like(a)
    c2 = a.__ctype64__()
    test.assertEqual(c2.grad, a.grad.ptr)
    test.assertNotEqual(c2.grad, c.grad)

    b = wp.zeros(10, dtype=float, device=device)[::2]
    test.assertEqual(b.__ctype64__().stride, 8)
    test.assertEqual(b.__ctype64__().shape[0], 5)
    test.assertEqual(b.__ctype64__().grad, 0)

    with test.assertRaisesRegex(TypeError, "1-D"):
        wp.zeros((2, 3), dtype=float, device=device).__ctype64__()

    # a large array gets a descriptor holding its full length
    _backing, big = _fake_large(device, dtype=wp.float32)
    test.assertEqual(big.__ctype64__().shape[0], 2**31 + 16)


def test_launch_accepts_small_and_large(test, device):
    small = wp.zeros(5, dtype=float, device=device)
    wp.launch(_noop_large, dim=1, inputs=[small], device=device)
    wp.launch(_noop_large, dim=1, inputs=[None], device=device)
    wp.launch(_noop_large, dim=1, inputs=[small.__ctype64__()], device=device)

    _backing, big = _fake_large(device, dtype=wp.float32)
    wp.launch(_noop_large, dim=1, inputs=[big], device=device)

    # adjoint launches pack the gradient into the 64-bit descriptor as well
    a = wp.zeros(5, dtype=float, device=device, requires_grad=True)
    wp.launch(_noop_large, dim=1, inputs=[a], adj_inputs=[a.grad], adjoint=True, device=device)
    wp.launch(_noop_large, dim=1, inputs=[big], adj_inputs=[big], adjoint=True, device=device)
    wp.synchronize_device(device)


def test_launch_mixed_signature(test, device):
    y = wp.zeros(1, dtype=float, device=device)
    wp.launch(_mixed_large, dim=1, inputs=[wp.zeros(5, dtype=float, device=device), y, 3], device=device)
    np.testing.assert_array_equal(y.numpy(), [3.0])

    _backing, big = _fake_large(device, dtype=wp.float32)
    wp.launch(_mixed_large, dim=1, inputs=[big, y, 7], device=device)
    np.testing.assert_array_equal(y.numpy(), [7.0])

    # the 32-bit parameter still rejects a large array
    with test.assertRaisesRegex(RuntimeError, r"argument 'y'.*more than 2\*\*31 - 1 elements"):
        wp.launch(_mixed_large, dim=1, inputs=[big, big, 7], device=device)

    x = wp.zeros(5, dtype=float, device=device, requires_grad=True)
    yg = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    wp.launch(_mixed_large, dim=1, inputs=[x, yg, 3], adj_inputs=[x.grad, yg.grad, 0], adjoint=True, device=device)
    wp.synchronize_device(device)


def test_launch_type_errors(test, device):
    two_d = wp.zeros((2, 2), dtype=float, device=device)
    with test.assertRaisesRegex(RuntimeError, "dimension"):
        wp.launch(_noop_large, dim=1, inputs=[two_d], device=device)

    data = wp.zeros(4, dtype=float, device=device)
    indices = wp.array([0, 1], dtype=int, device=device)
    with test.assertRaisesRegex(RuntimeError, "expects an array"):
        wp.launch(_noop_large, dim=1, inputs=[wp.indexedarray(data, [indices])], device=device)

    with test.assertRaisesRegex(RuntimeError, "dtype"):
        wp.launch(_noop_large, dim=1, inputs=[wp.zeros(4, dtype=wp.int32, device=device)], device=device)

    # a 32-bit descriptor has the wrong layout for a largearray parameter
    with test.assertRaisesRegex(RuntimeError, "expects an array"):
        wp.launch(_noop_large, dim=1, inputs=[data.__ctype__()], device=device)

    with test.assertRaisesRegex(RuntimeError, "expects an array"):
        wp.launch(_noop_large, dim=1, inputs=[np.zeros(4, dtype=np.float32)], device=device)

    # gradients passed to an adjoint launch are checked the same way
    with test.assertRaisesRegex(RuntimeError, "adjoint argument 'x'.*dimension"):
        wp.launch(_noop_large, dim=1, inputs=[data], adj_inputs=[two_d], adjoint=True, device=device)


class _PlainSubclass(wp.array):
    pass


@wp.kernel
def _noop_large_int8(x: wp.largearray[wp.int8]):
    pass


def test_launch_array_subclass(test, device):
    plain = _PlainSubclass(np.zeros(4, dtype=np.float32), dtype=float, device=device)
    wp.launch(_noop_large, dim=1, inputs=[plain], device=device)
    test.assertEqual(plain.__ctype64__().data, plain.ptr)

    custom = _SuperCtypeSubclass(np.zeros(4, dtype=np.float32), dtype=float, device=device)
    with test.assertRaisesRegex(RuntimeError, "custom descriptor are not supported for wp.largearray"):
        wp.launch(_noop_large, dim=1, inputs=[custom], device=device)
    wp.synchronize_device(device)


def test_large_into_array_param(test, device):
    _backing, big = _fake_large(device)
    with test.assertRaisesRegex(RuntimeError, r"argument 'x'.*more than 2\*\*31 - 1 elements"):
        wp.launch(_noop32, dim=1, inputs=[big], device=device)


def test_largearray_type_identity(test, device):
    # largearray parameters get their own native type and overload signature
    test.assertNotEqual(wp._src.types.get_type_code(wp.largearray[float]), wp._src.types.get_type_code(wp.array[float]))
    test.assertNotEqual(_generic_noop_large.sig, _generic_noop32.sig)

    module = _generic_noop.module
    hasher = wp._src.context.ModuleHasher([], module.resolve_options(wp.config))
    test.assertNotEqual(hasher.hash_kernel(_generic_noop_large, False), hasher.hash_kernel(_generic_noop32, False))

    test.assertEqual(wp._src.codegen.Var.type_to_ctype(wp.largearray[float]), "wp::largearray_t<wp::float32>")

    wp.launch(_generic_noop_large, dim=1, inputs=[wp.zeros(3, dtype=float, device=device)], device=device)
    wp.synchronize_device(device)


# Kernel access to largearray. All of these use small real arrays.


@wp.kernel
def _rw(x: wp.largearray[float], y: wp.largearray[float]):
    i = wp.linear_tid()
    y[i] = 2.0 * x[i]
    y[i] += 1.0


@wp.kernel
def _rw_int32(x: wp.largearray[float], y: wp.largearray[float]):
    i = wp.int32(wp.linear_tid())
    y[i] = 2.0 * x[i]
    y[i] += 1.0


@wp.kernel
def _rw_int64(x: wp.largearray[float], y: wp.largearray[float]):
    i = wp.int64(wp.linear_tid())
    y[i] = 2.0 * x[i]
    y[i] += 1.0


@wp.kernel
def _rw_uint32(x: wp.largearray[float], y: wp.largearray[float]):
    i = wp.uint32(wp.linear_tid())
    y[i] = 2.0 * x[i]
    y[i] += 1.0


@wp.kernel
def _rw_uint64(x: wp.largearray[float], y: wp.largearray[float]):
    i = wp.uint64(wp.linear_tid())
    y[i] = 2.0 * x[i]
    y[i] += 1.0


@wp.kernel
def _rw_int16(x: wp.largearray[float], y: wp.largearray[float]):
    i = wp.int16(wp.linear_tid())
    y[i] = 2.0 * x[i]
    y[i] += 1.0


def test_read_write(test, device):
    x = wp.array(np.arange(8, dtype=np.float32), device=device)
    y = wp.zeros(8, dtype=float, device=device)
    wp.launch(_rw, dim=8, inputs=[x, y], device=device)
    np.testing.assert_array_equal(y.numpy(), 2.0 * np.arange(8) + 1.0)


def test_index_types(test, device):
    x = wp.array(np.arange(8, dtype=np.float32), device=device)
    expected = 2.0 * np.arange(8) + 1.0
    for kernel in (_rw_int32, _rw_int64, _rw_uint32, _rw_uint64, _rw_int16):
        with test.subTest(kernel=kernel.key):
            y = wp.zeros(8, dtype=float, device=device)
            wp.launch(kernel, dim=8, inputs=[x, y], device=device)
            np.testing.assert_array_equal(y.numpy(), expected)


@wp.kernel
def _negative(x: wp.largearray[float], out: wp.array[float]):
    out[0] = x[-1]
    out[1] = x[wp.int64(-2)]
    x[-1] = 42.0


def test_negative_index(test, device):
    x = wp.array(np.arange(8, dtype=np.float32), device=device)
    out = wp.zeros(2, dtype=float, device=device)
    wp.launch(_negative, dim=1, inputs=[x, out], device=device)
    np.testing.assert_array_equal(out.numpy(), [7.0, 6.0])
    result = x.numpy()
    test.assertEqual(result[7], 42.0)
    np.testing.assert_array_equal(result[0:7], np.arange(7))


@wp.kernel(module="unique")
def _lit_read(x: wp.largearray[float], out: wp.array[float]):
    out[0] = x[2147483652]


@wp.kernel(module="unique")
def _lit_store(x: wp.largearray[float]):
    x[2147483652] = 1.0


@wp.kernel(module="unique")
def _lit_augassign(x: wp.largearray[float]):
    x[2147483652] += 1.0


@wp.kernel(module="unique")
def _lit_atomic(x: wp.largearray[float]):
    wp.atomic_add(x, 2147483652, 1.0)


@wp.kernel(module="unique")
def _lit_negative(x: wp.largearray[float], out: wp.array[float]):
    out[0] = x[-2147483652]


@wp.kernel(module="unique")
def _lit_component(x: wp.largearray[wp.vec2], v: float):
    x[2147483652][1] += v


_LIT_CONSTANT = wp.constant(2147483652)


@wp.kernel(module="unique")
def _lit_constant(x: wp.largearray[float], out: wp.array[float]):
    out[0] = x[_LIT_CONSTANT]


@wp.kernel(module="unique")
def _lit_assigned(x: wp.largearray[float], out: wp.array[float]):
    i = 2147483652
    out[0] = x[i]


@wp.kernel(module="unique")
def _lit_ok(x: wp.largearray[float], out: wp.array[float]):
    out[0] = x[wp.int64(3)]
    out[1] = x[3]
    out[2] = x[-1]
    out[3] = x[wp.uint32(2)]


def test_large_literal_index_rejected(test, device):
    x = wp.zeros(8, dtype=float, device=device)
    out = wp.zeros(4, dtype=float, device=device)
    xv = wp.zeros(8, dtype=wp.vec2, device=device)
    message = r"Index -?2147483652 into a wp.largearray does not fit in a 32-bit integer literal"
    cases = (
        (_lit_read, [x, out]),
        (_lit_store, [x]),
        (_lit_augassign, [x]),
        (_lit_atomic, [x]),
        (_lit_negative, [x, out]),
        (_lit_component, [xv, 1.0]),
        (_lit_constant, [x, out]),
        (_lit_assigned, [x, out]),
    )
    for kernel, inputs in cases:
        with test.subTest(kernel=kernel.key):
            with test.assertRaisesRegex(wp.WarpCodegenError, message):
                wp.launch(kernel, dim=1, inputs=inputs, device=device)


def test_small_literal_index_accepted(test, device):
    x = wp.array(np.arange(8, dtype=np.float32), device=device)
    out = wp.zeros(4, dtype=float, device=device)
    wp.launch(_lit_ok, dim=1, inputs=[x, out], device=device)
    np.testing.assert_array_equal(out.numpy(), [3.0, 3.0, 7.0, 2.0])


@wp.kernel
def _len_shape(x: wp.largearray[float], out: wp.array[wp.int64]):
    out[0] = len(x)
    out[1] = x.shape[0]


@wp.kernel(module="unique")
def _len_into_int32(x: wp.largearray[float], out: wp.array[wp.int32]):
    n = wp.int32(0)
    n = len(x)
    out[0] = n


@wp.kernel(module="unique")
def _shape_const_index(x: wp.largearray[float], out: wp.array[wp.int64]):
    out[0] = x.shape[1]


@wp.kernel(module="unique")
def _shape_var_index(x: wp.largearray[float], d: int, out: wp.array[wp.int64]):
    out[0] = x.shape[d]


def test_len_shape(test, device):
    x = wp.zeros(5, dtype=float, device=device)
    out = wp.zeros(2, dtype=wp.int64, device=device)
    wp.launch(_len_shape, dim=1, inputs=[x, out], device=device)
    np.testing.assert_array_equal(out.numpy(), [5, 5])

    # len() of a largearray is int64, so it does not fit a variable declared int32
    with test.assertRaisesRegex(wp.WarpCodegenTypeError, r"int32.*int64"):
        wp.launch(_len_into_int32, dim=1, inputs=[x, wp.zeros(1, dtype=wp.int32, device=device)], device=device)

    # a largearray shape has one dimension, so any index other than the constant 0 is rejected
    with test.assertRaisesRegex(wp.WarpCodegenError, r"shape can only be indexed with the constant 0"):
        wp.launch(_shape_const_index, dim=1, inputs=[x, out], device=device)
    with test.assertRaisesRegex(wp.WarpCodegenError, r"shape can only be indexed with the constant 0"):
        wp.launch(_shape_var_index, dim=1, inputs=[x, 0, out], device=device)


@wp.kernel
def _atomic_add_k(acc: Any, vals: wp.array[float]):
    wp.atomic_add(acc, 0, vals[wp.tid()])


@wp.kernel
def _atomic_sub_k(acc: Any, vals: wp.array[float]):
    wp.atomic_sub(acc, 0, vals[wp.tid()])


@wp.kernel
def _atomic_min_k(acc: Any, vals: wp.array[float]):
    wp.atomic_min(acc, 0, vals[wp.tid()])


@wp.kernel
def _atomic_max_k(acc: Any, vals: wp.array[float]):
    wp.atomic_max(acc, 0, vals[wp.tid()])


@wp.kernel
def _atomic_and_k(acc: Any, vals: wp.array[wp.int32]):
    wp.atomic_and(acc, 0, vals[wp.tid()])


@wp.kernel
def _atomic_or_k(acc: Any, vals: wp.array[wp.int32]):
    wp.atomic_or(acc, 0, vals[wp.tid()])


@wp.kernel
def _atomic_xor_k(acc: Any, vals: wp.array[wp.int32]):
    wp.atomic_xor(acc, 0, vals[wp.tid()])


@wp.kernel
def _atomic_cas_k(acc: Any, out: wp.array[float]):
    out[0] = wp.atomic_cas(acc, 0, 1.0, 5.0)  # swaps
    out[1] = wp.atomic_cas(acc, 0, 1.0, 9.0)  # does not swap


@wp.kernel
def _atomic_exch_k(acc: Any, out: wp.array[float]):
    out[0] = wp.atomic_exch(acc, 0, 3.0)
    out[1] = wp.atomic_exch(acc, 0, 4.0)


_COMMUTATIVE_ATOMICS = (
    (_atomic_add_k, float, 0.0),
    (_atomic_sub_k, float, 0.0),
    (_atomic_min_k, float, 1000.0),
    (_atomic_max_k, float, -1000.0),
    (_atomic_and_k, wp.int32, -1),
    (_atomic_or_k, wp.int32, 0),
    (_atomic_xor_k, wp.int32, 0),
)

# (largearray overload, wp.array overload, dtype, initial accumulator value)
_COMMUTATIVE_ATOMIC_OVERLOADS = tuple(
    (
        wp.overload(_kernel, {"acc": wp.largearray[_dtype], "vals": wp.array[_dtype]}),
        wp.overload(_kernel, {"acc": wp.array[_dtype], "vals": wp.array[_dtype]}),
        _dtype,
        _init,
    )
    for _kernel, _dtype, _init in _COMMUTATIVE_ATOMICS
)

# (largearray overload, wp.array overload, expected returned values, expected final value)
_EXCHANGE_ATOMIC_OVERLOADS = tuple(
    (
        wp.overload(_kernel, {"acc": wp.largearray[float], "out": wp.array[float]}),
        wp.overload(_kernel, {"acc": wp.array[float], "out": wp.array[float]}),
        _returned,
        _final,
    )
    for _kernel, _returned, _final in ((_atomic_cas_k, [1.0, 5.0], 5.0), (_atomic_exch_k, [1.0, 3.0], 4.0))
)


def test_atomics(test, device):
    rng = np.random.default_rng(42)
    float_vals = rng.permutation(64).astype(np.float32) - 20.0
    int_vals = rng.integers(0, 2**30, size=64, dtype=np.int32)

    for kernel_large, kernel32, dtype, init in _COMMUTATIVE_ATOMIC_OVERLOADS:
        with test.subTest(kernel=kernel_large.key):
            vals = wp.array(int_vals if dtype is wp.int32 else float_vals, dtype=dtype, device=device)
            acc_large = wp.full(1, init, dtype=dtype, device=device)
            acc32 = wp.full(1, init, dtype=dtype, device=device)
            wp.launch(kernel_large, dim=64, inputs=[acc_large, vals], device=device)
            wp.launch(kernel32, dim=64, inputs=[acc32, vals], device=device)
            test.assertNotEqual(acc32.numpy()[0], init)
            np.testing.assert_array_equal(acc_large.numpy(), acc32.numpy())

    # cas and exch race when several threads update one element, so compare single-thread launches
    for kernel_large, kernel32, returned, final in _EXCHANGE_ATOMIC_OVERLOADS:
        with test.subTest(kernel=kernel_large.key):
            for kernel in (kernel_large, kernel32):
                acc = wp.full(1, 1.0, dtype=float, device=device)
                out = wp.zeros(2, dtype=float, device=device)
                wp.launch(kernel, dim=1, inputs=[acc, out], device=device)
                np.testing.assert_array_equal(out.numpy(), returned)
                np.testing.assert_array_equal(acc.numpy(), [final])


@wp.struct
class _Elem:
    a: float
    b: wp.int32


@wp.kernel
def _copy_large(x: Any, y: Any):
    i = wp.linear_tid()
    y[i] = x[i]


@wp.kernel
def _fill_elem(x: wp.array[_Elem]):
    i = wp.tid()
    e = _Elem()
    e.a = float(i) + 0.5
    e.b = i * 3
    x[i] = e


_COPY_OVERLOADS = tuple(
    (_dtype, wp.overload(_copy_large, {"x": wp.largearray[_dtype], "y": wp.largearray[_dtype]}))
    for _dtype in (wp.vec3, wp.mat22, wp.int8, wp.float64, _Elem)
)


def test_dtypes(test, device):
    n = 6
    for dtype, kernel in _COPY_OVERLOADS:
        with test.subTest(dtype=dtype):
            if dtype is _Elem:
                x = wp.zeros(n, dtype=_Elem, device=device)
                wp.launch(_fill_elem, dim=n, inputs=[x], device=device)
            else:
                length = wp._src.types.type_size(dtype)
                values = np.arange(n * length).reshape((n, *getattr(dtype, "_shape_", ())))
                x = wp.array(values, dtype=dtype, device=device)
            y = wp.zeros(n, dtype=dtype, device=device)
            wp.launch(kernel, dim=n, inputs=[x, y], device=device)
            if dtype is _Elem:
                np.testing.assert_array_equal(y.numpy()["a"], np.arange(n) + 0.5)
                np.testing.assert_array_equal(y.numpy()["b"], np.arange(n) * 3)
            else:
                np.testing.assert_array_equal(y.numpy(), x.numpy())


def test_strided(test, device):
    x = wp.array(np.arange(8, dtype=np.float32), device=device)
    base = wp.zeros(16, dtype=float, device=device)
    wp.launch(_rw, dim=8, inputs=[x, base[::2]], device=device)
    result = base.numpy()
    np.testing.assert_array_equal(result[::2], 2.0 * np.arange(8) + 1.0)
    np.testing.assert_array_equal(result[1::2], np.zeros(8))


@wp.func
def _read_large(a: wp.largearray[float], i: wp.int64) -> float:
    return a[i]


@wp.func
def _read32(a: wp.array[float], i: int) -> float:
    return a[i]


@wp.kernel
def _call_read_large(x: wp.largearray[float], out: wp.largearray[float]):
    i = wp.linear_tid()
    out[i] = _read_large(x, i)


@wp.kernel(module="unique")
def _call_read_large_with_array(x: wp.array[float], out: wp.array[float]):
    out[0] = _read_large(x, wp.int64(0))


@wp.kernel(module="unique")
def _call_read32_with_largearray(x: wp.largearray[float], out: wp.array[float]):
    out[0] = _read32(x, 0)


def test_func_param(test, device):
    x = wp.array(np.arange(4, dtype=np.float32) + 10.0, device=device)
    out = wp.zeros(4, dtype=float, device=device)
    wp.launch(_call_read_large, dim=4, inputs=[x, out], device=device)
    np.testing.assert_array_equal(out.numpy(), np.arange(4) + 10.0)


def test_closed_type(test, device):
    x = wp.zeros(4, dtype=float, device=device)
    out = wp.zeros(1, dtype=float, device=device)
    # the error names both the parameter type and the argument type
    a_large = r"largearray\(ndim=1, dtype=float32\)"
    a32 = r"array\(ndim=1, dtype=float32\)"
    with test.assertRaisesRegex(wp.WarpCodegenError, rf"_read_large.*a: {a_large}.*\[{a32}, int64\]"):
        wp.launch(_call_read_large_with_array, dim=1, inputs=[x, out], device=device)
    with test.assertRaisesRegex(wp.WarpCodegenError, rf"_read32.*a: {a32}.*\[{a_large}, int32\]"):
        wp.launch(_call_read32_with_largearray, dim=1, inputs=[x, out], device=device)


@wp.kernel(module="unique")
def _slice_large(x: wp.largearray[float], out: wp.array[float]):
    v = x[0:2]
    out[0] = v[0]


@wp.kernel(module="unique")
def _lower_bound_large(x: wp.largearray[float], out: wp.array[int]):
    out[0] = wp.lower_bound(x, 1.0)


@wp.func(module="unique")
def _return_largearray(x: wp.largearray[float]) -> wp.array[float]:
    return x


@wp.kernel(module="unique")
def _call_return_largearray(x: wp.largearray[float]):
    _return_largearray(x)


def test_return_type_mismatch(test, device):
    # The return-type error names the type the code returns, so wp.largearray needs a printable name.
    x = wp.zeros(4, dtype=float, device=device)
    with test.assertRaisesRegex(wp.WarpCodegenError, r"returns a value of type `LargeArray\[float32\]`"):
        wp.launch(_call_return_largearray, dim=1, inputs=[x], device=device)


def test_unsupported(test, device):
    x = wp.zeros(4, dtype=float, device=device)
    with test.assertRaisesRegex(wp.WarpCodegenError, r"wp\.largearray does not support slicing or views\."):
        wp.launch(_slice_large, dim=1, inputs=[x, wp.zeros(1, dtype=float, device=device)], device=device)
    with test.assertRaisesRegex(wp.WarpCodegenError, r"lower_bound.*does not support wp\.largearray"):
        wp.launch(_lower_bound_large, dim=1, inputs=[x, wp.zeros(1, dtype=int, device=device)], device=device)

    class _S:
        a: wp.largearray[float]

    with test.assertRaisesRegex(TypeError, r"wp\.largearray is not supported in structs\."):
        wp.struct(_S)


@wp.kernel
def _uint32_index(x: wp.largearray[float], j: wp.uint32, k: wp.uint32, out: wp.array[float]):
    # Passed as parameters: a literal above 2**31 would be emitted as an int32 constant first.
    out[0] = x[j - k]


def test_uint32_index_small_array(test, device):
    # Smoke test of uint32 indexing on a small array. It cannot detect sign extension of a uint32
    # index above 2**31, which would need an array that large.
    x = wp.array(np.arange(8, dtype=np.float32), device=device)
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(_uint32_index, dim=1, inputs=[x, 4294967295, 4294967292, out], device=device)
    np.testing.assert_array_equal(out.numpy(), [3.0])


@wp.kernel
def _null_untouched(x: wp.largearray[float], y: wp.largearray[float]):
    y[wp.linear_tid()] = 1.0


def test_null_array(test, device):
    y = wp.zeros(3, dtype=float, device=device)
    wp.launch(_null_untouched, dim=3, inputs=[None, y], device=device)
    np.testing.assert_array_equal(y.numpy(), [1.0, 1.0, 1.0])


def _family_overloads(kernel, params, dtype=float):
    """Return the (wp.largearray, wp.array) overloads of ``kernel`` with ``params`` set to each family."""
    return tuple(wp.overload(kernel, dict.fromkeys(params, family[dtype])) for family in (wp.largearray, wp.array))


@wp.kernel
def _null_check_k(x: Any, y: Any, out: wp.array[wp.int32]):
    if x:
        out[0] = 1
    if not x:
        out[1] = 1
    if x and y:
        out[2] = 1
    if x or y:
        out[3] = 1


_NULL_CHECK = _family_overloads(_null_check_k, ("x", "y"))


def test_null_check(test, device):
    a = wp.zeros(2, dtype=float, device=device)
    b = wp.zeros(2, dtype=float, device=device)
    expected = {
        (False, False): [0, 1, 0, 0],
        (True, False): [1, 0, 0, 1],
        (False, True): [0, 1, 0, 1],
        (True, True): [1, 0, 1, 1],
    }
    for (has_x, has_y), want in expected.items():
        results = []
        for kernel in _NULL_CHECK:
            out = wp.zeros(4, dtype=wp.int32, device=device)
            wp.launch(kernel, dim=1, inputs=[a if has_x else None, b if has_y else None, out], device=device)
            results.append(out.numpy())
        with test.subTest(x=has_x, y=has_y):
            np.testing.assert_array_equal(results[0], results[1])
            np.testing.assert_array_equal(results[0], want)


# ---------------------------------------------------------------------------------------------
# Autodiff. Each computation runs through a wp.largearray overload and a wp.array overload of the
# same kernel, and the gradients must match.


@wp.kernel
def _grad_read_k(x: Any, loss: wp.array[float]):
    i = wp.tid()
    loss[0] += x[i] * x[i]


@wp.kernel
def _grad_write_k(x: Any, y: Any):
    i = wp.tid()
    y[i] = 3.0 * x[i]


@wp.kernel
def _grad_sum_k(y: Any, loss: wp.array[float]):
    i = wp.tid()
    loss[0] += y[i]


@wp.kernel
def _grad_atomic_k(x: Any, y: Any):
    i = wp.tid()
    wp.atomic_add(y, wp.int64(0), x[i])


@wp.kernel
def _grad_shape_k(x: Any, loss: wp.array[float]):
    i = wp.tid()
    # A negative index wraps by the length of the gradient array too. The length is read after
    # the element, so its reverse statement runs before the element's adjoint.
    v = x[i - 4]
    n = x.shape[0]
    m = len(x)
    loss[0] += v * v * float(n) + float(m)


@wp.kernel
def _grad_vec_k(x: Any, y: Any):
    i = wp.tid()
    y[i] = 2.0 * x[i]
    # composite slot updates: atomic add, atomic subtract, and store
    y[i][1] += x[i][0]
    y[i][0] -= x[i][1]
    y[i].z = 4.0 * x[i][2]


@wp.kernel
def _grad_vec_sum_k(y: Any, loss: wp.array[float]):
    i = wp.tid()
    loss[0] += wp.dot(y[i], wp.vec3(1.0, 2.0, 3.0))


@wp.kernel
def _grad_func_large(x: wp.largearray[float], loss: wp.array[float]):
    i = wp.linear_tid()
    v = _read_large(x, i)
    loss[0] += v * v


@wp.kernel
def _grad_func32(x: wp.array[float], loss: wp.array[float]):
    i = wp.tid()
    v = _read32(x, i)
    loss[0] += v * v


_GRAD_READ = _family_overloads(_grad_read_k, ("x",))
_GRAD_WRITE = _family_overloads(_grad_write_k, ("x", "y"))
_GRAD_SUM = _family_overloads(_grad_sum_k, ("y",))
_GRAD_ATOMIC = _family_overloads(_grad_atomic_k, ("x", "y"))
_GRAD_SHAPE = _family_overloads(_grad_shape_k, ("x",))
_GRAD_VEC = _family_overloads(_grad_vec_k, ("x", "y"), wp.vec3)
_GRAD_VEC_SUM = _family_overloads(_grad_vec_sum_k, ("y",), wp.vec3)

_X_NP = np.arange(8, dtype=np.float32) - 3.0


def _loss_grad(kernel, x, device):
    """Launch ``kernel(x, loss)`` on a tape, run it backward from ``loss``, and return the tape."""
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=x.shape[0], inputs=[x, loss], device=device)
    tape.backward(loss)
    return tape


def test_grad_read(test, device):
    grads = []
    for kernel in _GRAD_READ:
        x = wp.array(_X_NP, device=device, requires_grad=True)
        _loss_grad(kernel, x, device)
        grads.append(x.grad.numpy())
    np.testing.assert_allclose(grads[0], grads[1])
    np.testing.assert_allclose(grads[0], 2.0 * _X_NP)


def test_grad_write(test, device):
    grads = []
    for write, total in zip(_GRAD_WRITE, _GRAD_SUM, strict=True):
        x = wp.array(_X_NP, device=device, requires_grad=True)
        y = wp.zeros_like(x, requires_grad=True)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(write, dim=8, inputs=[x, y], device=device)
            wp.launch(total, dim=8, inputs=[y, loss], device=device)
        tape.backward(loss)
        np.testing.assert_allclose(y.numpy(), 3.0 * _X_NP)
        grads.append(x.grad.numpy())
    np.testing.assert_allclose(grads[0], grads[1])
    np.testing.assert_allclose(grads[0], np.full(8, 3.0))


def test_grad_atomic_add(test, device):
    grads = []
    for kernel in _GRAD_ATOMIC:
        x = wp.array(_X_NP, device=device, requires_grad=True)
        y = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=8, inputs=[x, y], device=device)
        tape.backward(grads={y: wp.ones(1, dtype=float, device=device)})
        np.testing.assert_allclose(y.numpy(), [_X_NP.sum()])
        grads.append(x.grad.numpy())
    np.testing.assert_allclose(grads[0], grads[1])
    np.testing.assert_allclose(grads[0], np.ones(8))


def test_grad_shape(test, device):
    # The gradient array is a view into a zeroed guard buffer, so a gradient written below
    # index 0 lands in the guard instead of outside the allocation.
    grads = []
    for kernel in _GRAD_SHAPE:
        x = wp.array(_X_NP, device=device, requires_grad=True)
        guard = wp.zeros(16, dtype=float, device=device)
        x.grad = guard[8:]
        _loss_grad(kernel, x, device)
        np.testing.assert_array_equal(guard.numpy()[:8], np.zeros(8))
        grads.append(x.grad.numpy())
    np.testing.assert_allclose(grads[0], grads[1])
    np.testing.assert_allclose(grads[0], 2.0 * 8.0 * _X_NP)


def test_grad_composite(test, device):
    grads = []
    for write, total in zip(_GRAD_VEC, _GRAD_VEC_SUM, strict=True):
        x = wp.array(np.arange(12, dtype=np.float32).reshape(4, 3), dtype=wp.vec3, device=device, requires_grad=True)
        y = wp.zeros_like(x, requires_grad=True)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(write, dim=4, inputs=[x, y], device=device)
            wp.launch(total, dim=4, inputs=[y, loss], device=device)
        tape.backward(loss)
        grads.append(x.grad.numpy())
    np.testing.assert_allclose(grads[0], grads[1])
    # loss = y . (1, 2, 3) with y = (2 x0 - x1, 2 x1 + x0, 4 x2)
    np.testing.assert_allclose(grads[0], np.tile([4.0, 3.0, 12.0], (4, 1)))


def test_grad_func(test, device):
    grads = []
    for kernel in (_grad_func_large, _grad_func32):
        x = wp.array(_X_NP, device=device, requires_grad=True)
        _loss_grad(kernel, x, device)
        grads.append(x.grad.numpy())
    np.testing.assert_allclose(grads[0], grads[1])
    np.testing.assert_allclose(grads[0], 2.0 * _X_NP)


def test_tape_zero(test, device):
    x = wp.array(_X_NP, device=device, requires_grad=True)
    tape = _loss_grad(_GRAD_READ[0], x, device)
    np.testing.assert_allclose(x.grad.numpy(), 2.0 * _X_NP)
    tape.zero()
    np.testing.assert_array_equal(x.grad.numpy(), np.zeros(8))


def test_grad_replaced(test, device):
    x = wp.array(_X_NP, device=device, requires_grad=True)
    tape = _loss_grad(_GRAD_READ[0], x, device)
    old_grad = x.grad
    np.testing.assert_allclose(old_grad.numpy(), 2.0 * _X_NP)

    # Replacing the gradient discards the cached 64-bit descriptor, and the next backward pass
    # accumulates into the new array only.
    x.grad = wp.zeros_like(x)
    test.assertEqual(x.__ctype64__().grad, x.grad.ptr)
    tape.backward()
    np.testing.assert_allclose(x.grad.numpy(), 2.0 * _X_NP)
    np.testing.assert_allclose(old_grad.numpy(), 2.0 * _X_NP)


@wp.kernel(module="unique", module_options={"enable_backward": False})
def _forward_only_large(x: wp.largearray[float], y: wp.largearray[float]):
    i = wp.linear_tid()
    y[i] = x[i] * x[i]


def test_enable_backward_false(test, device):
    x = wp.array(_X_NP, device=device)
    y = wp.zeros_like(x)
    wp.launch(_forward_only_large, dim=8, inputs=[x, y], device=device)
    np.testing.assert_allclose(y.numpy(), _X_NP * _X_NP)
    test.assertFalse(_forward_only_large.module.options["enable_backward"])


# ---------------------------------------------------------------------------------------------
# Interactions with other Warp features.

_R2R = {"deterministic": wp.DeterministicMode.RUN_TO_RUN}


@wp.kernel(module="unique", module_options=_R2R)
def _det_scatter_large(out: wp.largearray[float]):
    wp.atomic_add(out, 0, 1.0)


@wp.kernel(module="unique", module_options=_R2R)
def _det_read_grad_large(x: wp.largearray[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = x[i]


@wp.kernel(module="unique", module_options=_R2R)
def _det_counter_large(counter: wp.largearray[wp.int32], out: wp.array[wp.int32]):
    i = wp.tid()
    slot = wp.atomic_add(counter, 0, 1)
    out[slot] = i


@wp.kernel(module="unique", module_options={**_R2R, "enable_backward": False})
def _det_store_large(x: wp.largearray[float], y: wp.largearray[float], total: wp.array[float]):
    i = wp.linear_tid()
    y[i] = 2.0 * x[i]
    wp.atomic_add(total, 0, x[i])


def test_deterministic(test, device):
    # Deferred deterministic updates cannot target a wp.largearray. The error comes from code
    # generation, so the kernel never runs; a large destination is rejected the same way.
    pattern = r"array 'out' is a wp\.largearray and receives wp\.atomic_add\(\) updates"
    _backing, big = _fake_large(device, dtype=wp.float32)
    for out in (wp.zeros(1, dtype=float, device=device), big):
        with test.assertRaisesRegex(wp.WarpDeterminismError, pattern):
            wp.launch(_det_scatter_large, dim=4, inputs=[out], device=device)

    x = wp.zeros(4, dtype=float, device=device, requires_grad=True)
    with test.assertRaisesRegex(
        wp.WarpDeterminismError, r"array 'adj_x' is a wp\.largearray and receives gradient.*enable_backward=False"
    ):
        wp.launch(_det_read_grad_large, dim=4, inputs=[x, wp.zeros(4, dtype=float, device=device)], device=device)

    with test.assertRaisesRegex(wp.WarpDeterminismError, r"array 'counter' is a wp\.largearray"):
        counter = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            _det_counter_large, dim=4, inputs=[counter, wp.zeros(4, dtype=wp.int32, device=device)], device=device
        )

    # Reads and plain stores need no deferred update, and still work.
    x = wp.array(_X_NP, device=device)
    y = wp.zeros_like(x)
    total = wp.zeros(1, dtype=float, device=device)
    wp.launch(_det_store_large, dim=8, inputs=[x, y, total], device=device)
    np.testing.assert_allclose(y.numpy(), 2.0 * _X_NP)
    np.testing.assert_allclose(total.numpy(), [_X_NP.sum()])


_JET = wp.JetSpace(2, width=1)


@wp.kernel
def _jet_k(a: Any, b: Any, out: Any):
    i = wp.tid()
    u, v = _JET.directional(a[i], b[i], 1.0, 0.0)
    out[i] = (wp.sin(u * v) + u * u).grad[0]


_JET_OVERLOADS = _family_overloads(_jet_k, ("a", "b", "out"))


def test_jets(test, device):
    # Forward-mode jet derivative d/da, then reverse mode over it for d2/da2.
    a_np = np.array([0.7, -0.2, 0.4], dtype=np.float32)
    b_np = np.array([-0.4, 0.9, 1.5], dtype=np.float32)
    results = []
    for kernel in _JET_OVERLOADS:
        a = wp.array(a_np, device=device, requires_grad=True)
        b = wp.array(b_np, device=device, requires_grad=True)
        out = wp.zeros_like(a, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=3, inputs=[a, b, out], device=device)
        tape.backward(grads={out: wp.ones_like(out)})
        results.append((out.numpy(), a.grad.numpy()))
    for r_large, r32 in zip(*results, strict=True):
        np.testing.assert_allclose(r_large, r32)
    np.testing.assert_allclose(results[0][0], b_np * np.cos(a_np * b_np) + 2.0 * a_np, rtol=1e-5)
    np.testing.assert_allclose(results[0][1], -(b_np**2) * np.sin(a_np * b_np) + 2.0, rtol=1e-5)


def _check_cpu_blocks(test):
    device = "cpu"
    previous = wp.config.enable_cpu_blocks
    try:
        wp.config.enable_cpu_blocks = True
        grads = []
        for kernel in _GRAD_READ:
            # seven elements: one full block of four and a partial one
            x = wp.array(_X_NP[:7], device=device, requires_grad=True)
            loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=7, inputs=[x, loss], device=device, block_dim=4)
            tape.backward(loss)
            np.testing.assert_allclose(loss.numpy(), [np.sum(_X_NP[:7] ** 2)])
            grads.append(x.grad.numpy())
        np.testing.assert_allclose(grads[0], grads[1])
        np.testing.assert_allclose(grads[0], 2.0 * _X_NP[:7])
    finally:
        wp.config.enable_cpu_blocks = previous


def test_graph_capture(test, device):
    x = wp.array(_X_NP, device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        tape = wp.Tape()
        with tape:
            wp.launch(_GRAD_READ[0], dim=8, inputs=[x, loss], device=device)
        tape.backward(loss)

    for replay in range(1, 3):
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(loss.numpy(), [replay * np.sum(_X_NP**2)])
        np.testing.assert_allclose(x.grad.numpy(), replay * 2.0 * _X_NP)


def test_apic_rejected(test, device):
    # CPU graphs always record through APIC; CUDA records through it with apic=True.
    x = wp.array(_X_NP, device=device)
    y = wp.zeros_like(x)
    wp.launch(_GRAD_WRITE[0], dim=8, inputs=[x, y], device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False):
        with test.assertRaisesRegex(NotImplementedError, r"APIC capture does not yet support wp\.largearray"):
            wp.launch(_GRAD_WRITE[0], dim=8, inputs=[x, y], device=device)
    wp.synchronize_device(device)


def test_verify_autograd_array_access(test, device):
    previous = wp.config.verify_autograd_array_access
    try:
        wp.config.verify_autograd_array_access = True

        # Access tracking is recorded when a module is built. Kernels created here have not been
        # loaded before (e.g. by wp.force_load() in another test) without the option.
        @wp.kernel(module="unique")
        def _verify_read_large(x: wp.largearray[float], y: wp.largearray[float]):
            i = wp.linear_tid()
            y[i] = x[i] * x[i]

        @wp.kernel(module="unique")
        def _verify_write_large(z: wp.largearray[float], x: wp.largearray[float]):
            i = wp.linear_tid()
            x[i] = z[i]

        a = wp.array(_X_NP, device=device, requires_grad=True)
        b = wp.zeros_like(a)
        c = wp.zeros_like(a)
        tape = wp.Tape()
        with contextlib.redirect_stderr(io.StringIO()) as f:
            with tape:
                wp.launch(_verify_read_large, dim=8, inputs=[a, b], device=device)
                wp.launch(_verify_write_large, dim=8, inputs=[c, a], device=device)
        test.assertIn("is being written to but has already been read from in a previous launch", f.getvalue())
    finally:
        wp.config.verify_autograd_array_access = previous


def test_llvm_cuda(test, device):
    previous = wp.config.llvm_cuda
    try:
        wp.config.llvm_cuda = True

        # created here so that no earlier load (e.g. wp.force_load()) built it with NVRTC
        @wp.kernel(module="unique")
        def _llvm_cuda_large(x: wp.largearray[float], loss: wp.array[float]):
            i = wp.linear_tid()
            loss[0] += x[i] * x[i]

        x = wp.array(_X_NP, device=device, requires_grad=True)
        _loss_grad(_llvm_cuda_large, x, device)
        np.testing.assert_allclose(x.grad.numpy(), 2.0 * _X_NP)
    finally:
        wp.config.llvm_cuda = previous


def test_record_cmd(test, device):
    write_large = _GRAD_WRITE[0]
    x = wp.array(_X_NP, device=device)
    y = wp.zeros_like(x)
    cmd = wp.launch(write_large, dim=8, inputs=[x, y], device=device, record_cmd=True)
    cmd.launch()
    np.testing.assert_allclose(y.numpy(), 3.0 * _X_NP)

    # swap an array and shrink the launch
    y2 = wp.zeros_like(x)
    cmd.set_param_at_index(1, y2)
    cmd.set_dim(4)
    cmd.launch()
    np.testing.assert_allclose(y2.numpy(), np.concatenate((3.0 * _X_NP[:4], np.zeros(4))))

    with test.assertRaisesRegex(RuntimeError, "dimension"):
        cmd.set_param_at_index(0, wp.zeros((2, 4), dtype=float, device=device))

    # adjoint parameters go through the same 64-bit packing
    x_grad = wp.zeros_like(x)
    adj = wp.launch(
        write_large,
        dim=8,
        inputs=[x, y],
        adj_inputs=[x_grad, wp.ones_like(y)],
        device=device,
        adjoint=True,
        record_cmd=True,
    )
    x_grad2 = wp.zeros_like(x)
    adj.set_param_at_index(0, x_grad2, adjoint=True)
    adj.launch()
    np.testing.assert_allclose(x_grad2.numpy(), np.full(8, 3.0))
    np.testing.assert_array_equal(x_grad.numpy(), np.zeros(8))

    # a large array can be swapped in without touching its elements
    _backing, big = _fake_large(device, dtype=wp.float32)
    noop = wp.launch(_noop_large, dim=1, inputs=[x], device=device, record_cmd=True)
    noop.set_param_at_index(0, big)
    noop.launch()
    wp.synchronize_device(device)


@wp.kernel
def _mixed_contiguous(a: wp.array[float], x: wp.largearray[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * x[i]


def _check_cpu_linear_contiguous_mixed(test):
    device = "cpu"
    # The kernel is eligible for the CPU loop specialized for contiguous wp.array arguments; the
    # strided wp.largearray argument must keep its runtime stride in both passes.
    a = wp.array(np.arange(8, dtype=np.float32) + 1.0, device=device, requires_grad=True)
    x_base = wp.array(np.arange(16, dtype=np.float32), device=device)
    x = x_base[::2]
    x_grad_base = wp.zeros(16, dtype=float, device=device)
    x.grad = x_grad_base[::2]
    out = wp.zeros(8, dtype=float, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(_mixed_contiguous, dim=8, inputs=[a, x, out], device=device)
    tape.backward(grads={out: wp.ones_like(out)})
    np.testing.assert_allclose(out.numpy(), a.numpy() * 2.0 * np.arange(8))
    np.testing.assert_allclose(a.grad.numpy(), 2.0 * np.arange(8))
    np.testing.assert_allclose(x_grad_base.numpy()[::2], a.numpy())
    np.testing.assert_array_equal(x_grad_base.numpy()[1::2], np.zeros(8))


devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()


class TestLargeArray(unittest.TestCase):
    # Pure-Python checks and CPU-only features, which run once on a fixed device.

    def test_annotation(self):
        _check_annotation(self)

    def test_descriptor_layout(self):
        _check_descriptor_layout(self)

    def test_shape_boundaries(self):
        _check_shape_boundaries(self)

    def test_cpu_blocks(self):
        _check_cpu_blocks(self)

    def test_cpu_linear_contiguous_mixed(self):
        _check_cpu_linear_contiguous_mixed(self)

    def test_large_rejected_by_graph_coloring(self):
        _check_large_rejected_by_graph_coloring(self)


add_function_test(TestLargeArray, "test_large_shape_accepted", test_large_shape_accepted, devices=devices)
add_function_test(
    TestLargeArray,
    "test_large_rejected_through_class_method",
    test_large_rejected_through_class_method,
    devices=devices,
)
add_function_test(
    TestLargeArray,
    "test_large_rejected_after_shape_reassignment",
    test_large_rejected_after_shape_reassignment,
    devices=devices,
)
add_function_test(
    TestLargeArray,
    "test_small_descriptor_cached_and_rebuilt",
    test_small_descriptor_cached_and_rebuilt,
    devices=devices,
)
add_function_test(
    TestLargeArray, "test_large_overriding_subclass_rejected", test_large_overriding_subclass_rejected, devices=devices
)
add_function_test(
    TestLargeArray,
    "test_large_subclass_still_accepted_by_largearray",
    test_large_subclass_still_accepted_by_largearray,
    devices=devices,
)
add_function_test(
    TestLargeArray,
    "test_stale_descriptor_after_shape_reassignment",
    test_stale_descriptor_after_shape_reassignment,
    devices=devices,
)
add_function_test(TestLargeArray, "test_shape_limits", test_shape_limits, devices=devices)
add_function_test(
    TestLargeArray, "test_large_rejected_by_32bit_paths", test_large_rejected_by_32bit_paths, devices=devices
)
add_function_test(
    TestLargeArray,
    "test_large_rejected_after_descriptor_reset",
    test_large_rejected_after_descriptor_reset,
    devices=devices,
)
add_function_test(TestLargeArray, "test_large_rejected_elsewhere", test_large_rejected_elsewhere, devices=devices)
add_function_test(TestLargeArray, "test_large_rejected_by_volume", test_large_rejected_by_volume, devices=devices)
add_function_test(
    TestLargeArray,
    "test_large_rejected_by_mesh_and_hash_grid",
    test_large_rejected_by_mesh_and_hash_grid,
    devices=devices,
)
add_function_test(
    TestLargeArray, "test_large_noncontiguous_fill_and_copy", test_large_noncontiguous_fill_and_copy, devices=devices
)
add_function_test(TestLargeArray, "test_large_copy_on_tape_rejected", test_large_copy_on_tape_rejected, devices=devices)
add_function_test(TestLargeArray, "test_bsr_transpose_nnz_limit", test_bsr_transpose_nnz_limit, devices=devices)
add_function_test(TestLargeArray, "test_large_rejected_by_texture", test_large_rejected_by_texture, devices=devices)

add_function_test(TestLargeArray, "test_ctype64", test_ctype64, devices=devices)
add_function_test(
    TestLargeArray, "test_launch_accepts_small_and_large", test_launch_accepts_small_and_large, devices=devices
)
add_function_test(TestLargeArray, "test_launch_mixed_signature", test_launch_mixed_signature, devices=devices)
add_function_test(TestLargeArray, "test_launch_type_errors", test_launch_type_errors, devices=devices)
add_function_test(TestLargeArray, "test_launch_array_subclass", test_launch_array_subclass, devices=devices)
add_function_test(TestLargeArray, "test_large_into_array_param", test_large_into_array_param, devices=devices)
add_function_test(TestLargeArray, "test_largearray_type_identity", test_largearray_type_identity, devices=devices)
add_function_test(TestLargeArray, "test_read_write", test_read_write, devices=devices)
add_function_test(TestLargeArray, "test_index_types", test_index_types, devices=devices)
add_function_test(TestLargeArray, "test_negative_index", test_negative_index, devices=devices)
add_function_test(TestLargeArray, "test_len_shape", test_len_shape, devices=devices)
add_function_test(
    TestLargeArray, "test_large_literal_index_rejected", test_large_literal_index_rejected, devices=devices
)
add_function_test(
    TestLargeArray, "test_small_literal_index_accepted", test_small_literal_index_accepted, devices=devices
)
add_function_test(TestLargeArray, "test_atomics", test_atomics, devices=devices)
add_function_test(TestLargeArray, "test_dtypes", test_dtypes, devices=devices)
add_function_test(TestLargeArray, "test_strided", test_strided, devices=devices)
add_function_test(TestLargeArray, "test_func_param", test_func_param, devices=devices)
add_function_test(TestLargeArray, "test_closed_type", test_closed_type, devices=devices)
add_function_test(TestLargeArray, "test_unsupported", test_unsupported, devices=devices)
add_function_test(TestLargeArray, "test_return_type_mismatch", test_return_type_mismatch, devices=devices)
add_function_test(TestLargeArray, "test_uint32_index_small_array", test_uint32_index_small_array, devices=devices)
add_function_test(TestLargeArray, "test_null_array", test_null_array, devices=devices)
add_function_test(TestLargeArray, "test_null_check", test_null_check, devices=devices)

add_function_test(TestLargeArray, "test_grad_read", test_grad_read, devices=devices)
add_function_test(TestLargeArray, "test_grad_write", test_grad_write, devices=devices)
add_function_test(TestLargeArray, "test_grad_atomic_add", test_grad_atomic_add, devices=devices)
add_function_test(TestLargeArray, "test_grad_shape", test_grad_shape, devices=devices)
add_function_test(TestLargeArray, "test_grad_composite", test_grad_composite, devices=devices)
add_function_test(TestLargeArray, "test_grad_func", test_grad_func, devices=devices)
add_function_test(TestLargeArray, "test_tape_zero", test_tape_zero, devices=devices)
add_function_test(TestLargeArray, "test_grad_replaced", test_grad_replaced, devices=devices)
add_function_test(TestLargeArray, "test_enable_backward_false", test_enable_backward_false, devices=devices)

add_function_test(TestLargeArray, "test_deterministic", test_deterministic, devices=devices)
add_function_test(TestLargeArray, "test_jets", test_jets, devices=devices)
add_function_test(TestLargeArray, "test_graph_capture", test_graph_capture, devices=cuda_devices)
add_function_test(TestLargeArray, "test_apic_rejected", test_apic_rejected, devices=devices)
add_function_test(
    TestLargeArray, "test_verify_autograd_array_access", test_verify_autograd_array_access, devices=devices
)
add_function_test(TestLargeArray, "test_llvm_cuda", test_llvm_cuda, devices=cuda_devices)
add_function_test(TestLargeArray, "test_record_cmd", test_record_cmd, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
