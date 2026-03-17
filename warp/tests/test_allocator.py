# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp._src.context import (
    Allocator,
)

wp.init()


class TestAllocator(unittest.TestCase):
    def test_protocol_conformance_cpu(self):
        """Built-in CPU allocators satisfy the Allocator protocol."""
        wp.init()
        cpu = wp.get_device("cpu")
        self.assertIsInstance(cpu.default_allocator, Allocator)
        self.assertIsInstance(cpu.pinned_allocator, Allocator)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
    def test_protocol_conformance_cuda(self):
        """Built-in CUDA allocators satisfy the Allocator protocol."""
        wp.init()
        device = wp.get_device("cuda:0")
        self.assertIsInstance(device.default_allocator, Allocator)
        if device.is_mempool_supported:
            self.assertIsInstance(device.mempool_allocator, Allocator)


class _CountingAllocator:
    """Test allocator that counts allocations and delegates to the built-in."""

    def __init__(self, device):
        self._inner = device.default_allocator
        self.alloc_count = 0
        self.dealloc_count = 0

    def allocate(self, size_in_bytes):
        self.alloc_count += 1
        return self._inner.allocate(size_in_bytes)

    def deallocate(self, ptr, size_in_bytes):
        self.dealloc_count += 1
        self._inner.deallocate(ptr, size_in_bytes)


class TestCustomAllocator(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
    def test_set_allocator(self):
        """set_allocator routes array allocations through the custom allocator."""
        device = wp.get_device("cuda:0")
        alloc = _CountingAllocator(device)
        wp.set_allocator(alloc)
        try:
            a = wp.zeros(100, dtype=wp.float32, device=device)
            b = wp.empty(100, dtype=wp.float32, device=device)
            c = wp.full(100, value=1.0, dtype=wp.float32, device=device)
            self.assertEqual(alloc.alloc_count, 3)
            del a
            del b
            del c
            self.assertEqual(alloc.dealloc_count, 3)
        finally:
            wp.set_allocator(None)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
    def test_set_device_allocator(self):
        """set_device_allocator sets allocator on a specific device."""
        device = wp.get_device("cuda:0")
        alloc = _CountingAllocator(device)
        wp.set_device_allocator(device, alloc)
        try:
            a = wp.zeros(100, dtype=wp.float32, device=device)
            self.assertEqual(alloc.alloc_count, 1)
        finally:
            wp.set_device_allocator(device, None)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
    def test_set_device_allocator_cpu_raises(self):
        """set_device_allocator raises for CPU devices."""
        with self.assertRaises(RuntimeError):
            wp.set_device_allocator("cpu", _CountingAllocator(wp.get_device("cuda:0")))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
    def test_get_device_allocator(self):
        """get_device_allocator returns the effective allocator."""
        device = wp.get_device("cuda:0")
        alloc = _CountingAllocator(device)
        wp.set_device_allocator(device, alloc)
        try:
            self.assertIs(wp.get_device_allocator(device), alloc)
        finally:
            wp.set_device_allocator(device, None)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
    def test_reset_to_default(self):
        """set_allocator(None) restores the built-in allocator."""
        device = wp.get_device("cuda:0")
        original = wp.get_device_allocator(device)
        wp.set_allocator(_CountingAllocator(device))
        wp.set_allocator(None)
        self.assertIs(wp.get_device_allocator(device), original)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
    def test_scoped_allocator(self):
        """ScopedAllocator restores the previous allocator on exit."""
        device = wp.get_device("cuda:0")
        alloc = _CountingAllocator(device)
        original = wp.get_device_allocator(device)
        with wp.ScopedAllocator(device, alloc):
            self.assertIs(wp.get_device_allocator(device), alloc)
            wp.zeros(10, dtype=wp.float32, device=device)
            self.assertEqual(alloc.alloc_count, 1)
        self.assertIs(wp.get_device_allocator(device), original)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
    def test_scoped_allocator_restores_on_exception(self):
        """ScopedAllocator restores allocator even if body raises."""
        device = wp.get_device("cuda:0")
        alloc = _CountingAllocator(device)
        original = wp.get_device_allocator(device)
        with self.assertRaises(ValueError):
            with wp.ScopedAllocator(device, alloc):
                raise ValueError("test")
        self.assertIs(wp.get_device_allocator(device), original)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
    def test_allocator_swap_with_live_arrays(self):
        """Arrays allocated with a custom allocator survive allocator reset."""
        device = wp.get_device("cuda:0")
        alloc = _CountingAllocator(device)
        wp.set_device_allocator(device, alloc)
        a = wp.zeros(100, dtype=wp.float32, device=device)
        self.assertEqual(alloc.alloc_count, 1)
        wp.set_device_allocator(device, None)
        del a
        self.assertEqual(alloc.dealloc_count, 1)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
    def test_allocate_failure(self):
        """Allocation failure in custom allocator propagates cleanly."""
        device = wp.get_device("cuda:0")

        class _FailAllocator:
            def allocate(self, size_in_bytes):
                raise RuntimeError("deliberate failure")

            def deallocate(self, ptr, size_in_bytes):
                pass

        wp.set_device_allocator(device, _FailAllocator())
        try:
            with self.assertRaises(RuntimeError):
                wp.zeros(100, dtype=wp.float32, device=device)
        finally:
            wp.set_device_allocator(device, None)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
    def test_zero_size_allocation(self):
        """Custom allocator is not invoked for zero-size arrays."""
        device = wp.get_device("cuda:0")
        alloc = _CountingAllocator(device)
        wp.set_device_allocator(device, alloc)
        try:
            a = wp.zeros(0, dtype=wp.float32, device=device)
            self.assertEqual(alloc.alloc_count, 0)
            del a
            self.assertEqual(alloc.dealloc_count, 0)
        finally:
            wp.set_device_allocator(device, None)

    @unittest.skipUnless(wp.get_cuda_device_count() >= 2, "Multi-GPU not available")
    def test_set_allocator_broadcasts_to_all_devices(self):
        """set_allocator applies the allocator to every available CUDA device."""
        dev0 = wp.get_device("cuda:0")
        dev1 = wp.get_device("cuda:1")
        alloc = _CountingAllocator(dev0)
        wp.set_allocator(alloc)
        try:
            self.assertIs(wp.get_device_allocator(dev0), alloc)
            self.assertIs(wp.get_device_allocator(dev1), alloc)
        finally:
            wp.set_allocator(None)

    @unittest.skipUnless(wp.get_cuda_device_count() >= 2, "Multi-GPU not available")
    def test_per_device_isolation(self):
        """Setting allocator on one device does not affect another."""
        dev0 = wp.get_device("cuda:0")
        dev1 = wp.get_device("cuda:1")
        alloc0 = _CountingAllocator(dev0)
        alloc1 = _CountingAllocator(dev1)
        wp.set_device_allocator(dev0, alloc0)
        wp.set_device_allocator(dev1, alloc1)
        try:
            wp.zeros(100, dtype=wp.float32, device=dev0)
            wp.zeros(200, dtype=wp.float32, device=dev1)
            self.assertEqual(alloc0.alloc_count, 1)
            self.assertEqual(alloc1.alloc_count, 1)
        finally:
            wp.set_device_allocator(dev0, None)
            wp.set_device_allocator(dev1, None)


try:
    import rmm

    rmm_available = True
except ImportError:
    rmm_available = False


class TestRmmAllocator(unittest.TestCase):
    @unittest.skipUnless(rmm_available, "rmm not installed")
    @unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
    def test_rmm_allocator_basic(self):
        """RmmAllocator routes allocations through RMM."""
        rmm.reinitialize(pool_allocator=True, initial_pool_size=2**26)

        alloc = wp.RmmAllocator()
        device = wp.get_device("cuda:0")
        wp.set_device_allocator(device, alloc)
        try:
            a = wp.zeros(1000, dtype=wp.float32, device=device)
            self.assertIn(a.ptr, alloc._buffers)
            a.fill_(42.0)
            np.testing.assert_allclose(a.numpy(), 42.0)
            ptr = a.ptr
            del a
            self.assertNotIn(ptr, alloc._buffers)
        finally:
            wp.set_device_allocator(device, None)

    @unittest.skipUnless(rmm_available, "rmm not installed")
    @unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
    def test_rmm_allocator_interop_torch(self):
        """RMM-allocated Warp array can be exported to PyTorch."""
        try:
            import torch  # noqa: F401, PLC0415
        except ImportError:
            self.skipTest("torch not installed")

        rmm.reinitialize(pool_allocator=True, initial_pool_size=2**26)

        alloc = wp.RmmAllocator()
        device = wp.get_device("cuda:0")
        wp.set_device_allocator(device, alloc)
        try:
            a = wp.zeros(100, dtype=wp.float32, device=device)
            a.fill_(7.0)
            t = wp.to_torch(a)
            self.assertEqual(t.shape[0], 100)
            np.testing.assert_allclose(t.cpu().numpy(), 7.0)
        finally:
            wp.set_device_allocator(device, None)

    @unittest.skipUnless(rmm_available, "rmm not installed")
    @unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
    def test_rmm_allocator_stream(self):
        """RmmAllocator uses the provided RMM stream for stream-ordered allocation."""
        try:
            from cuda.bindings import runtime as cudart  # noqa: PLC0415
            from rmm.pylibrmm.stream import Stream as RmmStream  # noqa: PLC0415
        except ImportError:
            self.skipTest("cuda.bindings not available")

        rmm.reinitialize(pool_allocator=True, initial_pool_size=2**26)

        device = wp.get_device("cuda:0")
        cuda_stream = cudart.cudaStream_t(int(device.stream.cuda_stream))
        rmm_stream = RmmStream(obj=cuda_stream)

        alloc = wp.RmmAllocator(stream=rmm_stream)
        self.assertIs(alloc._stream, rmm_stream)

        wp.set_device_allocator(device, alloc)
        try:
            a = wp.zeros(1000, dtype=wp.float32, device=device)
            self.assertIn(a.ptr, alloc._buffers)
            np.testing.assert_allclose(a.numpy(), 0.0)
        finally:
            wp.set_device_allocator(device, None)

    @unittest.skipUnless(rmm_available, "rmm not installed")
    @unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
    def test_rmm_allocator_double_free(self):
        """deallocate raises RuntimeError for an already-freed or unknown pointer."""
        alloc = wp.RmmAllocator()
        ptr = alloc.allocate(256)
        self.assertIn(ptr, alloc._buffers)
        alloc.deallocate(ptr, 256)
        self.assertNotIn(ptr, alloc._buffers)
        # Second call with the same pointer must raise.
        with self.assertRaises(RuntimeError):
            alloc.deallocate(ptr, 256)
        # Completely unknown pointer must also raise.
        with self.assertRaises(RuntimeError):
            alloc.deallocate(0xDEADBEEF, 128)

    @unittest.skipIf(rmm_available, "rmm is installed")
    def test_rmm_allocator_import_error(self):
        """RmmAllocator raises ImportError when rmm is not installed."""
        with self.assertRaises(ImportError):
            wp.RmmAllocator()


if __name__ == "__main__":
    unittest.main()
