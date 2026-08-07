# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test that validation failures don't contaminate shared modules with invalid code."""

import os
import tempfile
import unittest
import uuid
from importlib import util

import warp as wp
from warp.tests.unittest_utils import *


def test_function_validation_failure_contamination(test, device):
    """Test that function validation failures don't contaminate modules.

    This test creates two scenarios in the same test module:
    1. A kernel that calls a function with invalid return type annotation
    2. A valid kernel that should work

    Without the fix, both kernels end up in the same module hash, and when
    the first kernel's function fails validation, it leaves undefined
    references that break C++ compilation of the entire module, causing
    the second kernel to fail too.
    """

    # First kernel: calls a function that will fail validation
    @wp.func
    def bad_return_type(x: int) -> tuple[int, int, int]:
        # Returns 2 values but annotation says 3 - validation will fail
        return (x + x, x * x)

    def bad_kernel_fn():
        _x, _y, _z = bad_return_type(123)

    # Second kernel: completely valid, should always work
    @wp.kernel
    def good_kernel():
        x = 1.0
        y = 2.0
        wp.expect_eq(x + y, 3.0)

    # The bad kernel should fail with WarpCodegenError
    bad_kernel = wp.Kernel(func=bad_kernel_fn)
    with test.assertRaisesRegex(
        wp.WarpCodegenError,
        r"has its return type annotated as a tuple of 3 elements but the code returns 2 values",
    ):
        wp.launch(bad_kernel, dim=1, device=device)

    # After the codegen failure, bad_kernel.adj.skip_build=True is set, which changes the
    # module hash (the failed kernel is excluded from the hash). Calling mark_modified()
    # clears the cached hash so the next load recomputes it and uses a different cache path.
    # Without this, on multi-GPU systems the second device would find the binary written
    # by the first device's successful good_kernel compilation and skip codegen entirely,
    # so the WarpCodegenError would never be raised for the subsequent devices.
    bad_kernel.module.mark_modified()

    # The good kernel should still work despite the bad kernel failure
    # This is the key test - without the fix, this will fail with
    # "use of undeclared identifier 'bad_return_type_1'" because both
    # kernels ended up in the same module and bad_return_type was never defined
    try:
        wp.launch(good_kernel, dim=1, device=device)
    except Exception as e:
        test.fail(f"good_kernel should not fail due to bad_kernel contamination, but got: {type(e).__name__}: {e}")


class TestModuleContamination(unittest.TestCase):
    pass


def _make_bad_kernel():
    """Import a fresh regular module whose kernel fails during codegen.

    A regular module is required here. Kernels declared with ``module="unique"``
    take a separate path that already clears the failed build state between
    devices.
    """
    name = f"_test_failed_build_{uuid.uuid4().hex[:12]}"
    code = """\
import warp as wp

@wp.kernel
def bad_kernel(a: wp.array[float]):
    i = wp.tid()
    a[i] = 1.0
    a[i] = wp.no_such_builtin_function(a[i])
"""
    file, file_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(file, "w") as f:
            f.write(code)

        spec = util.spec_from_file_location(name, file_path)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.remove(file_path)

    return module.bad_kernel


def test_codegen_failure_repeats_after_another_device(test, device):
    """Verify that a kernel that failed codegen elsewhere raises the same error here.

    ``skip_build`` lives on the ``Adjoint``, which is shared across devices, so a
    failure recorded on one device has to be reported on the rest rather than
    launching the partial state that failed build left behind.
    """
    others = [d for d in get_test_devices() if d != device]
    if not others:
        test.skipTest("Needs a second device to fail the build on first")

    kernel = _make_bad_kernel()

    with test.assertRaises(wp.WarpCodegenAttributeError):
        a = wp.zeros(4, dtype=float, device=others[0])
        wp.launch(kernel, dim=4, inputs=[a], device=others[0])

    with test.assertRaises(wp.WarpCodegenAttributeError):
        a = wp.zeros(4, dtype=float, device=device)
        wp.launch(kernel, dim=4, inputs=[a], device=device)


class TestFailedBuildRepeats(unittest.TestCase):
    """Verify that a build failure keeps failing the same way.

    A kernel that cannot be built must report the same error on every retry and
    on every device. Reporting it only the first time leaves later launches
    running whatever partial state the failed build left behind.
    """

    def test_codegen_failure_repeats_on_same_device(self):
        """Verify that relaunching a kernel that failed codegen raises again."""
        kernel = _make_bad_kernel()
        device = wp.get_device()
        a = wp.zeros(4, dtype=float, device=device)

        for attempt in range(2):
            with self.subTest(attempt=attempt), self.assertRaises(wp.WarpCodegenAttributeError):
                wp.launch(kernel, dim=4, inputs=[a], device=device)

    @staticmethod
    def _make_native_failure_module():
        """Import a fresh module whose native build fails, alongside a valid kernel.

        The snippet is not valid C++, so Warp's own codegen succeeds and the
        failure comes from the native compiler instead. That fails the whole
        module rather than a single kernel's adjoint.
        """
        name = f"_test_native_build_fail_{uuid.uuid4().hex[:12]}"
        code = '''\
import warp as wp

snippet = """
    not valid C++ #### ;;; @@@
"""

@wp.func_native(snippet)
def broken_native(a: wp.array[float], tid: int):
    ...

@wp.kernel
def native_kernel(a: wp.array[float]):
    tid = wp.tid()
    broken_native(a, tid)

@wp.kernel
def sibling_kernel(a: wp.array[float]):
    i = wp.tid()
    a[i] = 7.0
'''
        file, file_path = tempfile.mkstemp(suffix=".py")
        try:
            with os.fdopen(file, "w") as f:
                f.write(code)

            spec = util.spec_from_file_location(name, file_path)
            module = util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            os.remove(file_path)

        return module

    def test_native_build_failure_repeats_on_relaunch(self):
        """Verify that relaunching a kernel whose native build failed raises again."""
        module = self._make_native_failure_module()
        device = wp.get_device()
        a = wp.zeros(4, dtype=float, device=device)

        with self.assertRaises(Exception) as caught:
            wp.launch(module.native_kernel, dim=4, inputs=[a], device=device)

        with self.assertRaises(type(caught.exception)):
            wp.launch(module.native_kernel, dim=4, inputs=[a], device=device)

    def test_native_build_failure_reported_for_sibling_kernel(self):
        """Verify that a valid kernel reports its module's build failure instead of skipping."""
        module = self._make_native_failure_module()
        device = wp.get_device()
        a = wp.zeros(4, dtype=float, device=device)

        with self.assertRaises(Exception) as caught:
            wp.launch(module.native_kernel, dim=4, inputs=[a], device=device)

        # sibling_kernel is valid, but its module never built. Launching it has to
        # say so rather than returning as though the kernel had run.
        with self.assertRaises(type(caught.exception)):
            wp.launch(module.sibling_kernel, dim=4, inputs=[a], device=device)


devices = get_test_devices()
add_function_test(
    TestModuleContamination,
    func=test_function_validation_failure_contamination,
    name="test_function_validation_failure_contamination",
    devices=devices,
)
add_function_test(
    TestFailedBuildRepeats,
    func=test_codegen_failure_repeats_after_another_device,
    name="test_codegen_failure_repeats_after_another_device",
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
