# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test that a kernel that fails to build fails its module, and says why."""

import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from importlib import util

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


def _import_module(code: str, name: str | None = None):
    """Import a throwaway Warp module from source.

    The kernels under test have to live in a module of their own. Defining them
    in this file would fail the test file's own module, and the point of these
    tests is what a build failure does to the module that contains it.

    Args:
        code: The module's source.
        name: The module's name. Defaults to a unique name, so that the module
            shares nothing with other tests, including kernel cache entries.
    """
    if name is None:
        name = f"_test_module_contamination_{uuid.uuid4().hex[:12]}"

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


_VALIDATION_FAILURE_SOURCE = """\
import warp as wp

@wp.func
def bad_return_type(x: int) -> tuple[int, int, int]:
    # annotated as 3 elements, returns 2
    return (x + x, x * x)

@wp.kernel
def bad_kernel():
    _x, _y, _z = bad_return_type(123)

@wp.kernel
def sibling_kernel(a: wp.array[float]):
    i = wp.tid()
    a[i] = 7.0
"""

_NATIVE_FAILURE_SOURCE = '''\
import warp as wp

snippet = """
    not valid C++ #### ;;; @@@
"""

@wp.func_native(snippet)
def broken_native(a: wp.array[float], tid: int):
    ...

@wp.kernel
def bad_kernel(a: wp.array[float]):
    tid = wp.tid()
    broken_native(a, tid)

@wp.kernel
def sibling_kernel(a: wp.array[float]):
    i = wp.tid()
    a[i] = 7.0
'''

_ANNOTATION_ERROR = r"has its return type annotated as a tuple of 3 elements but the code returns 2 values"


def test_validation_failure_names_the_annotation(test, device):
    """Verify a failed return-type validation reports the annotation, not C++ fallout.

    Validation runs during build(), before any C++ referencing the function is
    emitted, so the error names the bad annotation. Emitting the call first would
    surface this as an undeclared identifier from the native compiler instead.
    """
    module = _import_module(_VALIDATION_FAILURE_SOURCE)

    with test.assertRaisesRegex(wp.WarpCodegenError, _ANNOTATION_ERROR):
        wp.launch(module.bad_kernel, dim=1, device=device)


def test_validation_failure_fails_the_module(test, device):
    """Verify a sibling of a kernel that failed validation reports that failure.

    A module is the compilation unit, so a kernel that cannot build fails the
    module rather than being dropped from it. Launching any other kernel in that
    module has to report the failure instead of returning as though it had run.
    """
    module = _import_module(_VALIDATION_FAILURE_SOURCE)
    a = wp.zeros(4, dtype=float, device=device)

    with test.assertRaisesRegex(wp.WarpCodegenError, _ANNOTATION_ERROR):
        wp.launch(module.bad_kernel, dim=1, device=device)

    with test.assertRaisesRegex(wp.WarpCodegenError, _ANNOTATION_ERROR):
        wp.launch(module.sibling_kernel, dim=4, inputs=[a], device=device)


def test_validation_failure_repeats_on_relaunch(test, device):
    """Verify a build failure keeps failing rather than reporting only once.

    The failure is reported from the recorded error on every later launch, so a
    kernel relaunched in a loop cannot quietly start running partial state.
    """
    module = _import_module(_VALIDATION_FAILURE_SOURCE)

    for _ in range(2):
        with test.assertRaisesRegex(wp.WarpCodegenError, _ANNOTATION_ERROR):
            wp.launch(module.bad_kernel, dim=1, device=device)


def test_native_failure_fails_the_module(test, device):
    """Verify a native compile failure behaves the same way as a codegen failure.

    The snippet is not valid C++, so Warp's own codegen succeeds and the native
    compiler fails instead. Both classes fail the module, so a sibling reports
    the failure rather than skipping.
    """
    module = _import_module(_NATIVE_FAILURE_SOURCE)
    a = wp.zeros(4, dtype=float, device=device)

    with test.assertRaises(Exception) as caught:
        wp.launch(module.bad_kernel, dim=4, inputs=[a], device=device)

    with test.assertRaises(type(caught.exception)):
        wp.launch(module.sibling_kernel, dim=4, inputs=[a], device=device)


def _redefine(module, code: str):
    """Run more source in an imported module's namespace, as running a notebook cell again does.

    Warp reads a kernel's source from its file when the kernel is defined, so the
    code has to come from a real file, which can be removed afterwards.
    """
    file, file_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(file, "w") as f:
            f.write(code)

        exec(compile(code, file_path, "exec"), module.__dict__)
    finally:
        os.remove(file_path)


_REDEFINITION_SOURCE = """\
import warp as wp

@wp.kernel
def exercise_kernel(a: wp.array[float]):
    i = wp.tid()
    a[i] = None  # unfinished, fails codegen

@wp.kernel
def sibling_kernel(a: wp.array[float]):
    i = wp.tid()
    a[i] = 7.0
"""

_FIXED_EXERCISE_SOURCE = """\
@wp.kernel
def exercise_kernel(a: wp.array[float]):
    i = wp.tid()
    a[i] = 2.0 * float(i)
"""

_STILL_FAILING_EXERCISE_SOURCE = """\
@wp.kernel
def exercise_kernel(a: wp.array[float]):
    i = wp.tid()
    a[i] = undefined_name
"""

_EXERCISE_ERROR = r'Error while parsing function "exercise_kernel"'


def test_redefinition_recovers_failed_module(test, device):
    """Verify redefining a kernel that failed to build lets its module build again.

    IPython keeps the last exception for ``%debug``, and its traceback references
    the failed definition, so the failed definition outlives a notebook cell that
    redefines the kernel. It must not keep failing the module after the fix.
    """
    module = _import_module(_REDEFINITION_SOURCE)
    a = wp.zeros(4, dtype=float, device=device)

    with test.assertRaisesRegex(wp.WarpCodegenTypeError, _EXERCISE_ERROR):
        wp.launch(module.exercise_kernel, dim=4, inputs=[a], device=device)

    # Keep the failed definition alive, as a retained traceback would.
    failed_definition = module.exercise_kernel

    _redefine(module, _FIXED_EXERCISE_SOURCE)
    test.assertIsNot(module.exercise_kernel, failed_definition)

    wp.launch(module.exercise_kernel, dim=4, inputs=[a], device=device)
    assert_np_equal(a.numpy(), np.array([0.0, 2.0, 4.0, 6.0]))

    wp.launch(module.sibling_kernel, dim=4, inputs=[a], device=device)
    assert_np_equal(a.numpy(), np.full(4, 7.0))


def test_replaced_failed_definition_reports_its_error(test, device):
    """Verify a failed definition launched after its replacement still reports its own error.

    Launching an earlier definition returns it to the module, so it fails as it
    would have without the redefinition.
    """
    module = _import_module(_REDEFINITION_SOURCE)
    a = wp.zeros(4, dtype=float, device=device)

    with test.assertRaisesRegex(wp.WarpCodegenTypeError, _EXERCISE_ERROR):
        wp.launch(module.exercise_kernel, dim=4, inputs=[a], device=device)

    failed_definition = module.exercise_kernel
    _redefine(module, _FIXED_EXERCISE_SOURCE)

    for _ in range(2):
        with test.assertRaisesRegex(wp.WarpCodegenTypeError, _EXERCISE_ERROR):
            wp.launch(failed_definition, dim=4, inputs=[a], device=device)


def test_failing_redefinition_fails_the_module(test, device):
    """Verify a redefinition that also fails to build fails the module with its own error."""
    module = _import_module(_REDEFINITION_SOURCE)
    a = wp.zeros(4, dtype=float, device=device)

    with test.assertRaisesRegex(wp.WarpCodegenTypeError, _EXERCISE_ERROR):
        wp.launch(module.exercise_kernel, dim=4, inputs=[a], device=device)

    # Keep the failed definition alive, as a retained traceback would.
    failed_definition = module.exercise_kernel
    _redefine(module, _STILL_FAILING_EXERCISE_SOURCE)

    with test.assertRaisesRegex(wp.WarpCodegenKeyError, "undefined_name"):
        wp.launch(module.exercise_kernel, dim=4, inputs=[a], device=device)

    with test.assertRaisesRegex(wp.WarpCodegenKeyError, "undefined_name"):
        wp.launch(module.sibling_kernel, dim=4, inputs=[a], device=device)


_TWO_EXERCISES_SOURCE = """\
import warp as wp

@wp.kernel
def first_exercise(a: wp.array[float]):
    i = wp.tid()
    a[i] = None  # unfinished, fails codegen

@wp.kernel
def second_exercise(a: wp.array[float]):
    i = wp.tid()
    a[i] = None  # unfinished, fails codegen
"""

_FIXED_EXERCISES_SOURCE = """\
@wp.kernel
def first_exercise(a: wp.array[float]):
    i = wp.tid()
    a[i] = 1.0

@wp.kernel
def second_exercise(a: wp.array[float]):
    i = wp.tid()
    a[i] = 2.0
"""


def test_redefining_failed_kernels_together_recovers(test, device):
    """Verify fixing two failed kernels in one notebook cell recovers after one run of the cell.

    A build stops at the first kernel that fails, so when the cell redefines both
    kernels, only one of them is known to have failed.
    """
    module = _import_module(_TWO_EXERCISES_SOURCE)
    a = wp.zeros(4, dtype=float, device=device)

    with test.assertRaises(wp.WarpCodegenTypeError):
        wp.launch(module.first_exercise, dim=4, inputs=[a], device=device)

    # Keep both failed definitions alive, as a retained traceback would.
    failed_definitions = (module.first_exercise, module.second_exercise)

    _redefine(module, _FIXED_EXERCISES_SOURCE)

    wp.launch(module.first_exercise, dim=4, inputs=[a], device=device)
    assert_np_equal(a.numpy(), np.full(4, 1.0))

    wp.launch(module.second_exercise, dim=4, inputs=[a], device=device)
    assert_np_equal(a.numpy(), np.full(4, 2.0))


_BLOCK_DIM_SOURCE = """\
import warp as wp

@wp.kernel
def tile_kernel(a: wp.array[float]):
    t = wp.tile(1.0)
    wp.tile_store(a, wp.tile_reshape(t, shape=(256,)))  # builds only with block_dim=256
"""

_REDEFINED_TILE_SOURCE = """\
@wp.kernel
def tile_kernel(a: wp.array[float]):
    i = wp.tid()
    a[i] = 2.0
"""


def test_replaced_definition_keeps_working_block_dims(test, device):
    """Verify an earlier definition that failed at one block_dim still builds at another after its replacement.

    Each block_dim builds its own variant of the module, so launching the
    set-aside definition at a block_dim where it builds succeeds.
    """
    module = _import_module(_BLOCK_DIM_SOURCE)
    a = wp.zeros(256, dtype=float, device=device)

    earlier_definition = module.tile_kernel
    with test.assertRaisesRegex(ValueError, 'Error while parsing function "tile_kernel"'):
        wp.launch(earlier_definition, dim=64, inputs=[a], block_dim=64, device=device)

    _redefine(module, _REDEFINED_TILE_SOURCE)

    wp.launch(earlier_definition, dim=256, inputs=[a], block_dim=256, device=device)
    assert_np_equal(a.numpy(), np.full(256, 1.0))


_HELPER_SOURCE = """\
import warp as wp

@wp.func
def helper():
    return 3.0

@wp.kernel
def uses_helper(a: wp.array[float]):
    i = wp.tid()
    a[i] = helper() + 0.5
"""


def test_repaired_helper_restores_earlier_definition(test, device):
    """Verify an earlier definition works again once a helper it calls is repaired.

    An earlier definition that built stays in the module, so a helper change
    that breaks it fails the module, and repairing the helper fixes both.
    """
    module = _import_module(_HELPER_SOURCE)
    a = wp.zeros(4, dtype=float, device=device)

    wp.launch(module.uses_helper, dim=4, inputs=[a], device=device)
    assert_np_equal(a.numpy(), np.full(4, 3.5))

    earlier_definition = module.uses_helper
    _redefine(module, "@wp.kernel\ndef uses_helper(a: wp.array[float]):\n    i = wp.tid()\n    a[i] = 1.0\n")

    # An int helper breaks the earlier definition, which still adds 0.5.
    _redefine(module, "@wp.func\ndef helper():\n    return 3\n")
    with test.assertRaisesRegex(RuntimeError, 'Error while parsing function "uses_helper"'):
        wp.launch(module.uses_helper, dim=4, inputs=[a], device=device)

    _redefine(module, "@wp.func\ndef helper():\n    return 3.0\n")
    wp.launch(earlier_definition, dim=4, inputs=[a], device=device)
    assert_np_equal(a.numpy(), np.full(4, 3.5))


_GENERIC_SOURCE = """\
import warp as wp
from typing import Any

@wp.kernel
def generic_exercise(a: wp.array[Any]):
    i = wp.tid()
    a[i] = None  # unfinished, fails codegen
"""


def test_new_overload_of_replaced_generic_definition_reports_its_error(test, device):
    """Verify a new overload of a replaced generic definition reports the definition's own error."""
    module = _import_module(_GENERIC_SOURCE)
    earlier_definition = module.generic_exercise
    _redefine(module, "@wp.kernel\ndef generic_exercise(a: wp.array[Any]):\n    i = wp.tid()\n    a[i] = a.dtype(1)\n")

    for dtype in (float, int):
        a = wp.zeros(4, dtype=dtype, device=device)
        with test.assertRaisesRegex(wp.WarpCodegenTypeError, 'Error while parsing function "generic_exercise"'):
            wp.launch(earlier_definition, dim=4, inputs=[a], device=device)


_CLOSURE_SOURCE = """\
import warp as wp

def make_fill_kernel(value):
    @wp.kernel
    def fill(a: wp.array[float]):
        i = wp.tid()
        a[i] = value

    return fill

fill_failing = make_fill_kernel("not a number")
"""


def test_failed_closure_is_not_replaced_by_another_closure(test, device):
    """Verify a kernel created by the same function does not count as a redefinition.

    Kernels a function creates on each call share its source and key. A failing
    one is still live and still fails the module when the next one is created.
    """
    module = _import_module(_CLOSURE_SOURCE)
    a = wp.zeros(4, dtype=float, device=device)

    with test.assertRaises(Exception) as caught:
        wp.launch(module.fill_failing, dim=4, inputs=[a], device=device)

    fill_two = module.make_fill_kernel(2.0)
    test.assertEqual(fill_two.key, module.fill_failing.key)

    with test.assertRaises(type(caught.exception)):
        wp.launch(fill_two, dim=4, inputs=[a], device=device)


_RECORDED_SOURCE = """\
import warp as wp

@wp.kernel
def recorded_kernel(a: wp.array[float]):
    pass
"""


def test_recorded_launch_of_replaced_definition_resizes(test, device):
    """Verify a recorded launch resizes after a redefinition sets its kernel aside.

    Resizing past the ``wp.tid()`` extent limit reads the kernel's build
    metadata, which returns the set-aside definition to the module.
    """
    module = _import_module(_RECORDED_SOURCE)
    a = wp.zeros(4, dtype=float, device=device)
    cmd = wp.launch(module.recorded_kernel, dim=4, inputs=[a], device=device, record_cmd=True)

    # A failing sibling puts the recorded kernel in a failed build.
    _redefine(module, "@wp.kernel\ndef failing_kernel(a: wp.array[float]):\n    i = wp.tid()\n    a[i] = None\n")
    with test.assertRaises(wp.WarpCodegenTypeError):
        wp.launch(module.failing_kernel, dim=4, inputs=[a], device=device)

    _redefine(module, "@wp.kernel\ndef recorded_kernel(a: wp.array[float]):\n    a[0] = 1.0\n")

    # The recorded kernel does not use wp.tid(), so it accepts extents past the limit.
    cmd.set_dim(2**32)
    test.assertEqual(cmd.bounds.size, 2**32)


_DEFERRED_STATIC_SOURCE = """\
import warp as wp

values = [{value}]

@wp.func
def helper():
    return 3

@wp.kernel
def deferred_static_kernel(a: wp.array[float]):
    x = helper() + 0.5  # an int helper fails the build before the loop
    for i in range(1):
        a[i] = wp.static(values[i])
"""


def _launch_restored_deferred_static(value: float) -> float:
    """Return what a restored definition writes from a ``wp.static()`` that depends on a loop variable.

    The definition fails to build before its static is resolved, and its
    replacement then builds without it. The module name is fixed, so processes
    that share a kernel cache build the same module.
    """
    module = _import_module(
        _DEFERRED_STATIC_SOURCE.format(value=value), name="_test_module_contamination_deferred_static"
    )
    a = wp.zeros(1, dtype=float, device="cpu")

    try:
        wp.launch(module.deferred_static_kernel, dim=1, inputs=[a], device="cpu")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected the int helper to fail the build")

    earlier_definition = module.deferred_static_kernel
    _redefine(module, "@wp.kernel\ndef deferred_static_kernel(a: wp.array[float]):\n    a[0] = 1.0\n")
    wp.launch(module.deferred_static_kernel, dim=1, inputs=[a], device="cpu")

    _redefine(module, "@wp.func\ndef helper():\n    return 3.0\n")
    wp.launch(earlier_definition, dim=1, inputs=[a], device="cpu")
    return float(a.numpy()[0])


class TestModuleContamination(unittest.TestCase):
    @unittest.skipUnless(wp.is_cpu_available(), "Requires CPU support")
    def test_restored_definition_does_not_reuse_other_static_values(self):
        """Verify a restored definition does not reuse a binary built for different ``wp.static()`` values.

        Each value runs in a fresh process, and the processes share a new kernel cache.
        """
        with tempfile.TemporaryDirectory() as cache_path:
            env = os.environ.copy()
            env["WARP_CACHE_PATH"] = cache_path
            for value in (10.0, 20.0):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "from warp.tests.test_module_contamination import _launch_restored_deferred_static; "
                        f"print(_launch_restored_deferred_static({value}))",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=180,
                )

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines()[-1], str(value))


devices = get_test_devices()

for _func in (
    test_validation_failure_names_the_annotation,
    test_validation_failure_fails_the_module,
    test_validation_failure_repeats_on_relaunch,
    test_native_failure_fails_the_module,
    test_redefinition_recovers_failed_module,
    test_replaced_failed_definition_reports_its_error,
    test_failing_redefinition_fails_the_module,
    test_redefining_failed_kernels_together_recovers,
    test_repaired_helper_restores_earlier_definition,
    test_new_overload_of_replaced_generic_definition_reports_its_error,
    test_failed_closure_is_not_replaced_by_another_closure,
    test_recorded_launch_of_replaced_definition_resizes,
):
    add_function_test(TestModuleContamination, func=_func, name=_func.__name__, devices=devices)

add_function_test(
    TestModuleContamination,
    func=test_replaced_definition_keeps_working_block_dims,
    name="test_replaced_definition_keeps_working_block_dims",
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
