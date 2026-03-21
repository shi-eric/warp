# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import unittest

import warp as wp
from warp._src.codegen import (
    VALID_COMPILE_GUARDS,
    compute_compile_guards,
    scan_source_for_guards,
)
from warp._src.context import ModuleBuilder, add_builtin, builtin_functions


class TestCompileGuards(unittest.TestCase):
    def test_add_builtin_validates_compile_guard(self):
        """Invalid compile_guard raises ValueError."""
        with self.assertRaises(ValueError):
            add_builtin(
                "_test_bad_guard",
                input_types={},
                value_type=int,
                compile_guard="WP_NO_FAKE",
            )

    def test_all_builtins_have_compile_guard(self):
        """Every registered builtin has a valid compile_guard."""
        for key, func in builtin_functions.items():
            if not hasattr(func, "overloads"):
                continue  # skip user-defined functions added via register_api_function
            for overload in func.overloads:
                self.assertIn(
                    overload.compile_guard,
                    VALID_COMPILE_GUARDS,
                    f"Builtin {key!r} has invalid compile_guard={overload.compile_guard!r}",
                )

    def test_guards_have_matching_cpp_ifndef(self):
        """Every guard in VALID_COMPILE_GUARDS has a #ifndef in the native headers."""
        native_dir = os.path.join(os.path.dirname(wp.__file__), "native")
        all_header_content = ""
        for path in glob.glob(os.path.join(native_dir, "*.h")):
            with open(path) as f:
                all_header_content += f.read()

        for guard in VALID_COMPILE_GUARDS:
            if guard is None:
                continue
            self.assertIn(
                f"#ifndef {guard}",
                all_header_content,
                f"Guard {guard!r} has no matching #ifndef in native headers",
            )

    def test_func_return_type_inspected_for_guards(self):
        """build_function must inspect return types, not just arg types."""

        # Track which types _inspect_type_for_guards is called with.
        inspected_types = []
        original_inspect = ModuleBuilder._inspect_type_for_guards

        def tracking_inspect(self, t):
            inspected_types.append(t)
            original_inspect(self, t)

        ModuleBuilder._inspect_type_for_guards = tracking_inspect
        try:
            builder = ModuleBuilder.__new__(ModuleBuilder)
            builder.functions = {}
            builder.structs = {}
            builder.deferred_functions = []
            builder.required_guards = set()
            builder.fatbins = {}
            builder.ltoirs = {}
            builder.ltoirs_decl = {}
            builder.shared_memory_bytes = {}
            builder.options = {"enable_backward": False}
            builder.module = None
            builder.kernels = []

            @wp.func
            def to_vec(x: float) -> wp.vec3:
                return wp.vec3(x, x, x)

            # Build the function — this should inspect args AND return type.
            builder.build_function(to_vec)

            # Verify _inspect_type_for_guards was called with the vec3 return
            # type.  The builtin resolution also adds WP_NO_VEC, but the
            # return type path is defense-in-depth for cases where builtins
            # can't provide the guard (e.g. types from external modules).
            inspected_generic_strs = [getattr(t, "_wp_generic_type_str_", None) for t in inspected_types]
            self.assertIn(
                "vec_t",
                inspected_generic_strs,
                "vec3 return type was not passed to _inspect_type_for_guards — "
                "build_function must inspect function return types",
            )
        finally:
            ModuleBuilder._inspect_type_for_guards = original_inspect

    # ------------------------------------------------------------------
    # compute_compile_guards() — dependency resolution
    # ------------------------------------------------------------------

    def test_compute_guards_empty_required(self):
        """No features required: all guards emitted."""
        result = compute_compile_guards(set())
        all_guards = VALID_COMPILE_GUARDS - {None}
        for g in all_guards:
            self.assertIn(f"#define {g}", result)

    def test_compute_guards_all_required(self):
        """All features required: no guards emitted."""
        all_guards = VALID_COMPILE_GUARDS - {None}
        result = compute_compile_guards(set(all_guards))
        self.assertEqual(result, "")

    def test_compute_guards_single_feature(self):
        """Requiring WP_NO_MESH prevents it from being emitted."""
        result = compute_compile_guards({"WP_NO_MESH"})
        self.assertNotIn("WP_NO_MESH", result)

    # ------------------------------------------------------------------
    # scan_source_for_guards() — safety-net source scanning
    # ------------------------------------------------------------------

    def test_scan_source_detects_vec(self):
        """Source containing vec3 triggers WP_NO_VEC."""
        required = set()
        scan_source_for_guards("wp::vec3 v = {};", required)
        self.assertIn("WP_NO_VEC", required)

    def test_scan_source_detects_mat(self):
        """Source containing mat_t< triggers WP_NO_MAT."""
        required = set()
        scan_source_for_guards("mat_t<3, 2, float32> m;", required)
        self.assertIn("WP_NO_MAT", required)

    def test_scan_source_detects_quat(self):
        """Source containing quat_t triggers WP_NO_QUAT."""
        required = set()
        scan_source_for_guards("quat_t<float32> q;", required)
        self.assertIn("WP_NO_QUAT", required)

    def test_scan_source_no_match(self):
        """Source with no type patterns leaves required empty."""
        required = set()
        scan_source_for_guards("float x = 1.0f;", required)
        self.assertEqual(required, set())

    def test_scan_source_idempotent(self):
        """Scanning with guard already present does not error."""
        required = {"WP_NO_VEC"}
        scan_source_for_guards("vec3 v;", required)
        self.assertIn("WP_NO_VEC", required)

    # ------------------------------------------------------------------
    # _inspect_type_for_guards() — type inspection
    # ------------------------------------------------------------------

    def _make_builder(self):
        """Create a minimal ModuleBuilder for testing type inspection."""
        builder = ModuleBuilder.__new__(ModuleBuilder)
        builder.required_guards = set()
        return builder

    def test_inspect_vec_type(self):
        builder = self._make_builder()
        builder._inspect_type_for_guards(wp.vec3)
        self.assertIn("WP_NO_VEC", builder.required_guards)

    def test_inspect_mat_type(self):
        builder = self._make_builder()
        builder._inspect_type_for_guards(wp.mat22)
        self.assertIn("WP_NO_MAT", builder.required_guards)

    def test_inspect_quat_type(self):
        builder = self._make_builder()
        builder._inspect_type_for_guards(wp.quat)
        self.assertIn("WP_NO_QUAT", builder.required_guards)

    def test_inspect_transform_type(self):
        builder = self._make_builder()
        builder._inspect_type_for_guards(wp.transformf)
        self.assertIn("WP_NO_QUAT", builder.required_guards)

    def test_inspect_float16_type(self):
        builder = self._make_builder()
        builder._inspect_type_for_guards(wp.float16)
        self.assertIn("WP_NO_FLOAT16_OPS", builder.required_guards)

    def test_inspect_float64_type(self):
        builder = self._make_builder()
        builder._inspect_type_for_guards(wp.float64)
        self.assertIn("WP_NO_FLOAT64_OPS", builder.required_guards)

    def test_inspect_array_of_vec(self):
        """Arrays should be unwrapped to inspect the dtype."""
        builder = self._make_builder()
        builder._inspect_type_for_guards(wp.array(dtype=wp.vec3))
        self.assertIn("WP_NO_VEC", builder.required_guards)

    def test_inspect_scalar_adds_nothing(self):
        builder = self._make_builder()
        builder._inspect_type_for_guards(wp.float32)
        self.assertEqual(builder.required_guards, set())

    # ------------------------------------------------------------------
    # require_guard(None) is no-op
    # ------------------------------------------------------------------

    def test_require_guard_none_is_noop(self):
        """require_guard(None) must not add anything."""
        builder = self._make_builder()
        builder.require_guard(None)
        self.assertEqual(builder.required_guards, set())


if __name__ == "__main__":
    unittest.main()
