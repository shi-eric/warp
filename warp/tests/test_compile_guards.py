# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import unittest

import warp as wp
from warp._src.codegen import (
    VALID_NATIVE_HEADERS,
    Adjoint,
    compute_compile_guards,
)
from warp._src.context import ModuleBuilder, add_builtin, builtin_functions


class TestCompileGuards(unittest.TestCase):
    def test_add_builtin_validates_native_header(self):
        """Invalid native_header raises ValueError."""
        with self.assertRaises(ValueError):
            add_builtin(
                "_test_bad_guard",
                input_types={},
                value_type=int,
                native_header="fake",
            )

    def test_all_builtins_have_native_header(self):
        """Every registered builtin has a valid native_header."""
        for key, func in builtin_functions.items():
            if not hasattr(func, "overloads"):
                continue  # skip user-defined functions added via register_api_function
            for overload in func.overloads:
                self.assertIn(
                    overload.native_header,
                    VALID_NATIVE_HEADERS,
                    f"Builtin {key!r} has invalid native_header={overload.native_header!r}",
                )

    def test_guards_have_matching_cpp_ifndef(self):
        """Every header in VALID_NATIVE_HEADERS has a #ifndef in the native headers."""
        native_dir = os.path.join(os.path.dirname(wp.__file__), "native")
        all_header_content = ""
        for path in glob.glob(os.path.join(native_dir, "*.h")):
            with open(path) as f:
                all_header_content += f.read()

        for header in VALID_NATIVE_HEADERS:
            if header is None:
                continue
            self.assertIn(
                f"#ifndef WP_NO_{header.upper()}",
                all_header_content,
                f"Header {header!r} has no matching #ifndef WP_NO_{header.upper()} in native headers",
            )

    def test_func_return_type_inspected_for_headers(self):
        """build_function must inspect return types, not just arg types."""

        # Track which types _inspect_type_for_headers is called with.
        inspected_types = []
        original_inspect = ModuleBuilder._inspect_type_for_headers

        def tracking_inspect(self, t):
            inspected_types.append(t)
            original_inspect(self, t)

        ModuleBuilder._inspect_type_for_headers = tracking_inspect
        try:
            builder = ModuleBuilder.__new__(ModuleBuilder)
            builder.functions = {}
            builder.structs = {}
            builder.deferred_functions = []
            builder.required_headers = set()
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

            # Verify _inspect_type_for_headers was called with the vec3 return
            # type.  The builtin resolution also adds "vec", but the
            # return type path is defense-in-depth for cases where builtins
            # can't provide the header (e.g. types from external modules).
            inspected_generic_strs = [getattr(t, "_wp_generic_type_str_", None) for t in inspected_types]
            self.assertIn(
                "vec_t",
                inspected_generic_strs,
                "vec3 return type was not passed to _inspect_type_for_headers — "
                "build_function must inspect function return types",
            )
        finally:
            ModuleBuilder._inspect_type_for_headers = original_inspect

    # ------------------------------------------------------------------
    # compute_compile_guards() — dependency resolution
    # ------------------------------------------------------------------

    def test_compute_guards_empty_required(self):
        """No features required: all guards emitted."""
        result = compute_compile_guards(set())
        all_headers = VALID_NATIVE_HEADERS - {None}
        for h in all_headers:
            self.assertIn(f"#define WP_NO_{h.upper()}", result)

    def test_compute_guards_all_required(self):
        """All features required: no guards emitted."""
        all_headers = VALID_NATIVE_HEADERS - {None}
        result = compute_compile_guards(set(all_headers))
        self.assertEqual(result, "")

    def test_compute_guards_single_feature(self):
        """Requiring "mesh" prevents WP_NO_MESH from being emitted."""
        result = compute_compile_guards({"mesh"})
        self.assertNotIn("WP_NO_MESH", result)

    # ------------------------------------------------------------------
    # add_var() — guard tracking during codegen
    # ------------------------------------------------------------------

    def test_add_var_tracks_vec_guard(self):
        """Creating a variable with a vec type registers "vec"."""
        builder = self._make_builder()
        adj = Adjoint.__new__(Adjoint)
        adj.builder = builder
        adj.variables = []
        adj.blocks = [type("Block", (), {"vars": []})()]
        adj.lineno = 0

        adj.add_var(type=wp.vec3)
        self.assertIn("vec", builder.required_headers)

    def test_add_var_tracks_mat_guard(self):
        """Creating a variable with a mat type registers "mat"."""
        builder = self._make_builder()
        adj = Adjoint.__new__(Adjoint)
        adj.builder = builder
        adj.variables = []
        adj.blocks = [type("Block", (), {"vars": []})()]
        adj.lineno = 0

        adj.add_var(type=wp.mat22)
        self.assertIn("mat", builder.required_headers)

    def test_add_var_scalar_adds_no_guard(self):
        """Creating a variable with a scalar type adds no headers."""
        builder = self._make_builder()
        adj = Adjoint.__new__(Adjoint)
        adj.builder = builder
        adj.variables = []
        adj.blocks = [type("Block", (), {"vars": []})()]
        adj.lineno = 0

        adj.add_var(type=wp.float32)
        self.assertEqual(builder.required_headers, set())

    # ------------------------------------------------------------------
    # _inspect_type_for_headers() — type inspection
    # ------------------------------------------------------------------

    def _make_builder(self):
        """Create a minimal ModuleBuilder for testing type inspection."""
        builder = ModuleBuilder.__new__(ModuleBuilder)
        builder.required_headers = set()
        return builder

    def test_inspect_vec_type(self):
        builder = self._make_builder()
        builder._inspect_type_for_headers(wp.vec3)
        self.assertIn("vec", builder.required_headers)

    def test_inspect_mat_type(self):
        builder = self._make_builder()
        builder._inspect_type_for_headers(wp.mat22)
        self.assertIn("mat", builder.required_headers)

    def test_inspect_quat_type(self):
        builder = self._make_builder()
        builder._inspect_type_for_headers(wp.quat)
        self.assertIn("quat", builder.required_headers)

    def test_inspect_transform_type(self):
        builder = self._make_builder()
        builder._inspect_type_for_headers(wp.transformf)
        self.assertIn("quat", builder.required_headers)

    def test_inspect_float16_type(self):
        builder = self._make_builder()
        builder._inspect_type_for_headers(wp.float16)
        self.assertIn("float16_ops", builder.required_headers)

    def test_inspect_float64_type(self):
        builder = self._make_builder()
        builder._inspect_type_for_headers(wp.float64)
        self.assertIn("float64_ops", builder.required_headers)

    def test_inspect_array_of_vec(self):
        """Arrays should be unwrapped to inspect the dtype."""
        builder = self._make_builder()
        builder._inspect_type_for_headers(wp.array(dtype=wp.vec3))
        self.assertIn("vec", builder.required_headers)

    def test_inspect_scalar_adds_nothing(self):
        builder = self._make_builder()
        builder._inspect_type_for_headers(wp.float32)
        self.assertEqual(builder.required_headers, set())

    # ------------------------------------------------------------------
    # require_header(None) is no-op
    # ------------------------------------------------------------------

    def test_require_header_none_is_noop(self):
        """require_header(None) must not add anything."""
        builder = self._make_builder()
        builder.require_header(None)
        self.assertEqual(builder.required_headers, set())


if __name__ == "__main__":
    unittest.main()
