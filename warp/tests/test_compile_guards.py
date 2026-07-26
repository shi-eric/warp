# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import unittest
from pathlib import Path

import warp as wp
from warp._src.codegen import CompileFamily, emit_compile_family_macros, scan_source_for_families
from warp._src.context import ModuleBuilder, add_builtin, builtin_functions


class TestCompileGuards(unittest.TestCase):
    def test_add_builtin_requires_explicit_family(self):
        """Catch builtin registrations that silently omit compile-family metadata."""
        with self.assertRaises(ValueError):
            add_builtin("_test_missing_family", input_types={}, value_type=int)

    def test_add_builtin_accepts_explicit_none(self):
        """Catch rejection of an explicit unconditional builtin classification."""
        func = add_builtin(
            "_test_unconditional_family",
            input_types={},
            value_type=int,
            compile_family=None,
        )
        self.assertIsNone(func.compile_family)

    def test_add_builtin_validates_compile_family(self):
        """Catch non-enum metadata crossing the builtin registration boundary."""
        with self.assertRaises(ValueError):
            add_builtin(
                "_test_bad_family",
                input_types={},
                value_type=int,
                compile_family="WP_NO_FAKE",
            )

    def test_all_builtins_have_compile_family(self):
        """Catch a registered builtin with missing or invalid family metadata."""
        for key, func in builtin_functions.items():
            if not hasattr(func, "overloads"):
                continue
            for overload in func.overloads:
                self.assertTrue(
                    overload.compile_family is None or isinstance(overload.compile_family, CompileFamily),
                    f"Builtin {key!r} has invalid compile_family={overload.compile_family!r}",
                )

    def test_compile_family_macros_are_unique(self):
        """Catch two positive families that disable the same native feature macro."""
        macros = [family.macro for family in CompileFamily]
        self.assertEqual(len(macros), len(set(macros)))

    def test_final_family_taxonomy(self):
        """Catch obsolete, missing, or accidentally split positive compile families."""
        self.assertEqual(
            {family.name for family in CompileFamily},
            {
                "MESH",
                "BVH",
                "INTERSECT",
                "HASHGRID",
                "VOLUME",
                "TEXTURE",
                "VECTOR",
                "MATRIX",
                "QUATERNION",
                "SVD",
                "TILE",
                "FLOAT16",
                "STOCHASTIC",
            },
        )

    def test_families_have_matching_cpp_ifndef(self):
        """Catch a family macro with no matching native-header feature block."""
        native_dir = os.path.join(os.path.dirname(wp.__file__), "native")
        all_header_content = ""
        for path in glob.glob(os.path.join(native_dir, "*.h")):
            with open(path) as file:
                all_header_content += file.read()

        for family in CompileFamily:
            self.assertIn(
                f"#ifndef {family.macro}",
                all_header_content,
                f"Family {family.name!r} has no matching #ifndef for {family.macro}",
            )

    def test_removed_macro_names_are_absent(self):
        """Catch reintroduction of removed Random, Noise, or Float64 guard macros."""
        root = Path(__file__).resolve().parents[1]
        paths = (
            root / "_src" / "codegen.py",
            root / "_src" / "context.py",
            root / "_src" / "builtins.py",
            root / "native" / "builtin.h",
            root / "native" / "noise.h",
        )
        source = "\n".join(path.read_text() for path in paths)
        for suffix in ("RAND", "NOISE", "FLOAT64_OPS"):
            removed_name = "WP_NO_" + suffix
            self.assertNotIn(removed_name, source)

    def test_func_return_type_inspected_for_families(self):
        """Catch function return types bypassing compile-family inspection."""
        inspected_types = []
        original_inspect = ModuleBuilder._inspect_type_for_families

        def tracking_inspect(self, value_type):
            inspected_types.append(value_type)
            original_inspect(self, value_type)

        ModuleBuilder._inspect_type_for_families = tracking_inspect
        try:
            builder = ModuleBuilder.__new__(ModuleBuilder)
            builder.functions = {}
            builder.structs = {}
            builder.deferred_functions = []
            builder.required_families = set()
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

            builder.build_function(to_vec)

            inspected_generic_strs = [
                getattr(value_type, "_wp_generic_type_str_", None) for value_type in inspected_types
            ]
            self.assertIn(
                "vec_t",
                inspected_generic_strs,
                "vec3 return type was not passed to _inspect_type_for_families",
            )
        finally:
            ModuleBuilder._inspect_type_for_families = original_inspect

    def test_emit_macros_empty_required(self):
        """Catch omission of a negative macro for an unused compile family."""
        result = emit_compile_family_macros(set())
        expected = "".join(f"#define {family.macro}\n" for family in CompileFamily)
        self.assertEqual(result, expected)

    def test_emit_macros_all_required(self):
        """Catch disabling a native family that the module directly requires."""
        result = emit_compile_family_macros(set(CompileFamily))
        self.assertEqual(result, "")

    def test_emit_macros_single_family(self):
        """Catch emission of the negative macro for one required family."""
        result = emit_compile_family_macros({CompileFamily.MESH})
        self.assertNotIn("WP_NO_MESH", result)

    def test_emit_macros_rejects_unknown_family(self):
        """Catch unknown metadata being silently accepted during macro emission."""
        with self.assertRaises(ValueError):
            emit_compile_family_macros({object()})

    def test_scan_source_detects_vec(self):
        """Catch generated vector source bypassing the safety-net family scan."""
        required = set()
        scan_source_for_families("wp::vec3 v = {};", required)
        self.assertIn(CompileFamily.VECTOR, required)

    def test_scan_source_detects_mat(self):
        """Catch generated matrix source bypassing the safety-net family scan."""
        required = set()
        scan_source_for_families("mat_t<3, 2, float32> m;", required)
        self.assertIn(CompileFamily.MATRIX, required)

    def test_scan_source_detects_quat(self):
        """Catch generated quaternion source bypassing the safety-net family scan."""
        required = set()
        scan_source_for_families("quat_t<float32> q;", required)
        self.assertIn(CompileFamily.QUATERNION, required)

    def test_scan_source_no_match(self):
        """Catch the source safety net adding families to scalar-only code."""
        required = set()
        scan_source_for_families("float x = 1.0f;", required)
        self.assertEqual(required, set())

    def test_scan_source_idempotent(self):
        """Catch repeated source scanning corrupting existing family metadata."""
        required = {CompileFamily.VECTOR}
        scan_source_for_families("vec3 v;", required)
        self.assertEqual(required, {CompileFamily.VECTOR})

    def _make_builder(self):
        builder = ModuleBuilder.__new__(ModuleBuilder)
        builder.required_families = set()
        return builder

    def test_inspect_vec_type(self):
        """Catch vector arguments failing to require the Vector family."""
        builder = self._make_builder()
        builder._inspect_type_for_families(wp.vec3)
        self.assertIn(CompileFamily.VECTOR, builder.required_families)

    def test_inspect_mat_type(self):
        """Catch matrix arguments failing to require the Matrix family."""
        builder = self._make_builder()
        builder._inspect_type_for_families(wp.mat22)
        self.assertIn(CompileFamily.MATRIX, builder.required_families)

    def test_inspect_quat_type(self):
        """Catch quaternion arguments failing to require the Quaternion family."""
        builder = self._make_builder()
        builder._inspect_type_for_families(wp.quat)
        self.assertIn(CompileFamily.QUATERNION, builder.required_families)

    def test_inspect_transform_type(self):
        """Catch transform arguments failing to require the Quaternion family."""
        builder = self._make_builder()
        builder._inspect_type_for_families(wp.transformf)
        self.assertIn(CompileFamily.QUATERNION, builder.required_families)

    def test_inspect_float16_type(self):
        """Catch Float16 arguments failing to require the Float16 family."""
        builder = self._make_builder()
        builder._inspect_type_for_families(wp.float16)
        self.assertIn(CompileFamily.FLOAT16, builder.required_families)

    def test_inspect_float64_type(self):
        """Catch Float64 arguments incorrectly requiring an optional family."""
        builder = self._make_builder()
        builder._inspect_type_for_families(wp.float64)
        self.assertEqual(builder.required_families, set())

    def test_inspect_array_of_vec(self):
        """Catch array element types bypassing compile-family inspection."""
        builder = self._make_builder()
        builder._inspect_type_for_families(wp.array(dtype=wp.vec3))
        self.assertIn(CompileFamily.VECTOR, builder.required_families)

    def test_inspect_scalar_adds_nothing(self):
        """Catch unconditional scalar types gaining an optional family."""
        builder = self._make_builder()
        builder._inspect_type_for_families(wp.float32)
        self.assertEqual(builder.required_families, set())

    def test_require_family_none_is_noop(self):
        """Catch explicit unconditional metadata entering the required family set."""
        builder = self._make_builder()
        builder.require_family(None)
        self.assertEqual(builder.required_families, set())


if __name__ == "__main__":
    unittest.main()
