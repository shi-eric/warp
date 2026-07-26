# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import threading
import unittest
from pathlib import Path
from unittest import mock

import warp as wp
import warp._src.context as context
from warp._src.codegen import CompileFamily, emit_compile_family_macros, scan_source_for_families
from warp._src.context import (
    ModuleBuilder,
    ModuleHasher,
    add_builtin,
    builtin_functions,
    get_compile_family_schema_hash,
)

_MISSING = object()


@wp.kernel
def scalar_kernel(values: wp.array(dtype=float)):
    values[0] = values[0] + 1.0


class TestCompileGuards(unittest.TestCase):
    def _assert_dictionary_mutation_waits_for_schema_snapshot(self, register):
        key = "_test_schema_dictionary_race"
        traversal_started = threading.Event()
        release_traversal = threading.Event()
        traversal_active = threading.Event()
        mutated_during_traversal = threading.Event()
        thread_errors = []

        class BlockingRegistry(dict):
            def items(self):
                iterator = iter(super().items())
                first = next(iterator)
                traversal_active.set()
                traversal_started.set()
                release_traversal.wait()
                try:
                    yield first
                    yield from iterator
                finally:
                    traversal_active.clear()

            def __setitem__(self, item_key, value):
                was_traversing = traversal_active.is_set()
                super().__setitem__(item_key, value)
                if was_traversing:
                    mutated_during_traversal.set()
                    release_traversal.set()

        class CoordinatedRLock:
            def __init__(self):
                self.lock = threading.RLock()
                self.acquire_count = 0
                self.count_lock = threading.Lock()

            def __enter__(self):
                with self.count_lock:
                    self.acquire_count += 1
                    if self.acquire_count == 2:
                        release_traversal.set()
                self.lock.acquire()
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self.lock.release()

        def compute_schema_hash():
            try:
                get_compile_family_schema_hash()
            except BaseException as error:
                thread_errors.append(error)

        def mutate_registry():
            try:
                register(key)
            except BaseException as error:
                thread_errors.append(error)

        original_registry = context.builtin_functions
        original_schema_hash = context._compile_family_schema_hash
        context.builtin_functions = BlockingRegistry(original_registry)
        context._compile_family_schema_hash = None
        try:
            with mock.patch.object(context, "_compile_family_schema_lock", CoordinatedRLock()):
                hash_thread = threading.Thread(target=compute_schema_hash)
                hash_thread.start()
                traversal_started.wait()

                mutation_thread = threading.Thread(target=mutate_registry)
                mutation_thread.start()

                hash_thread.join()
                mutation_thread.join()

            self.assertEqual(thread_errors, [])
            self.assertFalse(
                mutated_during_traversal.is_set(),
                "Builtin registry changed while the schema snapshot held its lock",
            )
        finally:
            context.builtin_functions = original_registry
            context._compile_family_schema_hash = original_schema_hash

    def _registration_state(self, key):
        return (
            builtin_functions.get(key, _MISSING),
            getattr(wp, key, _MISSING),
            context._compile_family_schema_hash,
        )

    def _restore_registration_state(self, key, state):
        registry_value, warp_value, schema_hash = state
        with context._compile_family_schema_lock:
            if registry_value is _MISSING:
                builtin_functions.pop(key, None)
            else:
                builtin_functions[key] = registry_value

            if warp_value is _MISSING:
                if hasattr(wp, key):
                    delattr(wp, key)
            else:
                setattr(wp, key, warp_value)

            context._compile_family_schema_hash = schema_hash

    def test_module_hash_includes_family_schema(self):
        module = scalar_kernel.module
        with mock.patch(
            "warp._src.context.get_compile_family_schema_hash",
            return_value=b"a" * 32,
        ):
            hash_a = ModuleHasher(
                module._get_live_kernels(),
                module.resolve_options(wp.config),
            ).get_hash()
        with mock.patch(
            "warp._src.context.get_compile_family_schema_hash",
            return_value=b"b" * 32,
        ):
            hash_b = ModuleHasher(
                module._get_live_kernels(),
                module.resolve_options(wp.config),
            ).get_hash()
        self.assertNotEqual(hash_a, hash_b)

    def test_registering_builtin_invalidates_cached_module_hash(self):
        key = "_test_schema_family"
        state_before = self._registration_state(key)
        try:
            module = scalar_kernel.module
            before = get_compile_family_schema_hash()
            self.assertEqual(before, get_compile_family_schema_hash())
            module_before = module.get_module_hash()
            add_builtin(
                key,
                input_types={},
                value_type=int,
                compile_family=CompileFamily.VECTOR,
                hidden=True,
            )
            after = get_compile_family_schema_hash()
            module_after = module.get_module_hash()
            self.assertNotEqual(before, after)
            self.assertNotEqual(module_before, module_after)
        finally:
            self._restore_registration_state(key, state_before)

        self.assertEqual(self._registration_state(key), state_before)

    def test_failed_builtin_registration_preserves_schema(self):
        key = "_test_schema_namespace_collision"

        def occupied_namespace():
            pass

        before = get_compile_family_schema_hash()
        setattr(wp, key, occupied_namespace)
        try:
            with self.assertRaises(RuntimeError):
                add_builtin(
                    key,
                    input_types={},
                    value_type=int,
                    compile_family=CompileFamily.VECTOR,
                    hidden=True,
                )

            self.assertNotIn(key, builtin_functions)
            self.assertEqual(before, get_compile_family_schema_hash())
        finally:
            builtin_functions.pop(key, None)
            delattr(wp, key)
            context._compile_family_schema_hash = None

        self.assertEqual(before, get_compile_family_schema_hash())

    def test_concurrent_registration_cannot_publish_stale_schema(self):
        key = "_test_schema_concurrent_registration"
        snapshot_ready = threading.Event()
        release_snapshot = threading.Event()
        thread_errors = []

        class CoordinatedRLock:
            def __init__(self):
                self.lock = threading.RLock()
                self.acquire_count = 0
                self.count_lock = threading.Lock()

            def __enter__(self):
                with self.count_lock:
                    self.acquire_count += 1
                    acquire_count = self.acquire_count
                if acquire_count == 2:
                    release_snapshot.set()
                self.lock.acquire()
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self.lock.release()

        coordinated_lock = CoordinatedRLock()
        original_dumps = context.json.dumps

        def blocking_dumps(*args, **kwargs):
            snapshot_ready.set()
            release_snapshot.wait()
            return original_dumps(*args, **kwargs)

        def compute_schema_hash():
            try:
                get_compile_family_schema_hash()
            except BaseException as error:
                thread_errors.append(error)

        def register_builtin():
            try:
                add_builtin(
                    key,
                    input_types={},
                    value_type=int,
                    compile_family=CompileFamily.VECTOR,
                    hidden=True,
                    export=False,
                )
            except BaseException as error:
                thread_errors.append(error)
            finally:
                release_snapshot.set()

        context._compile_family_schema_hash = None
        try:
            with (
                mock.patch.object(context, "_compile_family_schema_lock", coordinated_lock, create=True),
                mock.patch.object(context.json, "dumps", side_effect=blocking_dumps),
            ):
                hash_thread = threading.Thread(target=compute_schema_hash)
                hash_thread.start()
                snapshot_ready.wait()

                registration_thread = threading.Thread(target=register_builtin)
                registration_thread.start()

                hash_thread.join()
                registration_thread.join()

            self.assertEqual(thread_errors, [])
            published = get_compile_family_schema_hash()
            context._compile_family_schema_hash = None
            rebuilt = get_compile_family_schema_hash()
            self.assertEqual(published, rebuilt)
        finally:
            builtin_functions.pop(key, None)
            context._compile_family_schema_hash = None

    def test_add_builtin_waits_for_schema_dictionary_snapshot(self):
        """Catch dictionary mutation while compile-family schema records are traversed."""

        def register(key):
            add_builtin(
                key,
                input_types={},
                value_type=int,
                compile_family=CompileFamily.VECTOR,
                hidden=True,
                export=False,
            )

        self._assert_dictionary_mutation_waits_for_schema_snapshot(register)

    def test_register_api_function_waits_for_schema_dictionary_snapshot(self):
        """Catch API registration changing dictionary size during a schema snapshot."""

        def register(key):
            function = type("ApiFunction", (), {})()
            function.key = key
            context.register_api_function(function)

        self._assert_dictionary_mutation_waits_for_schema_snapshot(register)

    def test_add_builtin_waits_for_schema_overload_snapshot(self):
        """Catch builtin overload mutation while schema overload records are traversed."""
        key = "_test_schema_overload_race"
        traversal_started = threading.Event()
        release_traversal = threading.Event()
        traversal_active = threading.Event()
        mutated_during_traversal = threading.Event()
        thread_errors = []

        class BlockingOverloads(list):
            def __iter__(self):
                iterator = super().__iter__()
                first = next(iterator)
                traversal_active.set()
                traversal_started.set()
                release_traversal.wait()
                try:
                    yield first
                    yield from iterator
                finally:
                    traversal_active.clear()

            def append(self, value):
                was_traversing = traversal_active.is_set()
                super().append(value)
                if was_traversing:
                    mutated_during_traversal.set()
                    release_traversal.set()

        class CoordinatedRLock:
            def __init__(self):
                self.lock = threading.RLock()
                self.acquire_count = 0
                self.count_lock = threading.Lock()

            def __enter__(self):
                with self.count_lock:
                    self.acquire_count += 1
                    if self.acquire_count == 2:
                        release_traversal.set()
                self.lock.acquire()
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self.lock.release()

        def compute_schema_hash():
            try:
                get_compile_family_schema_hash()
            except BaseException as error:
                thread_errors.append(error)

        def add_overload():
            try:
                add_builtin(
                    key,
                    input_types={"value": int},
                    value_type=int,
                    compile_family=CompileFamily.VECTOR,
                    hidden=True,
                    export=False,
                )
            except BaseException as error:
                thread_errors.append(error)

        original_registry = context.builtin_functions
        original_schema_hash = context._compile_family_schema_hash
        context.builtin_functions = {}
        try:
            head = add_builtin(
                key,
                input_types={},
                value_type=int,
                compile_family=CompileFamily.VECTOR,
                hidden=True,
                export=False,
            )
            head.overloads = BlockingOverloads(head.overloads)
            context._compile_family_schema_hash = None

            with mock.patch.object(context, "_compile_family_schema_lock", CoordinatedRLock()):
                hash_thread = threading.Thread(target=compute_schema_hash)
                hash_thread.start()
                traversal_started.wait()

                mutation_thread = threading.Thread(target=add_overload)
                mutation_thread.start()

                hash_thread.join()
                mutation_thread.join()

            self.assertEqual(thread_errors, [])
            self.assertFalse(
                mutated_during_traversal.is_set(),
                "Builtin overloads changed while the schema snapshot held its lock",
            )
        finally:
            context.builtin_functions = original_registry
            context._compile_family_schema_hash = original_schema_hash

    def test_add_builtin_requires_explicit_family(self):
        """Catch builtin registrations that silently omit compile-family metadata."""
        with self.assertRaises(ValueError):
            add_builtin("_test_missing_family", input_types={}, value_type=int)

    def test_add_builtin_accepts_explicit_none(self):
        """Catch rejection of an explicit unconditional builtin classification."""
        key = "_test_unconditional_family"
        state_before = self._registration_state(key)
        try:
            func = add_builtin(
                key,
                input_types={},
                value_type=int,
                compile_family=None,
            )
            self.assertIsNone(func.compile_family)
        finally:
            self._restore_registration_state(key, state_before)

        self.assertEqual(self._registration_state(key), state_before)

    def test_namespace_collision_without_name_reports_registration_error(self):
        """Catch namespace-collision formatting that assumes ``__name__`` exists."""
        key = "_test_schema_nameless_collision"
        occupied_namespace = object()
        before = get_compile_family_schema_hash()
        state_before = self._registration_state(key)
        setattr(wp, key, occupied_namespace)
        try:
            with self.assertRaisesRegex(RuntimeError, "would overwrite existing object"):
                add_builtin(
                    key,
                    input_types={},
                    value_type=int,
                    compile_family=CompileFamily.VECTOR,
                    hidden=True,
                )
            self.assertNotIn(key, builtin_functions)
            self.assertEqual(before, get_compile_family_schema_hash())
        finally:
            self._restore_registration_state(key, state_before)

        self.assertEqual(self._registration_state(key), state_before)

    def test_clang_diagnostics_do_not_escape_parse_options(self):
        """Catch transfer of a diagnostic client backed by function-local options."""
        clang_source_path = Path(__file__).resolve().parents[1] / "native" / "clang" / "clang.cpp"
        source = clang_source_path.read_text()
        function_start = source.index("static std::unique_ptr<clang::CompilerInstance> create_compiler")
        function_end = source.index("\nstatic bool generate_pch", function_start)
        create_compiler_source = source[function_start:function_end]
        self.assertNotIn("diagnostic_engine->getClient()", create_compiler_source)

    def test_llvm22_diagnostics_initialize_virtual_file_system(self):
        """Catch LLVM 22 diagnostics dereferencing an unset virtual file system."""
        clang_source_path = Path(__file__).resolve().parents[1] / "native" / "clang" / "clang.cpp"
        source = clang_source_path.read_text()
        function_start = source.index("static std::unique_ptr<clang::CompilerInstance> create_compiler")
        llvm22_start = source.index("#if LLVM_VERSION_MAJOR >= 22", function_start)
        llvm22_end = source.index("#elif LLVM_VERSION_MAJOR == 21", llvm22_start)
        llvm22_branch = source[llvm22_start:llvm22_end]
        create_vfs = llvm22_branch.find("compiler_instance->createVirtualFileSystem()")
        create_diagnostics = llvm22_branch.find("compiler_instance->createDiagnostics()")
        self.assertGreaterEqual(create_vfs, 0)
        self.assertGreater(create_diagnostics, create_vfs)

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
        warp_root = Path(__file__).resolve().parents[1]
        production_roots = (
            warp_root / "_src",
            warp_root / "native",
        )
        source_suffixes = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py"}
        paths = sorted(
            path
            for production_root in production_roots
            for path in production_root.rglob("*")
            if path.is_file() and path.suffix in source_suffixes
        )
        for path in paths:
            source = path.read_text()
            for suffix in ("RAND", "NOISE", "FLOAT64_OPS"):
                removed_name = "WP_NO_" + suffix
                self.assertNotIn(removed_name, source, f"{removed_name} found in {path.relative_to(warp_root)}")

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
