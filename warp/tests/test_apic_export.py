# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for APIC graph export and targeted CUDA artifacts."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import warp as wp
import warp._src.context as wp_context
from warp._src.cuda_build import source_path
from warp.tests.unittest_utils import (
    add_function_test,
    get_cuda_test_devices,
    get_test_devices_with_cuda_graph_module_load,
)


@wp.kernel
def scale_kernel(input: wp.array[float], output: wp.array[float], s: float):
    i = wp.tid()
    output[i] = input[i] * s


@wp.kernel
def write_value_kernel(x: wp.array[float], value: float):
    x[wp.tid()] = value


@wp.kernel
def decrement_counter_kernel(c: wp.array[wp.int32]):
    c[0] = c[0] - 1


_APIC_TILE_BLOCK_DIM = 64


@wp.kernel(enable_backward=False, module="unique")
def apic_tile_matmul_kernel(
    a: wp.array2d[wp.float16],
    b: wp.array2d[wp.float16],
    out: wp.array2d[wp.float32],
):
    a_tile = wp.tile_load(a, shape=(8, 8))
    b_tile = wp.tile_load(b, shape=(8, 8))
    out_tile = wp.tile_zeros(shape=(8, 8), dtype=wp.float32)
    wp.tile_matmul(a_tile, b_tile, out_tile)
    wp.tile_store(out, out_tile)


def _make_const_writer(value):
    """Build a kernel that writes a compile-time constant.

    Two kernels from this factory share ``kernel.key``
    (``_make_const_writer__locals__kernel``) but compile to distinct modules
    via ``module="unique"``.
    """
    constant = value

    @wp.kernel(module="unique")
    def kernel(out: wp.array[wp.int32]):
        out[0] = wp.static(constant)

    return kernel


class TestApicExport(unittest.TestCase):
    def test_capture_save_bindings_are_per_save(self):
        source = wp.ones(4, dtype=float, device="cpu")
        output = wp.zeros_like(source)
        with wp.ScopedCapture(device="cpu", apic=True, force_module_load=False) as capture:
            wp.launch(scale_kernel, dim=4, inputs=[source, output, 3.0], device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            first_path = os.path.join(tmpdir, "first")
            wp.capture_save(capture.graph, first_path, inputs={"old_input": source}, outputs={"old_output": output})
            first = wp.capture_load(first_path, device="cpu")
            self.assertEqual(set(first.params), {"old_input", "old_output"})

            # A directory at the destination makes the native file write fail
            # after save metadata has been prepared.
            failed_path = os.path.join(tmpdir, "failed.wrp")
            os.mkdir(failed_path)
            with self.assertRaisesRegex(RuntimeError, "Failed to save APIC graph"):
                wp.capture_save(capture.graph, failed_path, outputs={"failed_output": output})

            final_path = os.path.join(tmpdir, "final")
            wp.capture_save(capture.graph, final_path, inputs={"new_input": source}, outputs={"new_output": output})
            loaded = wp.capture_load(final_path, device="cpu")
            self.assertEqual(set(loaded.params), {"new_input", "new_output"})
            loaded.set_param("new_input", wp.full(4, 2.0, dtype=float, device="cpu"))
            wp.capture_launch(loaded)
            result = wp.zeros_like(output)
            loaded.get_param("new_output", result)
            np.testing.assert_allclose(result.numpy(), np.full(4, 6.0, dtype=np.float32))

    def test_capture_save_target_options_reject_cpu_graph(self):
        with wp.ScopedCapture(device="cpu", apic=True, force_module_load=False) as capture:
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cpu_target")
            for options in ({"target_arch": 80}, {"target_arch": 80, "use_ptx": True}):
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, "CUDA"):
                    wp.capture_save(capture.graph, path, **options)
            with self.assertRaisesRegex(ValueError, "use_ptx=True.*target_arch"):
                wp.capture_save(capture.graph, path, use_ptx=True)
            self.assertFalse(os.path.exists(path + ".wrp"))
            self.assertFalse(os.path.exists(path + "_modules"))


def _load_capture_output(path, name, template, device):
    loaded = wp.capture_load(path, device=device)
    wp.capture_launch(loaded)
    result = wp.zeros_like(template)
    loaded.get_param(name, result)
    return result.numpy()


def _load_apic_binary_with_name(kernel, device, directory, basename):
    """Reload a real compiled module from a user-selected binary filename."""
    kernel.module.unload()
    module_exec = kernel.module.load(device)
    directory.mkdir()
    # Preserve the architecture ending when exercising collisions alone.
    extension = Path(module_exec.binary_path).suffix
    ending = f".sm{module_exec.compile_arch}{module_exec.arch_suffix}" if device.is_cuda else ""
    binary = directory / f"{basename}{ending if basename == 'shared' else ''}{extension}"
    shutil.copy2(module_exec.binary_path, binary)
    meta = directory / "module.meta"
    meta.write_text(json.dumps(module_exec.meta))
    compile_arch = module_exec.compile_arch
    kernel.module.unload()
    kernel.module.load(device, binary_path=binary, meta_path=meta, output_arch=compile_arch)


def test_capture_save_explicit_binary_names(test, device, use_ptx=False):
    """Keep distinct explicit binaries even with duplicate or custom basenames."""

    @wp.kernel(module="unique", enable_backward=False)
    def first(out: wp.array[int]):
        out[0] = 11

    @wp.kernel(module="unique", enable_backward=False)
    def second(out: wp.array[int]):
        out[0] = 22

    block_dim = 1 if device.is_cpu else 256
    for kernel in (first, second):
        wp.set_module_options(
            {"block_dim": block_dim, "cuda_output": "ptx" if use_ptx else "cubin"}, module=kernel.module
        )
    outputs = [wp.zeros(1, dtype=int, device=device) for _ in range(2)]
    for basename in ("shared", "custom"):
        with test.subTest(basename=basename), tempfile.TemporaryDirectory() as tmpdir:
            for index, kernel in enumerate((first, second)):
                _load_apic_binary_with_name(kernel, device, Path(tmpdir) / str(index), basename)
            with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
                wp.launch(first, dim=1, outputs=[outputs[0]], block_dim=block_dim, device=device)
                wp.launch(second, dim=1, outputs=[outputs[1]], block_dim=block_dim, device=device)
            path = os.path.join(tmpdir, "export")
            wp.capture_save(capture.graph, path, outputs={"first": outputs[0], "second": outputs[1]})
            test.assertEqual(len(list(Path(path + "_modules").iterdir())), 2)
            loaded = wp.capture_load(path, device=device)
            wp.capture_launch(loaded)
            for name, out, expected in zip(("first", "second"), outputs, (11, 22), strict=True):
                loaded.get_param(name, out)
                np.testing.assert_array_equal(out.numpy(), [expected])
            # CPU graph loads reuse module handles by filename; release this
            # graph before loading another bundle containing the same modules.
            del loaded


def test_capture_save_relative_aot_path(test, device):
    """Retain the AOT binary location across a working-directory change."""

    @wp.kernel(module="unique", enable_backward=False)
    def write(out: wp.array[int]):
        out[0] = 37

    block_dim = 1 if device.is_cpu else 256
    wp.set_module_options({"cpu_compiler_flags": "", "block_dim": block_dim}, module=write.module)
    out = wp.zeros(1, dtype=int, device=device)
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            wp.compile_aot_module(write.module, device=device, module_dir="aot")
            wp.load_aot_module(write.module, device=device, module_dir="aot")
            with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
                wp.launch(write, dim=1, outputs=[out], block_dim=block_dim, device=device)
        finally:
            os.chdir(original_cwd)
        path = os.path.join(tmpdir, "export")
        wp.capture_save(capture.graph, path, outputs={"out": out})
        np.testing.assert_array_equal(_load_capture_output(path, "out", out, device), [37])


def test_capture_save_target_formats_round_trip(test, device):
    """Export CUBIN and portable PTX artifacts and execute compatible targets."""
    out = wp.zeros(4, dtype=float, device=device)
    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(write_value_kernel, dim=4, inputs=[out, 9.0], device=device)

    cases = [(device.arch, False, ".cubin")]
    if 75 in wp.get_cuda_supported_archs() and device.arch >= 75:
        cases.append((75, True, ".ptx"))

    with tempfile.TemporaryDirectory() as tmpdir:
        for target_arch, use_ptx, extension in cases:
            with test.subTest(target_arch=target_arch, use_ptx=use_ptx):
                path = os.path.join(tmpdir, "targeted")
                wp.capture_save(
                    capture.graph,
                    path,
                    outputs={"out": out},
                    target_arch=target_arch,
                    use_ptx=use_ptx,
                )
                module_files = list(Path(path + "_modules").iterdir())
                test.assertTrue(module_files)
                test.assertTrue(all(module_file.suffix == extension for module_file in module_files))
                test.assertTrue(all(f".sm{target_arch}" in module_file.name for module_file in module_files))
                if use_ptx:
                    test.assertTrue(
                        all(f".target sm_{target_arch}" in module_file.read_text() for module_file in module_files)
                    )
                np.testing.assert_array_equal(
                    _load_capture_output(path, "out", out, device),
                    np.full(4, 9.0, dtype=np.float32),
                )

        other_arch = next((arch for arch in wp.get_cuda_supported_archs() if arch != device.arch), None)
        if other_arch is not None:
            path = os.path.join(tmpdir, "other_arch")
            wp.capture_save(capture.graph, path, target_arch=other_arch)
            module_files = list(Path(path + "_modules").iterdir())
            test.assertTrue(module_files)
            test.assertTrue(all(module_file.suffix == ".cubin" for module_file in module_files))
            test.assertTrue(all(f".sm{other_arch}" in module_file.name for module_file in module_files))


def test_capture_load_rejects_ptx_above_device_arch(test, device):
    target_arch = next((arch for arch in wp.get_cuda_supported_archs() if arch > device.arch), None)
    if target_arch is None:
        test.skipTest("The bundled compiler has no target newer than the test device")

    out = wp.zeros(1, dtype=float, device=device)
    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(write_value_kernel, dim=1, inputs=[out, 1.0], device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "too_new_ptx")
        wp.capture_save(capture.graph, path, target_arch=target_arch, use_ptx=True)
        with test.assertRaisesRegex(RuntimeError, "baseline PTX.*physical"):
            wp.capture_load(path, device=device)


def test_capture_save_target_options(test, device):
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        pass

    unsupported_arch = next(arch for arch in range(12345, 12400) if arch not in wp.get_cuda_supported_archs())
    cases = (
        ({"target_arch": True}, TypeError, "positive integer"),
        ({"target_arch": 0}, ValueError, "positive integer"),
        ({"target_arch": unsupported_arch}, ValueError, "supported"),
        ({"target_arch": device.arch, "use_ptx": 1}, TypeError, "Boolean"),
        ({"use_ptx": True}, ValueError, "target_arch"),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        for index, (options, error, message) in enumerate(cases):
            with test.subTest(options=options), test.assertRaisesRegex(error, message):
                wp.capture_save(capture.graph, os.path.join(tmpdir, str(index)), **options)
        test.assertEqual(list(Path(tmpdir).iterdir()), [])


def test_capture_save_rejects_mathdx(test, device):
    """Reject targeted export when a module contains MathDx link inputs."""
    if not wp_context.runtime.core.wp_is_mathdx_enabled():
        test.skipTest("Warp was built without MathDx")

    a = wp.ones((8, 8), dtype=wp.float16, device=device)
    b = wp.ones((8, 8), dtype=wp.float16, device=device)
    out = wp.zeros((8, 8), dtype=wp.float32, device=device)
    apic_tile_matmul_kernel.module.load(device, block_dim=_APIC_TILE_BLOCK_DIM)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch_tiled(
            apic_tile_matmul_kernel,
            dim=1,
            inputs=[a, b, out],
            block_dim=_APIC_TILE_BLOCK_DIM,
            device=device,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        wp.capture_save(capture.graph, os.path.join(tmpdir, "untargeted"), outputs={"out": out})
        with test.assertRaisesRegex(RuntimeError, "MathDx link inputs"):
            wp.capture_save(
                capture.graph,
                os.path.join(tmpdir, "targeted"),
                outputs={"out": out},
                target_arch=device.arch,
            )


def test_capture_save_preserves_captured_static_value(test, device):
    """Retarget the captured source without re-evaluating Python static inputs."""
    settings = {"offset": 1}

    @wp.kernel(module="unique", enable_backward=False)
    def captured_kernel(out: wp.array[int]):
        tile = wp.tile(1)
        out[0] = wp.static(len(tile) + settings["offset"])

    out = wp.zeros(1, dtype=int, device=device)
    captured_kernel.module.load(device, block_dim=64)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(captured_kernel, dim=1, outputs=[out], block_dim=64, device=device)
    settings["offset"] = 99

    with tempfile.TemporaryDirectory() as tmpdir:
        for use_ptx in (True, True, False):
            with test.subTest(use_ptx=use_ptx):
                path = os.path.join(tmpdir, "frozen")
                wp.capture_save(
                    capture.graph,
                    path,
                    outputs={"out": out},
                    target_arch=device.arch,
                    use_ptx=use_ptx,
                )
                np.testing.assert_array_equal(_load_capture_output(path, "out", out, device), [65])


def test_capture_save_forward_only_shared_memory(test, device):
    """Replay only the recorded direction even when its unused adjoint cannot run."""
    tile_size = int(device.max_shared_memory_per_block * 0.6) // 4
    block_dim = 64

    @wp.kernel(module="unique", enable_backward=True)
    def copy_tile(inp: wp.array[float], out: wp.array[float]):
        values = wp.tile_load(inp, shape=tile_size, storage="shared")
        wp.tile_store(out, values)

    inp = wp.ones(tile_size, dtype=float, device=device)
    out = wp.zeros(tile_size, dtype=float, device=device)
    module_exec = copy_tile.module.load(device, block_dim=block_dim)
    hooks = module_exec.get_kernel_hooks(copy_tile)
    test.assertLessEqual(hooks.forward_smem_bytes, device.max_shared_memory_per_block)
    test.assertGreater(hooks.backward_smem_bytes, device.max_shared_memory_per_block)

    def launch():
        wp.launch_tiled(copy_tile, dim=1, inputs=[inp], outputs=[out], block_dim=block_dim, device=device)

    condition = wp.ones(1, dtype=int, device=device)
    loop_condition = wp.ones(1, dtype=int, device=device)

    def loop_body():
        wp.capture_if(
            condition,
            on_true=launch if form == "nested_true" else None,
            on_false=launch if form == "nested_false" else None,
        )
        loop_condition.zero_()

    forms = ("direct", "nested_true", "nested_false") if wp.is_conditional_graph_supported() else ("direct",)
    with tempfile.TemporaryDirectory() as tmpdir:
        for form in forms:
            condition.fill_(int(form == "nested_true"))
            loop_condition.fill_(1)
            out.zero_()
            with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
                if form == "direct":
                    launch()
                else:
                    wp.capture_while(loop_condition, loop_body)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(out.numpy(), np.ones(tile_size))
            for target_options in ({}, {"target_arch": device.arch, "use_ptx": True}):
                with test.subTest(form=form, target_options=target_options):
                    path = os.path.join(tmpdir, "forward_only")
                    out.zero_()
                    loop_condition.fill_(1)
                    wp.capture_save(capture.graph, path, outputs={"out": out}, **target_options)
                    np.testing.assert_array_equal(_load_capture_output(path, "out", out, device), np.ones(tile_size))


def test_capture_save_uses_frozen_cuda_source(test, device):
    """Export the source retained by the captured executable, not mutable cache files."""
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(wp.config, "kernel_cache_dir", tmpdir):

        @wp.kernel(module="unique", enable_backward=False)
        def frozen_kernel(out: wp.array[int]):
            out[0] = 42

        out = wp.zeros(1, dtype=int, device=device)
        original_record = frozen_kernel.module.load(device).compile_record
        frozen_kernel.module.unload()
        module_exec = frozen_kernel.module.load(device)
        test.assertIs(module_exec.compile_record.source, original_record.source)
        with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
            wp.launch(frozen_kernel, dim=1, outputs=[out], device=device)

        record = module_exec.compile_record
        source = source_path(wp.config.kernel_cache_dir, record)
        manifest_path = Path(tmpdir) / "cuda" / "records" / f"{record.fingerprint()}.json"
        artifact_manifest = Path(module_exec.binary_path).parent / "artifact.json"
        test.assertNotIn("source", json.loads(manifest_path.read_text()))
        test.assertEqual(json.loads(artifact_manifest.read_text())["identity"]["record"], record.fingerprint())
        source.write_text("This is not valid CUDA source.\n")
        manifest_path.write_text("{}\n")

        path = os.path.join(tmpdir, "export")
        wp.capture_save(capture.graph, path, outputs={"out": out}, target_arch=device.arch, use_ptx=True)
        np.testing.assert_array_equal(_load_capture_output(path, "out", out, device), [42])
        test.assertEqual(source.read_bytes(), record.source)
        test.assertEqual(json.loads(manifest_path.read_text()), record.to_json())


def test_capture_save_rejects_changed_dependencies(test, device):
    """Reject changed external dependencies without breaking untargeted export."""
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(wp.config, "kernel_cache_dir", tmpdir):
        dependency = Path(tmpdir) / "dependency.h"
        dependency.write_text("// Original dependency\n")

        @wp.kernel(module="unique", enable_backward=False)
        def frozen_kernel(out: wp.array[int]):
            out[0] = 42

        wp.set_module_options(
            {"extra_build_options": wp.ModuleBuildOptions(extra_build_dependencies=[dependency])},
            module=frozen_kernel.module,
        )
        out = wp.zeros(1, dtype=int, device=device)
        frozen_kernel.module.load(device)
        with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
            wp.launch(frozen_kernel, dim=1, outputs=[out], device=device)
        path = os.path.join(tmpdir, "export")
        original = dependency.read_bytes()
        try:
            for missing in (False, True):
                with test.subTest(missing=missing):
                    if missing:
                        dependency.unlink()
                    else:
                        dependency.write_bytes(original + b"// Changed\n")
                    with test.assertRaisesRegex(RuntimeError, "dependency"):
                        wp.capture_save(capture.graph, path, target_arch=device.arch, use_ptx=True)
                    wp.capture_save(capture.graph, path, outputs={"out": out})
                    np.testing.assert_array_equal(_load_capture_output(path, "out", out, device), [42])
        finally:
            dependency.write_bytes(original)


def test_capture_save_targeted_module_variants(test, device):
    """Retain distinct block-dimension variants from one captured module."""

    @wp.kernel(module="unique", enable_backward=False)
    def block_dim_kernel(out: wp.array[int]):
        tile = wp.tile(1)
        out[0] = wp.static(len(tile))

    outputs = [wp.zeros(1, dtype=int, device=device) for _ in range(2)]
    block_dim_kernel.module.load(device, block_dim=64)
    block_dim_kernel.module.load(device, block_dim=256)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        for out, block_dim in zip(outputs, (64, 256), strict=True):
            wp.launch(block_dim_kernel, dim=1, outputs=[out], block_dim=block_dim, device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "variants")
        wp.capture_save(
            capture.graph,
            path,
            outputs={"out64": outputs[0], "out256": outputs[1]},
            target_arch=device.arch,
            use_ptx=True,
        )
        test.assertEqual(len(list(Path(path + "_modules").glob("*.ptx"))), 2)
        np.testing.assert_array_equal(_load_capture_output(path, "out64", outputs[0], device), [64])
        np.testing.assert_array_equal(_load_capture_output(path, "out256", outputs[1], device), [256])


def test_capture_save_targeted_distinct_modules_same_key(test, device):
    """Retain distinct captured modules whose kernels share a Python key."""
    kernels = (_make_const_writer(111), _make_const_writer(222))
    test.assertEqual(kernels[0].key, kernels[1].key)
    outputs = [wp.zeros(1, dtype=int, device=device) for _ in kernels]
    for kernel in kernels:
        kernel.module.load(device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        for kernel, out in zip(kernels, outputs, strict=True):
            wp.launch(kernel, dim=1, inputs=[out], device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "same_key")
        wp.capture_save(
            capture.graph,
            path,
            outputs={"first": outputs[0], "second": outputs[1]},
            target_arch=device.arch,
            use_ptx=True,
        )
        test.assertEqual(len(list(Path(path + "_modules").glob("*.ptx"))), 2)
        np.testing.assert_array_equal(_load_capture_output(path, "first", outputs[0], device), [111])
        np.testing.assert_array_equal(_load_capture_output(path, "second", outputs[1], device), [222])


def test_capture_save_rejects_unsupported_cluster_target(test, device):
    """Fail atomically when the requested PTX target cannot represent the graph."""
    if device.arch < 90:
        test.skipTest("A cluster-capable capture device is required")
    target_arch = max((arch for arch in wp.get_cuda_supported_archs() if arch < 90), default=None)
    if target_arch is None:
        test.skipTest("The bundled compiler has no pre-sm_90 PTX target")

    @wp.kernel(module="unique", enable_backward=False)
    def clustered_kernel(out: wp.array[int]):
        out[wp.tid()] = 1

    wp.set_module_options({"cluster_dim": 2}, module=clustered_kernel.module)
    out = wp.zeros(512, dtype=int, device=device)
    clustered_kernel.module.load(device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(clustered_kernel, dim=512, outputs=[out], device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cluster")
        wp.capture_save(capture.graph, path, outputs={"out": out})
        wrp_before = Path(path + ".wrp").read_bytes()
        modules_before = {p.name: p.read_bytes() for p in Path(path + "_modules").iterdir()}
        with test.assertRaisesRegex(RuntimeError, "cluster_dim=2"):
            wp.capture_save(capture.graph, path, target_arch=target_arch, use_ptx=True)
        test.assertEqual(Path(path + ".wrp").read_bytes(), wrp_before)
        test.assertEqual({p.name: p.read_bytes() for p in Path(path + "_modules").iterdir()}, modules_before)
        np.testing.assert_array_equal(_load_capture_output(path, "out", out, device), np.ones(512, dtype=np.int32))


def test_capture_save_targeted_conditionals(test, device):
    """Rebuild targeted PTX conditionals for the physical load device."""
    if not wp.is_conditional_graph_supported():
        test.skipTest("CUDA conditional graph nodes require Toolkit and driver 12.4+")
    if 75 not in wp.get_cuda_supported_archs() or device.arch < 75:
        test.skipTest("compute_75 PTX is unavailable on this toolkit/device")

    out = wp.zeros(4, dtype=float, device=device)
    condition = wp.array([1], dtype=wp.int32, device=device)
    counter = wp.array([3], dtype=wp.int32, device=device)

    def on_true():
        wp.launch(write_value_kernel, dim=4, inputs=[out, 11.0], device=device)

    def while_body():
        wp.launch(decrement_counter_kernel, dim=1, inputs=[counter], device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.capture_if(condition, on_true=on_true)
        wp.capture_while(counter, while_body=while_body)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "conditionals")
        wp.capture_save(
            capture.graph,
            path,
            outputs={"out": out, "counter": counter},
            target_arch=75,
            use_ptx=True,
        )
        loaded = wp.capture_load(path, device=device)
        wp.capture_launch(loaded)
        loaded_out = wp.zeros_like(out)
        loaded_counter = wp.zeros_like(counter)
        loaded.get_param("out", loaded_out)
        loaded.get_param("counter", loaded_counter)
        np.testing.assert_array_equal(loaded_out.numpy(), np.full(4, 11.0, dtype=np.float32))
        np.testing.assert_array_equal(loaded_counter.numpy(), [0])


devices_with_cuda_graph_module_load = get_test_devices_with_cuda_graph_module_load()

add_function_test(
    TestApicExport,
    "test_capture_save_explicit_binary_names",
    test_capture_save_explicit_binary_names,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApicExport,
    "test_capture_save_relative_aot_path",
    test_capture_save_relative_aot_path,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApicExport,
    "test_capture_save_explicit_ptx_names",
    test_capture_save_explicit_binary_names,
    devices=[d for d in devices_with_cuda_graph_module_load if d.is_cuda],
    use_ptx=True,
)
add_function_test(
    TestApicExport,
    "test_capture_save_target_formats_round_trip",
    test_capture_save_target_formats_round_trip,
    devices=[d for d in devices_with_cuda_graph_module_load if d.is_cuda],
)
add_function_test(
    TestApicExport,
    "test_capture_load_rejects_ptx_above_device_arch",
    test_capture_load_rejects_ptx_above_device_arch,
    devices=[d for d in devices_with_cuda_graph_module_load if d.is_cuda],
)
add_function_test(
    TestApicExport,
    "test_capture_save_target_options",
    test_capture_save_target_options,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestApicExport,
    "test_capture_save_rejects_mathdx",
    test_capture_save_rejects_mathdx,
    devices=[d for d in devices_with_cuda_graph_module_load if d.is_cuda],
)
add_function_test(
    TestApicExport,
    "test_capture_save_preserves_captured_static_value",
    test_capture_save_preserves_captured_static_value,
    devices=[d for d in devices_with_cuda_graph_module_load if d.is_cuda],
)
add_function_test(
    TestApicExport,
    "test_capture_save_targeted_module_variants",
    test_capture_save_targeted_module_variants,
    devices=[d for d in devices_with_cuda_graph_module_load if d.is_cuda],
)
for test_function in (
    test_capture_save_forward_only_shared_memory,
    test_capture_save_uses_frozen_cuda_source,
    test_capture_save_rejects_changed_dependencies,
):
    add_function_test(
        TestApicExport,
        test_function.__name__,
        test_function,
        devices=[d for d in devices_with_cuda_graph_module_load if d.is_cuda],
    )
add_function_test(
    TestApicExport,
    "test_capture_save_targeted_distinct_modules_same_key",
    test_capture_save_targeted_distinct_modules_same_key,
    devices=[d for d in devices_with_cuda_graph_module_load if d.is_cuda],
)
add_function_test(
    TestApicExport,
    "test_capture_save_rejects_unsupported_cluster_target",
    test_capture_save_rejects_unsupported_cluster_target,
    devices=[d for d in devices_with_cuda_graph_module_load if d.is_cuda],
)
add_function_test(
    TestApicExport,
    "test_capture_save_targeted_conditionals",
    test_capture_save_targeted_conditionals,
    devices=[d for d in devices_with_cuda_graph_module_load if d.is_cuda],
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
