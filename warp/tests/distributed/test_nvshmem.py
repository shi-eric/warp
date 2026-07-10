# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import build_lib as warp_build_lib
import warp as wp
import warp._src.build as warp_build
import warp._src.context as warp_context
from warp.tests.unittest_utils import *

# Auxiliary scripts for multi-PE tests (same directory)
_DISTRIBUTED_DIR = os.path.dirname(__file__)


def _is_nvshmem_device_library_available():
    size = ctypes.c_size_t()
    kind = ctypes.c_int()
    return bool(warp_context.runtime.core.wp_nvshmem_get_device_library(ctypes.byref(size), ctypes.byref(kind)))


class TestNvshmemBuild(unittest.TestCase):
    """Tests for selecting and embedding NVSHMEM device libraries."""

    def test_packaged_fatbin_generates_embedding(self):
        """The packaged fatbin should be staged for exact embedding in ``warp.so``."""
        with tempfile.TemporaryDirectory(prefix="wp_nvshmem_install_") as temp_dir:
            nvshmem_path = os.path.join(temp_dir, "nvshmem")
            nvshmem_lib_path = os.path.join(nvshmem_path, "lib")
            os.makedirs(nvshmem_lib_path)
            version_header_path = os.path.join(nvshmem_path, "include", "non_abi")
            os.makedirs(version_header_path)
            with open(os.path.join(version_header_path, "nvshmem_version.h"), "w", encoding="utf-8") as header_file:
                header_file.write('#define NVSHMEM_BUILD_VARS "CUDA_HOME=/toolkit/r13.2/build/cuda"\n')
            fatbin_path = os.path.join(nvshmem_lib_path, warp_build_lib.NVSHMEM_DEVICE_FATBIN_FILENAME)
            with open(fatbin_path, "wb") as fatbin_file:
                fatbin_file.write(b"fatbin")

            selected = warp_build_lib.find_nvshmem_device_library(nvshmem_path)
            self.assertEqual(selected, ("fatbin", fatbin_path))
            self.assertEqual(warp_build_lib.get_nvshmem_build_cuda_version(nvshmem_path), (13, 2))

            assembly_path, staged_path = warp_build_lib.generate_nvshmem_device_library_assembly(temp_dir, *selected)
            with open(staged_path, "rb") as staged_file:
                self.assertEqual(staged_file.read(), b"fatbin")
            with open(assembly_path, encoding="utf-8") as assembly_file:
                assembly = assembly_file.read()
            self.assertIn(f'.incbin "{staged_path}"', assembly)
            self.assertIn("wp_nvshmem_device_library_start", assembly)
            self.assertIn("wp_nvshmem_device_library_end", assembly)

            build_path = os.path.join(temp_dir, "warp-package")
            bin_path = os.path.join(build_path, "bin")
            os.makedirs(bin_path)
            stale_ltoir_path = os.path.join(bin_path, warp_build_lib.NVSHMEM_DEVICE_LTOIR_FILENAME)
            with open(stale_ltoir_path, "wb") as ltoir_file:
                ltoir_file.write(b"ltoir")
            warp_build_lib.remove_legacy_nvshmem_device_libraries(build_path)
            self.assertFalse(os.path.exists(stale_ltoir_path))


class TestNvshmemArray(unittest.TestCase):
    """Tests for symmetric array allocation (requires NVSHMEM)."""

    def test_symmetric_requires_cuda(self):
        """Symmetric arrays must be on a CUDA device."""
        with self.assertRaises(RuntimeError):
            wp.zeros(10, dtype=wp.float32, device="cpu", symmetric=True)

    def test_symmetric_requires_nvshmem_build(self):
        """Symmetric arrays require NVSHMEM-enabled build."""
        if wp.is_nvshmem_enabled():
            self.skipTest("NVSHMEM is enabled, cannot test disabled path.")
        with self.assertRaises(RuntimeError):
            wp.zeros(10, dtype=wp.float32, device="cuda:0", symmetric=True)


class TestNvshmemCodegen(unittest.TestCase):
    """Tests for NVSHMEM code generation and cached module metadata."""

    def test_cached_module_restores_nvshmem_usage(self):
        """Cached CUDA modules should restore NVSHMEM initialization metadata."""

        @wp.kernel(module="unique")
        def nvshmem_kernel(out_pe: wp.array[wp.int32]):
            out_pe[0] = wp.nvshmem_my_pe()

        source_module = nvshmem_kernel.module
        options = source_module.resolve_options(wp.config) | {"output_arch": 80}
        _source, _extension, meta, _ltoirs, _fatbins, uses_nvshmem = source_module._run_codegen(options, is_cpu=False)

        self.assertTrue(uses_nvshmem)
        self.assertTrue(meta[warp_context._MODULE_USES_NVSHMEM_META_KEY])

        with tempfile.TemporaryDirectory(prefix="wp_nvshmem_cache_") as cache_dir:
            binary_path = os.path.join(cache_dir, "module.cubin")
            meta_path = os.path.join(cache_dir, "module.meta")
            with open(binary_path, "wb"):
                pass
            with open(meta_path, "w", encoding="utf-8") as meta_file:
                json.dump(meta, meta_file)

            cached_module = warp_context.Module("cached_nvshmem_test")
            cached_module.resolved_options[cached_module.options["block_dim"]] = {}
            cached_module.get_module_hash = mock.Mock(return_value=b"\0" * 32)
            cached_module.get_module_identifier = mock.Mock(return_value="cached_nvshmem_test")
            cached_module._refresh_deterministic_metadata_for_cache_hit = mock.Mock()

            fake_device = SimpleNamespace(context=1, is_cpu=False, is_cuda=True, alias="cuda:test")
            nvshmem_version = (3 << 16) | (7 << 8) | 1
            fake_core = SimpleNamespace(
                wp_nvshmem_cumodule_init=mock.Mock(return_value=-1),
                wp_nvshmem_get_build_version=mock.Mock(return_value=nvshmem_version),
                wp_nvshmem_get_loaded_version=mock.Mock(return_value=nvshmem_version),
            )
            fake_runtime = SimpleNamespace(core=fake_core, get_device=mock.Mock(return_value=fake_device), tape=None)

            with (
                mock.patch.object(warp_context, "runtime", fake_runtime),
                mock.patch.object(warp_build, "load_cuda", return_value=123),
                mock.patch.object(warp_context, "log_warning") as log_warning,
                mock.patch.object(warp_context, "ModuleExec", return_value=mock.sentinel.module_exec) as module_exec,
            ):
                loaded_exec = cached_module.load(
                    fake_device,
                    binary_path=binary_path,
                    output_arch=80,
                    meta_path=meta_path,
                )

        self.assertIs(loaded_exec, mock.sentinel.module_exec)
        self.assertTrue(cached_module.uses_nvshmem)
        fake_core.wp_nvshmem_cumodule_init.assert_called_once_with(123)
        log_warning.assert_called_once_with(
            "nvshmemx_cumodule_init failed (error -1). "
            "NVSHMEM device functions will not work until NVSHMEM is initialized "
            "and the module is reloaded."
        )
        loaded_meta = module_exec.call_args.args[3]
        self.assertNotIn(warp_context._MODULE_USES_NVSHMEM_META_KEY, loaded_meta)

    def test_nvshmem_version_mismatch_is_rejected(self):
        """NVSHMEM host and device library versions must match exactly."""
        fake_core = SimpleNamespace(
            wp_nvshmem_get_build_version=mock.Mock(return_value=(3 << 16) | (7 << 8) | 1),
            wp_nvshmem_get_loaded_version=mock.Mock(return_value=(3 << 16) | (4 << 8) | 5),
        )
        fake_runtime = SimpleNamespace(core=fake_core)

        with (
            mock.patch.object(warp_context, "runtime", fake_runtime),
            self.assertRaisesRegex(
                RuntimeError,
                r"Warp was built for NVSHMEM 3\.7\.1, but the loaded host library is NVSHMEM 3\.4\.5",
            ),
        ):
            warp_context._validate_nvshmem_version()

    def test_nvshmem_kernel_compiles(self):
        """A kernel using NVSHMEM builtins should compile without error."""
        if not wp.is_nvshmem_enabled():
            self.skipTest("NVSHMEM not enabled in this build.")
        if not _is_nvshmem_device_library_available():
            self.skipTest("NVSHMEM device library not embedded in this build.")

        @wp.kernel
        def nvshmem_kernel(out_pe: wp.array[wp.int32], out_npes: wp.array[wp.int32]):
            pe = wp.nvshmem_my_pe()
            npes = wp.nvshmem_n_pes()
            out_pe[0] = pe
            out_npes[0] = npes

        wp.load_module(device="cuda:0")


class TestNvshmemMultiPE(unittest.TestCase):
    """Multi-PE tests launched via mpirun subprocess."""

    def _run_distributed(self, script_name, n_pes=2):
        """Run a script from warp/tests/distributed/ via mpirun."""
        mpirun = shutil.which("mpirun")
        if mpirun is None:
            self.skipTest("mpirun not found in PATH")

        if not wp.is_nvshmem_enabled():
            self.skipTest("NVSHMEM not enabled in this build.")
        if not _is_nvshmem_device_library_available():
            self.skipTest("NVSHMEM device library not embedded in this build.")

        script = os.path.join(_DISTRIBUTED_DIR, script_name)
        result = subprocess.run(
            [mpirun, "-np", str(n_pes), sys.executable, script],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            self.fail(
                f"mpirun failed with return code {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
        return result.stdout

    def test_pe_query(self):
        """Verify my_pe and n_pes return correct values."""
        output = self._run_distributed("aux_nvshmem_pe_query.py")
        self.assertIn("PASSED", output)

    def test_float_p(self):
        """Verify nvshmem_float_p writes to remote PE."""
        output = self._run_distributed("aux_nvshmem_float_p.py")
        self.assertIn("PASSED", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
