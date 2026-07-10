# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import warp as wp
import warp._src.build as warp_build
import warp._src.context as warp_context
from warp.tests.unittest_utils import *

# Auxiliary scripts for multi-PE tests (same directory)
_DISTRIBUTED_DIR = os.path.dirname(__file__)


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
            fake_core = SimpleNamespace(wp_nvshmem_cumodule_init=mock.Mock(return_value=-1))
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

    def test_nvshmem_kernel_compiles(self):
        """A kernel using NVSHMEM builtins should compile without error."""
        if not wp.is_nvshmem_enabled():
            self.skipTest("NVSHMEM not enabled in this build.")

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
