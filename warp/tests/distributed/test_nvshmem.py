# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import sys
import unittest

import warp as wp
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
    """Tests for NVSHMEM codegen (requires NVSHMEM build)."""

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
