# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import json
import os
import tempfile
import unittest

import warp as wp
import warp._src.codegen
from warp.tests.distributed.test_nvshmem import _is_nvshmem_device_library_available, run_distributed_script


class TestNVSHMEMAutodiff(unittest.TestCase):
    """Tests for differentiating kernels that call the NVSHMEM built-ins."""

    def test_adjoint_rules(self):
        """Verify that the adjoints of the push built-ins pull the destination gradients from the remote PE."""
        output = run_distributed_script(self, "aux_nvshmem_adjoint_rules.py", timeout=300)
        for check in ("p element gradients", "put bulk gradients", "put_signal gradients", "no grad attached"):
            self.assertIn(f"PE 0: {check}: PASSED", output)

    def test_single_pe_behaviors(self):
        """Verify the NVSHMEM barriers and diagnostics of adjoint launches, on every launch path."""
        output = run_distributed_script(self, "aux_nvshmem_adjoint_single_pe.py", n_pes=1, timeout=300)
        for check in (
            "records survive a cache hit",
            "capture before warm-up raises",
            "flagged adjoint launch adds one barrier",
            "unflagged adjoint launch adds none",
            "forward launch adds none",
            "zero-size adjoint launch adds one",
            "record_cmd",
            "config off adds none",
            "deterministic path adds one barrier",
            "tape adds trailing barrier",
            "graph capture after warm-up",
            "pull warning once",
            "fused warning once",
            "symmetric check probe",
            "non-symmetric grad raises",
            "non-symmetric adj_inputs raises",
            "local arrays pass",
            "validation skipped without remote access",
        ):
            self.assertIn(f"PE 0: {check}: PASSED", output)

    def test_schedule(self):
        """Verify that the gradients of a two-PE Jacobi run match a one-GPU run on every pass.

        The passes repeat the run, synchronize the forward steps with signals, and replay a captured backward pass.
        """
        output = run_distributed_script(self, "aux_nvshmem_adjoint_schedule.py", timeout=600)
        for check in (
            "gradient matches reference",
            "second iteration matches reference",
            "pipelined signals match reference",
            "captured backward matches reference",
        ):
            for pe in range(2):
                self.assertIn(f"PE {pe}: {check}: PASSED", output)

    def test_empty_pe(self):
        """Verify that a PE with no threads in a flagged adjoint launch still takes part in its barrier."""
        output = run_distributed_script(self, "aux_nvshmem_adjoint_empty_pe.py", timeout=300)
        self.assertIn("PE 0: empty PE participates: PASSED", output)

    def test_codegen_records(self):
        """Verify that module metadata records the NVSHMEM traffic of each kernel's adjoint."""
        if not wp.is_nvshmem_enabled():
            self.skipTest("NVSHMEM not enabled in this build.")
        if not _is_nvshmem_device_library_available():
            self.skipTest("NVSHMEM device library not embedded in this build.")

        # Importing the kernels registers an NVSHMEM module with Warp, which every later load of all
        # modules (such as the one at the start of a graph capture) would then try to load.
        from warp.tests.distributed import aux_nvshmem_record_kernels as record_kernels  # noqa: PLC0415

        # kernel: (remote_access, pull_builtins, fused_shape)
        expected = {
            record_kernels.push_kernel: (True, [], False),
            record_kernels.via_func_kernel: (True, [], False),
            record_kernels.pull_kernel: (True, ["nvshmem_g"], False),
            record_kernels.get_only_kernel: (False, ["nvshmem_get"], False),
            record_kernels.fused_kernel: (True, [], True),
            record_kernels.pipelined_kernel: (True, [], False),
            record_kernels.local_kernel: (False, [], False),
        }

        # The second compile rebuilds the same kernel adjoints in this process, and must not inherit the
        # records of the first build (the pipelined kernel would then look like the fused shape).
        for compile_index in range(2):
            with tempfile.TemporaryDirectory(prefix="wp_nvshmem_records_") as module_dir:
                wp.compile_aot_module(record_kernels, device="cuda:0", module_dir=module_dir)
                meta_paths = glob.glob(os.path.join(module_dir, "*.meta"))
                self.assertEqual(len(meta_paths), 1, meta_paths)
                with open(meta_paths[0]) as meta_file:
                    meta = json.load(meta_file)

            for kernel, (remote_access, pull_builtins, fused_shape) in expected.items():
                with self.subTest(compile_index=compile_index, kernel=kernel.key):
                    key = warp._src.codegen.cuda_kernel_backward_name(kernel) + "_nvshmem"
                    self.assertEqual(
                        meta[key],
                        {"remote_access": remote_access, "pull_builtins": pull_builtins, "fused_shape": fused_shape},
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
