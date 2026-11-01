# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run each stage of the NVSHMEM halo-exchange guide on two PEs and check it against one GPU.

The stages are in ``aux_nvshmem_halo_stages.py``, which the documentation includes. Each test runs that script once
through ``mpirun`` with the stage selected by ``WP_TEST_HALO_STAGE``. The delayed tests set ``WP_TEST_HALO_DELAY`` so
that PE 1 spins before each of its pushes, which makes PE 0 wait for rows that arrive late. The guide starts from
``example_jacobi_nvshmem_device.py`` and ends with ``example_jacobi_nvshmem_overlap.py``, which must agree, also when
``aux_nvshmem_overlap_delayed.py`` runs the second with the pushes of PE 1 delayed in the same way.
"""

import os
import re
import unittest

from warp.tests.distributed.test_nvshmem import run_distributed_module, run_distributed_script

# The arguments of the examples, which aux_nvshmem_overlap_delayed.py also passes.
EXAMPLE_ARGS = ["--iterations", "40", "--size", "64"]


class TestNVSHMEMHaloExchange(unittest.TestCase):
    """Tests for the stages of the NVSHMEM halo-exchange guide."""

    def _run_stage(self, stage, delay=False):
        """Run one stage on two PEs and check that both PEs match the one-GPU reference."""
        env = dict(os.environ, WP_TEST_HALO_STAGE=str(stage))
        if delay:
            env["WP_TEST_HALO_DELAY"] = "1"
        output = run_distributed_script(self, "aux_nvshmem_halo_stages.py", env=env, timeout=300)
        for pe in range(2):
            self.assertIn(f"PE {pe}: stage {stage}: PASSED", output)

    def test_stage_3_element_puts_host_barrier(self):
        """Verify the stencil that puts each boundary element, with a host barrier after each step."""
        self._run_stage(3)

    def test_stage_4_device_barrier(self):
        """Verify the same stencil with a one-thread barrier kernel in place of the host barrier."""
        self._run_stage(4)

    def test_stage_5_signals(self):
        """Verify row pushes that set a signal, which the next step waits for."""
        self._run_stage(5)

    def test_stage_6_two_streams(self):
        """Verify the push on a second stream, overlapping the interior rows."""
        self._run_stage(6)

    def test_stage_7_residual(self):
        """Verify the two-stream loop with a residual that every PE sums from exchanged partial sums."""
        self._run_stage(7)

    def test_stage_5_signals_delayed(self):
        """Verify the signal stage when PE 1 pushes late."""
        self._run_stage(5, delay=True)

    def test_stage_6_two_streams_delayed(self):
        """Verify the two-stream stage when PE 1 pushes late."""
        self._run_stage(6, delay=True)

    def test_stage_7_residual_delayed(self):
        """Verify the residual stage when PE 1 pushes late."""
        self._run_stage(7, delay=True)

    def _final_norm(self, output):
        """Return the norm that a Jacobi example run with ``EXAMPLE_ARGS`` printed at the end."""
        match = re.search(r"l2 norm after 40 iterations: ([0-9.]+)", output)
        self.assertIsNotNone(match, output)
        return float(match.group(1))

    def _simple_example_norm(self):
        """Run the first Jacobi example of the guide on two PEs and return its final norm."""
        output = run_distributed_module(
            self, "warp.examples.distributed.example_jacobi_nvshmem_device", EXAMPLE_ARGS, timeout=300
        )
        return self._final_norm(output)

    def test_examples_agree(self):
        """Verify that the first and last Jacobi examples of the guide print the same residual."""
        overlap = run_distributed_module(
            self, "warp.examples.distributed.example_jacobi_nvshmem_overlap", EXAMPLE_ARGS, timeout=300
        )
        # Both examples run the same stencil in the same order; only the order of the float32 atomic sums differs.
        self.assertAlmostEqual(self._simple_example_norm(), self._final_norm(overlap), delta=1e-5)

    def test_examples_agree_delayed(self):
        """Verify that the last Jacobi example of the guide agrees with the first when PE 1 pushes late."""
        delayed = run_distributed_script(self, "aux_nvshmem_overlap_delayed.py", timeout=300)
        self.assertIn("PE 1: 40 pushes delayed", delayed)
        self.assertAlmostEqual(self._simple_example_norm(), self._final_norm(delayed), delta=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
