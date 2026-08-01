# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check numerical equivalence for fixed iteration workspace reuse."""

import argparse
from collections.abc import Sequence

import numpy as np

from warp.examples.optimizations.allocation_reuse.reused_iteration_workspace.benchmark import (
    build_case,
)
from warp.examples.optimizations.harness import check_correctness


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--seed", default=20260730, type=int)
    return parser


def _assert_repeatable_trial(variant) -> None:
    snapshots = []
    for _ in range(2):
        variant.prepare_trial()
        variant.run()
        variant.synchronize()
        snapshots.append({name: np.asarray(value).copy() for name, value in variant.outputs().items()})

    for name in snapshots[0]:
        np.testing.assert_array_equal(snapshots[0][name], snapshots[1][name])


def main(argv: Sequence[str] | None = None) -> int:
    """Run the card's deterministic workspace-reuse correctness check."""

    arguments = _build_parser().parse_args(argv)
    workload = {
        "size": arguments.size,
        "iterations": arguments.iterations,
        "seed": arguments.seed,
    }
    case = build_case(arguments.device, workload)
    _assert_repeatable_trial(case.baseline)
    _assert_repeatable_trial(case.candidate)
    result = check_correctness(case)
    print(f"size={arguments.size} iterations={arguments.iterations}: {'PASS' if result.passed else 'FAIL'}")
    print("OUTPUT  PASSED  FINITE  MAX_ABS  MAX_REL  MAX_NORMALIZED")
    for name in sorted(result.outputs):
        output = result.outputs[name]
        max_abs = "null" if output.max_abs is None else f"{output.max_abs:.12g}"
        max_rel = "null" if output.max_rel is None else f"{output.max_rel:.12g}"
        max_normalized = "null" if output.max_normalized is None else f"{output.max_normalized:.12g}"
        print(
            f"{name}  {'yes' if output.passed else 'no'}  "
            f"{'yes' if output.finite else 'no'}  {max_abs}  {max_rel}  "
            f"{max_normalized}"
        )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
