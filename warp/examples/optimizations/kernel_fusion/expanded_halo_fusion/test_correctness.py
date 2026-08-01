# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check a separable stencil against its expanded-halo product form."""

import argparse
from collections.abc import Sequence

import numpy as np

from warp.examples.optimizations.harness import check_correctness
from warp.examples.optimizations.kernel_fusion.expanded_halo_fusion.benchmark import (
    build_correctness_case,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--shape",
        action="append",
        required=True,
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
    )
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--radius", required=True, type=int)
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
    """Run deterministic CUDA correctness checks for requested shapes."""

    arguments = _build_parser().parse_args(argv)
    passed = True
    for height, width in arguments.shape:
        workload = {
            "height": height,
            "iterations": arguments.iterations,
            "radius": arguments.radius,
            "seed": arguments.seed,
            "width": width,
        }
        case = build_correctness_case(arguments.device, workload)
        _assert_repeatable_trial(case.baseline)
        _assert_repeatable_trial(case.candidate)
        result = check_correctness(case)
        print(
            f"shape={height}x{width} iterations={arguments.iterations} "
            f"radius={arguments.radius}: {'PASS' if result.passed else 'FAIL'}"
        )
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
        passed = passed and result.passed
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
