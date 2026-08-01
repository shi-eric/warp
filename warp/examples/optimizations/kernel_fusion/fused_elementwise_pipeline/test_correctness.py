# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check numerical equivalence for the fused elementwise pipeline card."""

import argparse
from collections.abc import Sequence

from warp.examples.optimizations.harness import check_correctness
from warp.examples.optimizations.kernel_fusion.fused_elementwise_pipeline.benchmark import build_case


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--size", action="append", required=True, type=int)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--seed", default=20260729, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the card's deterministic correctness check."""

    arguments = _build_parser().parse_args(argv)
    passed = True
    for size in arguments.size:
        workload = {
            "size": size,
            "iterations": arguments.iterations,
            "seed": arguments.seed,
        }
        result = check_correctness(build_case(arguments.device, workload))
        print(f"size={size}: {'PASS' if result.passed else 'FAIL'}")
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
