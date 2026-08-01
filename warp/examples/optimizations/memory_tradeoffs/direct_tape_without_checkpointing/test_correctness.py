# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check segmented recomputation against one bounded direct Warp Tape."""

import argparse
from collections.abc import Sequence

import numpy as np

from warp.examples.optimizations.harness import check_correctness
from warp.examples.optimizations.memory_tradeoffs.direct_tape_without_checkpointing.benchmark import (
    build_case,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--segment-length", required=True, type=int)
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
    """Run the deterministic CUDA state-and-gradient correctness check."""

    arguments = _build_parser().parse_args(argv)
    workload = {
        "seed": arguments.seed,
        "segment_length": arguments.segment_length,
        "size": arguments.size,
        "steps": arguments.steps,
    }
    case = build_case(arguments.device, workload)
    _assert_repeatable_trial(case.baseline)
    _assert_repeatable_trial(case.candidate)
    result = check_correctness(case)
    print(
        f"size={arguments.size} steps={arguments.steps} "
        f"segment_length={arguments.segment_length}: "
        f"{'PASS' if result.passed else 'FAIL'}"
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
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
