# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check exchange equivalence, aliasing, ownership, and stream ordering."""

import argparse
from collections.abc import Sequence

import numpy as np

from warp.examples.optimizations.harness import check_correctness
from warp.examples.optimizations.interoperability.device_resident_torch_exchange.after import (
    DeviceResidentVariant,
)
from warp.examples.optimizations.interoperability.device_resident_torch_exchange.benchmark import (
    build_case,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--seed", default=20260730, type=int)
    parser.add_argument("--verify-ordering", action="store_true")
    return parser


def _reference_values(*, size: int, iterations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.uniform(-0.75, 0.75, size=size).astype(np.float32)
    for _ in range(iterations):
        values = (np.float32(0.5) * values + np.sin(values)).astype(np.float32)
    return values


def _assert_repeatable_trial(variant) -> dict[str, np.ndarray]:
    snapshots = []
    for _ in range(2):
        variant.prepare_trial()
        variant.run()
        variant.synchronize()
        snapshots.append({name: np.asarray(value).copy() for name, value in variant.outputs().items()})

    for name in snapshots[0]:
        np.testing.assert_array_equal(snapshots[0][name], snapshots[1][name])
    return snapshots[-1]


def _assert_ordering(candidate: DeviceResidentVariant) -> None:
    probe_value = np.float32(0.25)
    candidate.trial_state.run_ordering_probe(float(probe_value))
    candidate.synchronize()
    candidate.trial_state.assert_storage_alias()
    expected = np.float32(np.float32(0.5) * probe_value + np.sin(probe_value))
    np.testing.assert_allclose(
        candidate.trial_state.ordering_output(),
        expected,
        atol=2.0e-6,
        rtol=2.0e-5,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run deterministic correctness and optional ordering checks."""

    arguments = _build_parser().parse_args(argv)
    workload = {
        "iterations": arguments.iterations,
        "seed": arguments.seed,
        "size": arguments.size,
    }
    case = build_case(arguments.device, workload)
    if not isinstance(case.candidate, DeviceResidentVariant):
        raise AssertionError("candidate must expose retained device-resident state")

    baseline_snapshot = _assert_repeatable_trial(case.baseline)
    candidate_snapshot = _assert_repeatable_trial(case.candidate)
    expected = _reference_values(**workload)
    np.testing.assert_allclose(
        baseline_snapshot["values"],
        expected,
        atol=2.0e-6,
        rtol=2.0e-5,
    )
    np.testing.assert_allclose(
        candidate_snapshot["values"],
        expected,
        atol=2.0e-6,
        rtol=2.0e-5,
    )

    result = check_correctness(case)
    case.baseline.trial_state.assert_stream_contract()
    case.candidate.trial_state.assert_stream_contract()
    case.candidate.trial_state.assert_storage_alias()

    print(f"size={arguments.size} iterations={arguments.iterations}: {'PASS' if result.passed else 'FAIL'}")
    print("storage_alias=PASS")
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

    if arguments.verify_ordering:
        _assert_ordering(case.candidate)
        print("stream=non-default")
        print("producer_warp_consumer=PASS")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
