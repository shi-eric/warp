# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check affine state, input gradient, and the nonlinear contraindication."""

import argparse
from collections.abc import Sequence

import numpy as np

import warp as wp
from warp.examples.optimizations.autodiff.gradient_safe_intermediate_lifetime.after import (
    affine_adjoint,
)
from warp.examples.optimizations.autodiff.gradient_safe_intermediate_lifetime.benchmark import (
    build_case,
)
from warp.examples.optimizations.harness import check_correctness


@wp.kernel
def sin_step(x: wp.array[float], y: wp.array[float]):
    i = wp.tid()
    y[i] = wp.sin(x[i])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--steps", required=True, type=int)
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


def _nonlinear_counterexample(device: str) -> bool:
    input_values = np.asarray([-0.75, -0.25, 0.25, 0.75], dtype=np.float32)
    size = len(input_values)
    input_array = wp.array(input_values, dtype=wp.float32, device=device, requires_grad=True)
    output_array = wp.empty(size, dtype=wp.float32, device=device, requires_grad=True)
    output_array.grad.fill_(1.0)
    tape = wp.Tape()
    with tape:
        wp.launch(sin_step, dim=size, inputs=[input_array], outputs=[output_array], device=device)
    tape.backward()

    constant_gradient = wp.empty(size, dtype=wp.float32, device=device)
    adjacent = wp.ones(size, dtype=wp.float32, device=device)
    wp.launch(
        affine_adjoint,
        dim=size,
        inputs=[adjacent, constant_gradient, 1.0],
        device=device,
    )
    wp.synchronize_device(device)
    actual = input_array.grad.numpy()
    incorrect = constant_gradient.numpy()
    expected = np.cos(input_values)
    np.testing.assert_allclose(actual, expected, atol=2.0e-6, rtol=2.0e-5)
    try:
        np.testing.assert_allclose(incorrect, expected, atol=2.0e-6, rtol=2.0e-5)
    except AssertionError:
        return True
    return False


def main(argv: Sequence[str] | None = None) -> int:
    """Run the card's deterministic affine and contraindication checks."""

    arguments = _build_parser().parse_args(argv)
    workload = {
        "derivative_depends_on_state": False,
        "size": arguments.size,
        "steps": arguments.steps,
        "seed": arguments.seed,
    }
    case = build_case(arguments.device, workload)
    _assert_repeatable_trial(case.baseline)
    _assert_repeatable_trial(case.candidate)
    result = check_correctness(case)
    nonlinear_rejected = _nonlinear_counterexample(arguments.device)
    print(f"size={arguments.size} steps={arguments.steps}: {'PASS' if result.passed else 'FAIL'}")
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
    print(f"nonlinear_counterexample: {'PASS' if nonlinear_rejected else 'FAIL'}")
    return 0 if result.passed and nonlinear_rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
