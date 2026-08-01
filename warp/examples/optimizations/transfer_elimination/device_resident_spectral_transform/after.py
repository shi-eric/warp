# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate spectral filter that keeps every iteration on CUDA."""

from collections.abc import Callable

import numpy as np

import warp as wp
from warp.examples.optimizations.harness import Variant

wp.set_module_options({"enable_backward": False})

TRANSFORM_SIZE = 256
BLOCK_DIM = 64
NORMALIZATION = wp.vec2f(
    wp.float32(1.0 / TRANSFORM_SIZE),
    wp.float32(1.0 / TRANSFORM_SIZE),
)


@wp.func
def multiply_and_normalize(value: wp.vec2f, gain: wp.vec2f):
    real = value[0] * gain[0] - value[1] * gain[1]
    imaginary = value[0] * gain[1] + value[1] * gain[0]
    return wp.cw_mul(wp.vec2f(real, imaginary), NORMALIZATION)


@wp.kernel
def filter_rows(
    state: wp.array2d[wp.vec2f],
    gain: wp.array2d[wp.vec2f],
    output: wp.array2d[wp.vec2f],
):
    row = wp.tid()
    values = wp.tile_load(
        state,
        shape=(1, TRANSFORM_SIZE),
        offset=(row, 0),
    )
    weights = wp.tile_load(
        gain,
        shape=(1, TRANSFORM_SIZE),
        offset=(0, 0),
    )
    wp.tile_fft(values)
    values = wp.tile_map(multiply_and_normalize, values, weights)
    wp.tile_ifft(values)
    wp.tile_store(output, values, offset=(row, 0))


def build_variant(
    *,
    initial_pairs: np.ndarray,
    gain_pairs: np.ndarray,
    iterations: int,
    device: str,
    synchronize: Callable[[], None],
) -> Variant:
    """Build the tiled candidate with preallocated ping-pong storage."""

    batch = initial_pairs.shape[0]
    initial_state = wp.array(initial_pairs, dtype=wp.vec2f, device=device)
    gain = wp.array(gain_pairs, dtype=wp.vec2f, device=device)
    state_a = wp.empty_like(initial_state)
    state_b = wp.empty_like(initial_state)
    wp.copy(state_a, initial_state)
    final_state = state_a

    def prepare_trial() -> None:
        nonlocal final_state
        wp.copy(state_a, initial_state)
        final_state = state_a

    def run() -> None:
        nonlocal final_state
        source = state_a
        target = state_b
        for _ in range(iterations):
            wp.launch_tiled(
                filter_rows,
                dim=[batch],
                inputs=[source, gain],
                outputs=[target],
                block_dim=BLOCK_DIM,
                device=device,
            )
            source, target = target, source
        final_state = source

    def outputs() -> dict[str, np.ndarray]:
        host_pairs = final_state.numpy()
        return {
            "imaginary": host_pairs[..., 1],
            "real": host_pairs[..., 0],
        }

    return Variant(
        label="device-resident tiled candidate",
        prepare_trial=prepare_trial,
        run=run,
        synchronize=synchronize,
        outputs=outputs,
    )
