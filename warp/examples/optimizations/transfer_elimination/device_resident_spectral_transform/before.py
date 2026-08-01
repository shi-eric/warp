# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseline spectral filter with a repeated CUDA-to-host round trip."""

from collections.abc import Callable

import numpy as np

import warp as wp
from warp.examples.optimizations.harness import Variant


def build_variant(
    *,
    initial_pairs: np.ndarray,
    gain_pairs: np.ndarray,
    iterations: int,
    device: str,
    synchronize: Callable[[], None],
) -> Variant:
    """Build the host-transform baseline from matching complex inputs."""

    initial_state = wp.array(initial_pairs, dtype=wp.vec2f, device=device)
    host_gain = gain_pairs[..., 0] + 1j * gain_pairs[..., 1]
    state = initial_state

    def prepare_trial() -> None:
        nonlocal state
        state = initial_state

    def run() -> None:
        nonlocal state
        for _ in range(iterations):
            host_pairs = state.numpy()
            host_complex = host_pairs[..., 0] + 1j * host_pairs[..., 1]
            spectrum = np.fft.fft(host_complex, axis=1)
            filtered = np.fft.ifft(spectrum * host_gain, axis=1)
            next_pairs = np.stack((filtered.real, filtered.imag), axis=-1).astype(np.float32)
            state = wp.array(next_pairs, dtype=wp.vec2f, device=device)

    def outputs() -> dict[str, np.ndarray]:
        host_pairs = state.numpy()
        return {
            "imaginary": host_pairs[..., 1],
            "real": host_pairs[..., 0],
        }

    return Variant(
        label="host-round-trip baseline",
        prepare_trial=prepare_trial,
        run=run,
        synchronize=synchronize,
        outputs=outputs,
    )
