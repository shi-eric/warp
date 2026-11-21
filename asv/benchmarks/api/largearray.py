# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the cost of opting in to 64-bit indexing with ``wp.largearray``.

Each benchmark runs the same ``axpy`` kernel written two ways: with
``wp.array`` parameters and ``wp.tid()`` (``array``), and with ``wp.largearray``
parameters and ``wp.linear_tid()`` (``largearray``). Both variants run on the
same 2**24-element arrays, which fit the 32-bit descriptor. The comparison
records what a user pays for opting in, which is more than the 64-bit
descriptor and index alone: on the CPU, the ``array`` variant also gets the
linear-contiguous specialisation, which ``largearray`` parameters do not. There
is no pass/fail threshold.

Warp builds without ``wp.largearray`` skip the ``largearray`` variant.
"""

import functools

import warp as wp

from ..benchmarks_utils import LAUNCHES_PER_SAMPLE, CudaEventTimer, capture_launches, setup_once

N = 1 << 24
VARIANTS = ("array", "largearray")


@wp.kernel(enable_backward=False)
def axpy_array(x: wp.array[float], y: wp.array[float], a: float):
    i = wp.tid()
    y[i] = a * x[i] + y[i]


@functools.cache
def _axpy_largearray():
    # Defined on first use so this file still imports on Warp builds without wp.largearray.
    @wp.kernel(enable_backward=False, module="unique")
    def axpy_largearray(x: wp.largearray[float], y: wp.largearray[float], a: float):
        i = wp.linear_tid()
        y[i] = a * x[i] + y[i]

    return axpy_largearray


def _get_kernel(variant):
    if variant == "array":
        return axpy_array
    if not hasattr(wp, "largearray"):
        raise NotImplementedError("wp.largearray is not available in this Warp version")
    return _axpy_largearray()


def _record_axpy(variant, n, device):
    """Allocate ``x`` and ``y`` with ``n`` elements and return the recorded launch and the arrays."""
    kernel = _get_kernel(variant)
    x = wp.ones(n, dtype=float, device=device)
    y = wp.zeros(n, dtype=float, device=device)
    cmd = wp.launch(kernel, dim=n, inputs=[x, y, 2.0], device=device, record_cmd=True)
    return cmd, x, y


class LargeArrayAxpyCPU:
    """Time ``axpy`` over 2**24 ``float32`` elements on the CPU."""

    params = (VARIANTS,)
    param_names = ("variant",)
    repeat = 10
    number = 1

    @setup_once
    def setup(self, variant):
        wp.init()
        self.cmd, self.x, self.y = _record_axpy(variant, N, "cpu")
        # Warm up; this also compiles and loads the module.
        self.cmd.launch()

    def time_axpy(self, variant):
        self.cmd.launch()


class LargeArrayAxpyCUDA:
    """Time ``axpy`` over 2**24 ``float32`` elements on CUDA, measured on the GPU timeline."""

    timer = CudaEventTimer()
    params = (VARIANTS,)
    param_names = ("variant",)
    repeat = 10
    number = 1

    @setup_once
    def setup(self, variant):
        wp.init()
        self.cmd, self.x, self.y = _record_axpy(variant, N, "cuda:0")
        self.cmd.launch()
        wp.synchronize_device("cuda:0")
        self.graph = capture_launches(self.cmd.launch, LAUNCHES_PER_SAMPLE)
        wp.capture_launch(self.graph)
        wp.synchronize_device("cuda:0")

    def time_axpy(self, variant):
        wp.capture_launch(self.graph)


class LargeArrayLaunchOverhead:
    """Time the host overhead of a ``dim=1`` ``axpy`` launch, including argument packing.

    A synchronize at the end of the timed function is intentionally omitted.
    """

    params = (VARIANTS, ("cpu", "cuda:0"))
    param_names = ("variant", "device")
    number = 1000

    @setup_once
    def setup(self, variant, device):
        wp.init()
        self.kernel = _get_kernel(variant)
        self.device = device
        self.x = wp.ones(1, dtype=float, device=device)
        self.y = wp.zeros(1, dtype=float, device=device)
        wp.launch(self.kernel, dim=1, inputs=[self.x, self.y, 2.0], device=device)
        wp.synchronize_device(device)

    def teardown(self, variant, device):
        wp.synchronize_device(device)

    def time_launch(self, variant, device):
        wp.launch(self.kernel, dim=1, inputs=[self.x, self.y, 2.0], device=self.device)
