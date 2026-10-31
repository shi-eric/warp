# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Barriers and checks for the adjoint launches of NVSHMEM kernels.

The launch paths in ``warp._src.context`` and ``warp._src.deterministic`` import this module lazily: it defines a
Warp kernel, which needs a fully imported Warp, and those modules are part of that import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
import warp._src.context as warp_context
from warp._src.logger import log_warning

if TYPE_CHECKING:
    from warp._src.context import Device, Kernel, KernelHooks, Stream

# The barrier kernel is never differentiated, so skip generating its adjoint.
wp.set_module_options({"enable_backward": False})

_loaded_contexts: set[int] = set()
"""CUDA contexts of the devices on which the barrier kernel has been launched, so its module is loaded there.

Keyed by context because devices compare equal by context and are not hashable.
"""


@wp.kernel
def _barrier_all():
    wp.nvshmem_barrier_all()


def adjoint_barrier_required(hooks: KernelHooks) -> bool:
    """Return whether an adjoint launch of the kernel with ``hooks`` must be preceded by a barrier."""
    record = hooks.nvshmem_adjoint
    return bool(wp.config.nvshmem_adjoint_barrier and record is not None and record.remote_access)


def launch_adjoint_barrier(device: Device, stream: Stream) -> None:
    """Enqueue a barrier across all processing elements on ``stream``.

    The barrier is a one-thread kernel, so it is ordered after the work already on ``stream`` without blocking the
    host, and a CUDA graph can capture it.

    Raises:
        RuntimeError: If ``stream`` is capturing and the barrier kernel has not been loaded on ``device`` yet.
    """
    if device.context not in _loaded_contexts and stream.is_capturing:
        # Loading the module registers it with NVSHMEM, which is not known to be safe during a stream capture.
        raise RuntimeError(
            "Warp: the NVSHMEM adjoint barrier kernel is not loaded yet and the stream is capturing; "
            "run one backward pass before capturing a CUDA graph."
        )

    wp.launch(_barrier_all, dim=1, device=device, stream=stream, record_tape=False)
    _loaded_contexts.add(device.context)
    device.nvshmem_adjoint_barriers += 1


def warn_adjoint_launch(kernel: Kernel, hooks: KernelHooks) -> None:
    """Warn, once per kernel, about the NVSHMEM patterns in ``kernel`` that automatic differentiation does not support.

    The warnings fire at adjoint launch rather than at code generation, so that forward-only programs, for which these
    patterns are legitimate, see nothing.
    """
    record = hooks.nvshmem_adjoint
    if record is None:
        return

    if record.pull_builtins:
        log_warning(
            f'gradients do not flow through wp.nvshmem_g() or wp.nvshmem_get() in kernel "{kernel.key}"; '
            "restructure the exchange so the owning PE pushes the data with wp.nvshmem_p() or wp.nvshmem_put().",
            once=True,
        )
    if record.fused_shape:
        log_warning(
            f'kernel "{kernel.key}" writes to a remote PE before waiting on a signal; a cross-PE dependency within '
            "one launch is not supported by automatic differentiation. Move the wait to the start of the next launch.",
            once=True,
        )


def validate_symmetric_gradients(kernel: Kernel, fwd_args: list | None, adj_args: list | None) -> None:
    """Check that each symmetric array argument of an adjoint launch of ``kernel`` has a symmetric gradient.

    The adjoint of a push reads the destination's gradient on the remote PE at the address of the local one, which
    NVSHMEM translates only within the symmetric heap. The gradient checked is the one the launch uses: the adjoint
    array it passes for the argument, or the argument's ``grad`` when it passes none or an empty one, because the
    adjoint kernel falls back to ``grad`` for a null adjoint. Arguments whose memory NVSHMEM cannot classify are
    skipped.

    Args:
        kernel: The kernel whose adjoint is launched.
        fwd_args: The forward arguments of the launch, aligned with the kernel's arguments, or ``None`` to skip the
            check.
        adj_args: The adjoint arguments of the launch, aligned with the kernel's arguments, or ``None`` to skip the
            check.

    Raises:
        RuntimeError: If an array argument is in symmetric memory and its gradient array is not.
    """
    if fwd_args is None or adj_args is None:
        return

    is_symmetric = warp_context.runtime.core.wp_nvshmem_is_symmetric
    for i, arr in enumerate(fwd_args):
        if not isinstance(arr, wp.array) or not arr.ptr:
            continue
        adjoint = adj_args[i] if i < len(adj_args) else None
        if not isinstance(adjoint, wp.array) or not adjoint.ptr:
            adjoint = arr.grad
        if adjoint is None or not adjoint.ptr:
            continue
        # A result of -1 means NVSHMEM cannot tell, which skips the argument.
        if is_symmetric(arr.ptr) == 1 and is_symmetric(adjoint.ptr) == 0:
            raise RuntimeError(
                f'Warp: argument "{kernel.adj.args[i].label}" of kernel "{kernel.key}" is in NVSHMEM symmetric memory '
                "but its gradient array is not; allocate the gradient with nvshmem4py in the same order on every PE "
                "and assign it to array.grad."
            )


def before_adjoint_launch(
    kernel: Kernel,
    hooks: KernelHooks,
    device: Device,
    stream: Stream,
    fwd_args: list | None,
    adj_args: list | None,
) -> None:
    """Prepare an adjoint launch of ``kernel`` on ``stream``, before its adjoint kernel is enqueued.

    The launch paths call this on every PE for every adjoint launch of a kernel whose module uses NVSHMEM,
    including launches with no threads, so that the barriers match across PEs. It warns about NVSHMEM patterns that
    automatic differentiation does not support, validates the gradient arrays of kernels with remote gradient
    traffic, and enqueues the barrier.

    Args:
        kernel: The kernel whose adjoint is launched.
        hooks: The kernel hooks of the loaded module, which carry the kernel's NVSHMEM records.
        device: The CUDA device of the launch.
        stream: The stream the adjoint kernel is enqueued on.
        fwd_args: The forward arguments of the launch, or ``None`` when the launch does not retain them.
        adj_args: The adjoint arguments of the launch, which may contain ``None``, or ``None`` when the launch
            does not retain them.

    Raises:
        RuntimeError: If a symmetric array argument has a gradient array that is not symmetric.
    """
    warn_adjoint_launch(kernel, hooks)
    if hooks.nvshmem_adjoint is not None and hooks.nvshmem_adjoint.remote_access:
        validate_symmetric_gradients(kernel, fwd_args, adj_args)
    if adjoint_barrier_required(hooks):
        launch_adjoint_barrier(device, stream)
