# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class RmmAllocator:
    """Allocator that routes Warp device memory through RAPIDS Memory Manager (RMM).

    Each allocation delegates to ``rmm.DeviceBuffer``, which uses whichever
    ``DeviceMemoryResource`` is active at the time of allocation (as set by
    ``rmm.mr.set_current_device_resource()``). Changing the RMM resource
    between allocations will affect subsequent allocations.

    Requires the ``rmm`` package (Linux only, ``pip install rmm-cu12``).

    A single ``RmmAllocator`` instance can safely be shared across multiple
    CUDA devices. Allocations always happen on the correct device because
    ``warp.array`` wraps each ``allocate()`` call in a ``device.context_guard``.
    This class is not thread-safe; concurrent calls from multiple threads
    require external synchronization.

    Args:
        stream: An RMM stream object (``rmm.pylibrmm.stream.Stream``) to use
            for stream-ordered allocation. When provided, each ``DeviceBuffer``
            is allocated on the given stream, enabling stream-ordered memory
            reuse and allowing allocation during CUDA graph capture when paired
            with a stream-ordered ``DeviceMemoryResource`` such as
            ``rmm.mr.CudaAsyncMemoryResource``. When ``None`` (default), RMM's
            default stream is used.

            The caller is responsible for constructing the stream object.
            Example using ``cuda.bindings``:

            .. code-block:: python

                from cuda.bindings import runtime as cudart
                from rmm.pylibrmm.stream import Stream as RmmStream

                device = wp.get_device("cuda:0")
                cuda_stream = cudart.cudaStream_t(int(device.stream.cuda_stream))
                wp.set_device_allocator(device, wp.RmmAllocator(stream=RmmStream(obj=cuda_stream)))

    Example:
        .. code-block:: python

            import rmm
            import warp as wp

            rmm.reinitialize(pool_allocator=True, initial_pool_size=2**30)
            wp.set_allocator(wp.RmmAllocator())
            # All subsequent wp.array allocations go through the RMM pool
    """

    def __init__(self, stream=None):
        try:
            import rmm  # noqa: PLC0415, F401
        except ImportError as e:
            raise ImportError(
                "Failed to import 'rmm'. Ensure it is installed and compatible with your CUDA version. "
                "See https://docs.rapids.ai/install/ for installation instructions."
            ) from e
        self._stream = stream
        self._buffers: dict[int, object] = {}

    def allocate(self, size_in_bytes: int) -> int:
        """Allocate device memory via RMM and return a device pointer."""
        if size_in_bytes == 0:
            return 0
        import rmm  # noqa: PLC0415

        buf = (
            rmm.DeviceBuffer(size=size_in_bytes)
            if self._stream is None
            else rmm.DeviceBuffer(size=size_in_bytes, stream=self._stream)
        )
        ptr = buf.ptr
        self._buffers[ptr] = buf
        return ptr

    def deallocate(self, ptr: int, size_in_bytes: int) -> None:
        """Free device memory by releasing the RMM DeviceBuffer."""
        if ptr == 0:
            return  # Zero-size allocation; nothing was allocated.
        try:
            del self._buffers[ptr]
        except KeyError:
            raise RuntimeError(
                f"RmmAllocator.deallocate called with unrecognized pointer {ptr:#x} "
                f"(size={size_in_bytes}). This may indicate a double-free or a "
                f"pointer that was not allocated by this RmmAllocator instance."
            ) from None

    def __repr__(self):
        stream_repr = "default" if self._stream is None else repr(self._stream)
        return f"RmmAllocator(stream={stream_repr}, active_buffers={len(self._buffers)})"
