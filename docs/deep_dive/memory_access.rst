CPU/GPU cross-device memory access
==================================

.. currentmodule:: warp

Warp arrays are associated with an allocation :class:`Device` such as
``"cpu"`` or ``"cuda:0"``, and kernels run on a launch device. The portable
default is to launch kernels on the same device as their array arguments. On
systems with hardware-supported CPU/GPU memory access, some cross-device
patterns can also be valid: a GPU may be able to read or write unpinned CPU
memory directly, and some systems can let CPU code directly access GPU-resident
CUDA managed memory.

This page describes how Warp exposes those hardware capabilities and how to use
them when writing mixed CPU/GPU code.


Launching with arrays on the same device
----------------------------------------

The launch device determines where a kernel runs, and the array device describes
where the array allocation lives:

.. code:: python

    cpu_array = wp.zeros(1024, dtype=float, device="cpu")
    gpu_array = wp.zeros(1024, dtype=float, device="cuda:0")

    wp.launch(kernel, dim=cpu_array.size, inputs=[cpu_array], device="cpu")
    wp.launch(kernel, dim=gpu_array.size, inputs=[gpu_array], device="cuda:0")

The same-device pattern works on all supported systems. Passing an array from
one device to a kernel running on another device depends on the capabilities of
the device that performs the access.


Device capability properties
----------------------------

Each device exposes CPU/GPU memory access and managed-memory properties:

- :attr:`Device.is_cpu_memory_access_from_gpu_supported <warp.Device.is_cpu_memory_access_from_gpu_supported>`
- :attr:`Device.is_gpu_memory_access_from_cpu_supported <warp.Device.is_gpu_memory_access_from_cpu_supported>`
- :attr:`Device.is_cpu_gpu_atomic_supported <warp.Device.is_cpu_gpu_atomic_supported>`
- :attr:`Device.is_managed_memory_supported <warp.Device.is_managed_memory_supported>`
- :attr:`Device.is_concurrent_managed_access_supported <warp.Device.is_concurrent_managed_access_supported>`

This deep dive focuses on how those capabilities affect cross-device launches,
managed memory, atomics, and diagnostics.

On CPU devices, these properties are always ``False``. On GPU devices, each
property describes a specific access path or operation; support for one does not
imply support for another. For example, a system can allow GPU access to CPU
memory without allowing CPU access to GPU-resident managed memory.


Common CPU/GPU memory models
----------------------------

The exact values are reported by the CUDA driver and may vary by platform,
driver, kernel, and GPU generation. The following table summarizes the models
advanced users commonly need to reason about:

.. list-table::
   :header-rows: 1
   :widths: 24 25 25 26

   * - System model
     - GPU access to CPU arrays
     - CPU access to GPU-resident managed memory
     - CPU/GPU atomics
   * - Discrete GPU without HMM
     - Usually no
     - Usually no
     - Usually no
   * - Discrete GPU with Linux HMM
     - Yes
     - Usually no
     - Usually no
   * - Jetson Thor-style ATS
     - Yes
     - Platform-dependent for managed memory
     - Hardware capability only; current Warp CPU/GPU atomics unsupported
   * - Host-page-table ATS with distinct CPU/GPU physical memory
     - Yes
     - Only when reported by the driver
     - Hardware capability only; current Warp CPU/GPU atomics unsupported

HMM stands for Heterogeneous Memory Management; for background, see NVIDIA's
`HMM overview <https://developer.nvidia.com/blog/simplifying-gpu-application-development-with-heterogeneous-memory-management/>`__.
ATS stands for Address Translation Services. Warp does not require users to
classify the platform manually. Query the :class:`Device` properties and branch
on the behavior your program needs.

For the CUDA-level model behind these categories, see the CUDA Programming
Guide's `Unified and System Memory chapter
<https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html>`__,
especially its `Unified Memory paradigms table
<https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#table-unified-memory-levels>`__.

Do not infer CPU access to GPU-resident CUDA managed memory from ATS, C2C, or a
product family name. For example, a DGX Spark-class GB10 system can report ATS
and GPU access to CPU memory while
``device.is_gpu_memory_access_from_cpu_supported`` is ``False``. Query the
property directly before CPU code reads or writes GPU-resident managed memory.


Launching GPU kernels with CPU arrays
-------------------------------------

When ``device.is_cpu_memory_access_from_gpu_supported`` is true, a GPU kernel can
directly read or write a CPU array:

.. code:: python

    device = wp.get_device("cuda:0")
    a = wp.zeros(1024, dtype=float, device="cpu")

    if device.is_cpu_memory_access_from_gpu_supported:
        wp.launch(kernel, dim=a.size, inputs=[a], device=device)
    else:
        a_gpu = a.to(device)
        wp.launch(kernel, dim=a_gpu.size, inputs=[a_gpu], device=device)

This can avoid explicit copies on HMM and coherent CPU/GPU systems. If the
capability is false and the kernel actually dereferences the CPU pointer, CUDA
will report a runtime error such as an illegal memory access.


Accessing GPU data from CPU code
--------------------------------

CPU access to GPU-resident managed memory is a separate capability:

.. code:: python

    device = wp.get_device("cuda:0")
    if device.is_gpu_memory_access_from_cpu_supported:
        ...

CUDA arrays created by standard Warp array constructors, such as
:func:`zeros`, :func:`empty`, and :func:`ones`, are not CUDA managed-memory
allocations. This is true whether the array comes from Warp's :ref:`mempool
allocator <mempool_allocators>` or the built-in default CUDA allocator. For
those arrays, use an explicit copy before CPU code reads or writes the data:

.. code:: python

    a = wp.zeros(1024, dtype=float, device=device)
    a_cpu = a.to("cpu")
    wp.launch(cpu_kernel, dim=a_cpu.size, inputs=[a_cpu], device="cpu")

For explicit CUDA managed-memory arrays, construct a :class:`ManagedAllocator`
and install it with the existing allocator APIs. The allocator instance is not
bound to one CUDA device, but each allocation still happens under the target
device's CUDA context and that device must report CUDA managed-memory support:

.. code:: python

    managed = wp.ManagedAllocator()
    device = wp.get_device("cuda:0")

    with wp.ScopedAllocator(device, managed):
        a = wp.zeros(1024, dtype=float, device=device)

Managed arrays remain CUDA arrays in Warp: ``a.device`` is still ``"cuda:0"``,
and CUDA Unified Memory manages physical page migration. Their
:attr:`allocation_kind <warp.array.allocation_kind>` is
``wp.AllocationKind.CUDA_MANAGED``; this reports allocation provenance, not
current physical residency. See
:ref:`the allocator comparison <managed_memory_allocation_options>` for how
managed memory differs from Warp's default CUDA, CUDA mempool, and pinned CPU
allocation options.

This is the opt-in Warp allocation path for CPU kernels that need to operate
directly on CUDA-side data without maintaining a separate CPU copy. Ordinary
Warp CUDA arrays remain non-managed allocations and still need explicit copies
before CPU kernels read or write them.

Managed arrays can be used by kernels captured in CUDA graphs. Allocating a new
managed array while CUDA graph capture is active requires a CUDA 13.0+ Warp
build and managed memory-pool support on the device. CUDA 12.x builds, including
Warp's CUDA 12.9 PyPI build, allocate managed arrays through
``cudaMallocManaged()`` outside capture only; create those arrays before capture
begins.

Use :func:`can_access` before CPU code directly reads or writes a managed array:

.. code:: python

    if wp.can_access("cpu", a):
        wp.launch(cpu_kernel, dim=a.size, inputs=[a], device="cpu")
    else:
        a_cpu = a.to("cpu")
        wp.launch(cpu_kernel, dim=a_cpu.size, inputs=[a_cpu], device="cpu")

``wp.can_access("cpu", a)`` returns ``True`` for a managed CUDA array only when
the owning CUDA device reports concurrent managed access or direct CPU access to
GPU memory. On limited managed-memory systems, Warp returns ``False`` because it
cannot prove that a direct CPU access is synchronized with GPU use.

Do not assume that GPU access to CPU memory implies CPU access to GPU-resident
memory. Some systems support the former but not the latter.


Checking access for a specific array with ``wp.can_access()``
-------------------------------------------------------------

The function :func:`warp.can_access` answers whether code running on one device
can directly access a specific Warp array:

.. code:: python

    launch_device = wp.get_device("cuda:0")
    data = wp.empty(1024, dtype=float, device="cpu")

    if wp.can_access(launch_device, data):
        ...

For CPU arrays passed to CUDA kernels, pinned CPU arrays are accepted on CUDA
devices with unified virtual addressing, and unpinned CPU arrays require
``is_cpu_memory_access_from_gpu_supported``. For CUDA arrays, managed-memory
allocations use CUDA managed-memory support on the launch device, default CUDA
allocations use CUDA peer-access state, and memory pool allocations use
memory-pool access state. See :ref:`mempool_access` for the distinction between
peer access for default CUDA allocations and memory-pool access for mempool
allocations.

``wp.can_access(device, array)`` returns ``False`` when Warp cannot verify that
the array is directly accessible. This includes cross-device arrays backed by
custom allocators or externally wrapped allocations whose allocation kind is not
known to Warp. A ``False`` result means "not verified accessible"; it does not
prove that the hardware could never access the pointer.

``wp.can_access()`` is a resource-oriented API. In this release, the second
argument must be a concrete Warp array instance. Annotation-only arrays such as
``wp.array(dtype=float)`` or ``wp.array[float]`` and device objects are not
supported.


Checking coarse device access with ``Device.can_access()``
----------------------------------------------------------

The method :meth:`Device.can_access` is a coarse device-level query for cases
where no concrete array is available:

.. code:: python

    launch_device = wp.get_device("cuda:0")
    array_device = wp.get_device("cpu")

    if launch_device.can_access(array_device):
        ...

For GPU kernels accessing CPU arrays, this method uses
``is_cpu_memory_access_from_gpu_supported`` because standard Warp CPU arrays use
unpinned CPU memory. For CPU code accessing CUDA arrays, it returns ``False`` for
standard Warp CUDA arrays because the built-in CUDA allocators do not create
CUDA managed-memory allocations. For GPU/GPU pairs, it reflects the target
device's current built-in allocator mode: memory-pool access when memory pools
are enabled on the target device, and peer access otherwise.

``Device.can_access()`` is not authoritative for existing arrays. An array may
have been allocated before memory-pool settings changed, may use a custom
allocator, or may wrap external memory. Code that has an actual array should use
``wp.can_access(device, array)`` instead.


.. _launch_array_access_checks:

Checking array access before launch
-----------------------------------

By default, Warp launches kernels after type, dtype, and dimension validation
without checking array accessibility. This keeps the launch path lightweight and
allows hardware-supported mixed CPU/GPU launches to work.

If you want a clear Python error before the kernel runs, set
:attr:`warp.config.launch_array_access_mode`:

.. code:: python

    wp.config.launch_array_access_mode = wp.config.LaunchArrayAccessMode.CHECKED

- ``wp.config.LaunchArrayAccessMode.RELAXED`` is the default and performs no pre-launch
  array access checks beyond type, dtype, and dimension validation.
- ``wp.config.LaunchArrayAccessMode.STRICT`` restores Warp's original same-device rule
  and requires every Warp array argument to be allocated on the launch device.
- ``wp.config.LaunchArrayAccessMode.CHECKED`` raises an error before launch when Warp can
  determine that a cross-device Warp array argument is not accessible from the
  launch device. This is useful when debugging mixed-device launches on systems
  that do not support direct CPU/GPU memory access or on multi-GPU systems where
  peer and memory-pool access are configured separately.

Arrays backed by custom or externally wrapped allocators are a limitation of this
diagnostic. Warp does not know the allocation kind for those arrays, so
``wp.config.LaunchArrayAccessMode.CHECKED`` emits a ``UserWarning`` once per
``(kernel, argument name, source device, launch device)`` pattern and allows the
launch to proceed. Use ``wp.config.LaunchArrayAccessMode.STRICT`` if unknown allocation
provenance should be rejected, or ``wp.config.LaunchArrayAccessMode.RELAXED`` to suppress
the diagnostic.

Objects exposing ``__array_interface__`` are accepted only for CPU launches.
Warp treats that protocol as a CPU-addressable pointer and does not infer CUDA
allocation provenance from it, so ``wp.config.LaunchArrayAccessMode.CHECKED`` has no
cross-device access decision to make for that protocol.

Directly passing an object that exposes ``__cuda_array_interface__`` is
different from passing a Warp array. The protocol lets Warp construct the kernel
argument at launch time, but it does not identify the allocation device or
allocation kind. In this phase, ``wp.config.LaunchArrayAccessMode.CHECKED`` does not fully
verify directly passed objects exposing this protocol. Advanced users who know
such an allocation is valid are responsible for ensuring that the launch device
can legally access the pointer.

.. code:: python

    wp.config.launch_array_access_mode = wp.config.LaunchArrayAccessMode.CHECKED
    wp.launch(kernel, dim=a.size, inputs=[a], device="cuda:0")

:attr:`warp.config.launch_array_access_mode` can add launch overhead in
``wp.config.LaunchArrayAccessMode.STRICT`` and ``wp.config.LaunchArrayAccessMode.CHECKED`` modes.
Use ``wp.config.LaunchArrayAccessMode.RELAXED`` in performance-sensitive code that has
already validated its launch accessibility assumptions.

Unlike :attr:`warp.config.verify_cuda`,
:attr:`warp.config.launch_array_access_mode` can be used during CUDA graph
capture because ``wp.config.LaunchArrayAccessMode.CHECKED`` checks run before each launch
is recorded. For cross-GPU graph capture, enable peer access or memory-pool
access with Warp APIs before capture begins so verification can use the recorded
access state during capture. When a CUDA graph captures a launch with CPU array
arguments, replay uses the same captured CPU pointers. If the arrays remain
alive, CPU updates made between replays are visible to kernels on devices that
can access CPU memory.


Checking CPU/GPU atomic capability
----------------------------------

Direct loads and stores do not imply atomic safety. Warp exposes
:attr:`Device.is_cpu_gpu_atomic_supported <warp.Device.is_cpu_gpu_atomic_supported>`
so users and diagnostics can see the CUDA hardware capability reported for a
device. Treat this property as a hardware capability bit only. It is not a Warp
API contract that :func:`wp.atomic_add() <warp._src.lang.atomic_add>` or the
other ``wp.atomic_*`` functions can be used concurrently from CPU and GPU
kernels.

.. warning::

   Current Warp CPU ``wp.atomic_*`` operations are not lowered to CPU hardware
   atomics. They are ordinary read/modify/write updates in generated CPU code,
   which is sufficient for Warp's serial CPU kernel execution model but not
   safe when CPU and GPU code touch the same address concurrently. CUDA-side
   ``wp.atomic_*`` operations are also not documented as system-scope
   host/device atomics. Do not rely on overlapping CPU/GPU ``wp.atomic_*``
   updates, even when
   :attr:`Device.is_cpu_gpu_atomic_supported <warp.Device.is_cpu_gpu_atomic_supported>`
   is ``True``.

This limitation is separate from memory accessibility. A GPU may be able to
read or write CPU memory, and a device may report CPU/GPU atomic hardware
support, while current Warp atomics still are not safe for overlapping CPU/GPU
updates. If CPU and GPU work must update the same allocation today, sequence
ownership with explicit synchronization, copy through separate buffers, or use
custom native code that implements real CPU atomics and the required GPU
system-scope operations.

The property also does not make non-managed CUDA allocations CPU-accessible. CPU
code should still copy arrays backed by those allocations before reading or
writing them:

.. code:: python

    values = wp.zeros(1024, dtype=float, device=device)
    values_cpu = values.to("cpu")


Choosing a memory access pattern
--------------------------------

Use the same-device pattern unless you need zero-copy CPU/GPU access. When you
already have an array, use :func:`wp.can_access(device, array) <warp.can_access>`
to decide whether a specific launch device can directly access that allocation.
Capability flags are most useful before allocation, when deciding what kind of
allocation or access pattern to create:

- GPU kernel reads or writes unpinned CPU arrays: check
  ``device.is_cpu_memory_access_from_gpu_supported``.
- GPU kernel reads or writes pinned CPU arrays: use ``pinned=True`` and check
  ``device.is_uva``.
- CPU code reads or writes arrays backed by non-managed CUDA allocations: copy
  the data to ``"cpu"`` first.
- CPU kernels read or write Warp CUDA arrays directly: allocate those arrays
  with :class:`ManagedAllocator` and use ``wp.can_access("cpu", array)`` before
  launching the CPU kernel.
- CPU code accesses externally provided GPU-resident CUDA managed memory: check
  ``device.is_gpu_memory_access_from_cpu_supported``.
- CPU and GPU both need to update the same allocation: do not overlap current
  Warp ``wp.atomic_*`` operations across CPU and GPU kernels. Sequence ownership
  with synchronization or use a custom native implementation for true
  interprocessor atomics.
- GPU kernels use arrays from another GPU: enable peer access for default CUDA
  allocations, or :ref:`memory-pool access <mempool_access>` for CUDA
  memory-pool allocations, then check the concrete array with
  :func:`wp.can_access(device, array) <warp.can_access>`.
- Debugging mixed-device launch failures: temporarily set
  :attr:`warp.config.launch_array_access_mode` to
  ``wp.config.LaunchArrayAccessMode.CHECKED``.

Prefer capability checks over platform-name checks. They make code portable
across discrete GPUs, HMM-enabled systems, Jetson, Grace, and future coherent
CPU/GPU platforms.
