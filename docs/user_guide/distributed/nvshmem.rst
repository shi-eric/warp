.. _nvshmem_in_kernels:

NVSHMEM in Warp Kernels
=======================

.. currentmodule:: warp

`NVSHMEM <https://docs.nvidia.com/nvshmem/api/latest/index.html>`__ implements the OpenSHMEM programming model
for clusters of NVIDIA GPUs:
a partitioned global address space that spans the memory of the GPUs in a job,
with operations that move data between GPUs from inside a CUDA kernel, on a CUDA stream, or from the host.
Each process of an NVSHMEM job is a processing element (PE) bound to one GPU.
Warp kernels call NVSHMEM's device API through the ``wp.nvshmem_*`` built-ins,
which write to and read from arrays on other PEs, signal other PEs, and synchronize with them.
Warp does not initialize NVSHMEM or allocate its memory.
The program does both with
`nvshmem4py <https://docs.nvidia.com/nvshmem/api/latest/api/language_bindings/python/index.html>`__,
NVSHMEM's Python bindings, and passes the memory to kernels as Warp arrays.

This page is the reference for that mechanism:
what to install, how to initialize NVSHMEM and allocate its memory,
what Warp does when it compiles a kernel that calls the built-ins,
what each built-in guarantees, and what Warp's error messages mean.

.. _nvshmem_requirements:

Requirements and Installation
-----------------------------

The NVSHMEM built-ins are available in Linux builds of Warp with CUDA support.
:func:`wp.is_nvshmem_enabled() <warp.is_nvshmem_enabled>` returns ``True``
when the installed build can link NVSHMEM device code into kernels.
Kernels that call the built-ins run only on CUDA devices,
and Warp cannot compile them with the ``llvm_cuda`` module option
(:attr:`wp.config.llvm_cuda <warp.config.llvm_cuda>`):
it raises ``NVSHMEM built-ins are not supported with the llvm_cuda module option.`` instead.

Install nvshmem4py and the NVSHMEM wheel for the CUDA major version that Warp was built with,
which is CUDA 13 for the PyPI wheels (see :doc:`/user_guide/installation`):

.. code-block:: sh

    pip install nvshmem4py-cu13 "nvidia-nvshmem-cu13>=3.9"  # or the -cu12 packages with the same floor

The ``nvidia-nvshmem-cu13`` or ``nvidia-nvshmem-cu12`` wheel contains the NVSHMEM host library.
nvshmem4py depends on it without a minimum version, so the command sets one.
The `nvshmem4py installation guide
<https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/nvshmem4py-install-proc.html>`__
covers the other installation channels and building from source.
Warp requires NVSHMEM 3.9 or newer,
because 3.9 is the first release whose wheels also ship the device library, ``libnvshmem_device.ltoir.fatbin``,
that Warp links into kernels that call the built-ins.

For a custom NVSHMEM build, set the ``WARP_NVSHMEM_DEVICE_LIBRARY`` environment variable
to that build's ``libnvshmem_device.ltoir.fatbin`` or to a raw ``.ltoir`` file.
Warp then links that file instead of the one it finds itself (see :ref:`nvshmem_first_launch`),
and it skips its check that the loaded host library is NVSHMEM 3.9 or newer.
NVSHMEM still checks that the device library and the host library are compatible when Warp loads the module.

Start one process per GPU with a launcher: ``mpirun`` or ``srun`` for a program that uses MPI,
or NVSHMEM's own ``nvshmrun``.
NVSHMEM's `launch documentation
<https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/nvshmem-install-proc.html#launching-nvshmem-programs>`__
lists the launchers it supports.

.. _nvshmem_symmetric_memory:

Processing Elements and Symmetric Memory
----------------------------------------

.. figure:: /img/distributed/symmetric_heap-light.svg
    :class: only-light
    :alt: Three PEs, each with a symmetric heap that holds the array u at the same offset; a kernel on PE 0 puts into u on PE 1, and Warp wraps each PE's copy of u as an array.
    :width: 100%

    Processing elements and the symmetric heap

.. figure:: /img/distributed/symmetric_heap-dark.svg
    :class: only-dark
    :alt: Three PEs, each with a symmetric heap that holds the array u at the same offset; a kernel on PE 0 puts into u on PE 1, and Warp wraps each PE's copy of u as an array.
    :width: 100%

    Processing elements and the symmetric heap

A PE is one process bound to one GPU.
Memory that other PEs can access lives in each PE's symmetric heap.
An allocation on the symmetric heap is collective:
every PE calls ``nvshmem.buffer()`` with the same size, in the same order as the other PEs,
and the result lives at the same offset in every PE's heap.
One array name therefore stands for ``n_pes`` copies, one on each PE,
and a kernel names the copy on another PE by passing its local array and that PE's index to a built-in.

Warp wraps the local copy as an array that does not own the memory:
``wp.array(ptr=int(buf.handle), shape=..., dtype=..., device=device)``.
Views, slices and multi-dimensional shapes work as for any other array,
and a built-in that receives a view accesses the same view on the remote PE.
An array that a kernel accesses on another PE must be in symmetric memory,
because NVSHMEM translates addresses between PEs only within symmetric memory.

The code on this page imports nvshmem4py as the examples do,
with ``import nvshmem.core as nvshmem`` and ``from nvshmem.bindings import nvshmem as nvshmem_bindings``.
The examples allocate each symmetric array with this helper,
which appends the buffer to a list, ``symmetric_buffers``, so that the program can free it at the end:

.. literalinclude:: ../../../warp/examples/distributed/example_jacobi_nvshmem_device.py
    :language: python
    :pyobject: symmetric

nvshmem4py does not zero the memory it allocates, so the helper zeroes each array.
Before any PE writes into another PE's arrays, every PE synchronizes its device and waits at a barrier,
so that no PE puts data into an array that its owner is still zeroing or initializing:

.. code-block:: python

    wp.synchronize_device(device)
    nvshmem_bindings.barrier_all()

Freeing is collective as well, and no PE may free memory that another PE still uses.
At the end of the program, every PE synchronizes its device, waits at a barrier,
and then frees its buffers in the same order as the other PEs, before ``nvshmem.finalize()``:

.. literalinclude:: ../../../warp/examples/distributed/example_jacobi_nvshmem_overlap.py
    :language: python
    :start-after: # docs: teardown begin
    :end-before: # docs: teardown end

An array that wraps a buffer must not be used after the buffer is freed.

``NVSHMEM_SYMMETRIC_SIZE`` sets the size of each PE's symmetric heap.
Set it in the environment before ``nvshmem.init()`` when the default is too small for the program's arrays;
NVSHMEM's `environment variable reference <https://docs.nvidia.com/nvshmem/api/latest/gen/env.html#standard-options>`__
gives the default and the accepted formats.
The `NVSHMEM memory model <https://docs.nvidia.com/nvshmem/api/latest/gen/mem-model.html>`__
has the complete rules for symmetric objects and their addresses,
and the `nvshmem4py memory management reference
<https://docs.nvidia.com/nvshmem/api/latest/api/language_bindings/python/memory_management.html>`__
describes ``nvshmem.buffer()`` and ``nvshmem.free()``.

.. _nvshmem_initialization:

Initialization Order
--------------------

Select the GPU with :func:`wp.set_device() <warp.set_device>` before calling ``nvshmem.init()``:
nvshmem4py binds to the CUDA context that ``wp.set_device()`` makes current.
With an MPI launcher, select the GPU from the process's rank within its node,
or ``cuda:0`` when a launcher such as ``srun --gpus-per-task=1`` lets each process see only its own GPU
(see :ref:`communication_layers_devices`),
and pass the MPI communicator with ``initializer_method="mpi"``:

.. code-block:: python

    import nvshmem.core as nvshmem
    from mpi4py import MPI
    from nvshmem.bindings import nvshmem as nvshmem_bindings

    import warp as wp

    mpi_comm = MPI.COMM_WORLD
    local_rank = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED).Get_rank()
    wp.set_device(f"cuda:{local_rank}" if wp.get_cuda_device_count() > 1 else "cuda:0")
    device = wp.get_device()
    nvshmem.init(mpi_comm=mpi_comm, initializer_method="mpi")
    my_pe = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()

Without MPI, for example under ``nvshmrun``, initialize from a unique ID instead.
One process creates the ID with ``nvshmem.get_unique_id()``,
the program delivers it to the other processes by any means it has, such as a file, a socket or a broadcast,
and every process passes it together with its own rank and the number of processes.
``nvshmrun`` provides the rank and the number of processes in ``PMI_RANK`` and ``PMI_SIZE``.

.. code-block:: python

    nvshmem.init(uid=uid, rank=rank, nranks=nranks, initializer_method="uid")

The `nvshmem4py initialization reference
<https://docs.nvidia.com/nvshmem/api/latest/api/language_bindings/python/initialization.html#initialization-methods>`__
describes both methods, including how the other processes create an empty ID to fill
and why mpi4py must be built against the MPI library that runs the job.

NVSHMEM must be initialized before the first launch of a kernel that calls the built-ins,
because Warp registers the kernel's module with NVSHMEM when it loads the module.

.. _nvshmem_first_launch:

What Warp Does on the First Launch
----------------------------------

.. figure:: /img/distributed/warp_nvshmem_stack-light.svg
    :class: only-light
    :alt: Inside one PE, the host runs the launcher, wp.set_device, nvshmem.init, nvshmem.buffer and wp.launch; on the first launch, Warp generates code, compiles it with NVRTC, links the NVSHMEM device library with nvJitLink, and registers the module with nvshmemx_cumodule_init.
    :width: 100%

    Warp and NVSHMEM inside one PE

.. figure:: /img/distributed/warp_nvshmem_stack-dark.svg
    :class: only-dark
    :alt: Inside one PE, the host runs the launcher, wp.set_device, nvshmem.init, nvshmem.buffer and wp.launch; on the first launch, Warp generates code, compiles it with NVRTC, links the NVSHMEM device library with nvJitLink, and registers the module with nvshmemx_cumodule_init.
    :width: 100%

    Warp and NVSHMEM inside one PE

Warp compiles a module on the first launch of one of its kernels, as described in :ref:`Compilation Model`.
For a module whose kernels call NVSHMEM built-ins,
NVRTC compiles the generated CUDA code with link-time optimization,
and nvJitLink links the result with the NVSHMEM device library.
After loading the module, Warp registers it with NVSHMEM through ``nvshmemx_cumodule_init``,
which sets up the NVSHMEM state that the device code reads.
Registration needs NVSHMEM to be initialized; compilation does not.
A machine with the NVSHMEM wheel installed can compile such modules without initializing NVSHMEM,
for example :ref:`ahead of time <ahead_of_time_compilation_workflows>`.
Modules whose kernels do not call the built-ins are compiled and loaded as usual.

Warp does not load NVSHMEM itself.
It uses the host library that the application already loaded, through nvshmem4py or another library,
and takes the device library from the same installation so that the two match.
It looks for the device library in this order:

1. The file that ``WARP_NVSHMEM_DEVICE_LIBRARY`` names, if the variable is set.
2. The directory of the loaded host library.
3. The installed ``nvidia-nvshmem-cu<major>`` wheel for the CUDA major version that Warp was built with,
   which lets Warp compile a module before NVSHMEM is loaded.

Warp records the SHA-256 digest of the device library with each module in the kernel cache,
and it compiles a cached module again if the module was linked against a different device library,
so upgrading NVSHMEM rebuilds the affected modules on their next launch.
Several processes can share one kernel cache directory:
each process builds a module in a directory of its own and then moves the result into place.

NVSHMEM cannot register a module while a stream is being captured into a CUDA graph,
and with a driver that supports CUDA 12.3 or newer,
Warp loads a module during a capture if no earlier launch has loaded it.
Launch the kernels once before capturing them, for example by running the first steps of a loop eagerly,
or capture with ``wp.ScopedCapture(force_module_load=True)``, which loads the modules before the capture begins.
Otherwise, NVSHMEM reports a CUDA error and the process exits (see :ref:`nvshmem_errors`).

.. _nvshmem_builtins:

The Built-ins
-------------

The built-ins are called from kernels.
Their array arguments must be symmetric arrays,
and their ``pe`` arguments are PE indices from ``0`` to ``wp.nvshmem_n_pes() - 1``.
The :doc:`language reference </language_reference/builtins>` lists their signatures.

PE Queries
~~~~~~~~~~

:func:`wp.nvshmem_my_pe() <warp._src.lang.nvshmem_my_pe>` returns the index of the calling PE,
and :func:`wp.nvshmem_n_pes() <warp._src.lang.nvshmem_n_pes>` returns the number of PEs.
They return the same values as ``nvshmem.my_pe()`` and ``nvshmem.n_pes()`` on the host.

Element Access
~~~~~~~~~~~~~~

:func:`wp.nvshmem_p() <warp._src.lang.nvshmem_p>` writes one element of a symmetric array on another PE,
and :func:`wp.nvshmem_g() <warp._src.lang.nvshmem_g>` reads one and returns it.
``wp.nvshmem_p()`` takes the array, one index per dimension, the value and then the PE index;
``wp.nvshmem_g()`` takes the array, one index per dimension and then the PE index.
The value must have the array's dtype.
On an ``int32`` array, a value and a PE index in swapped positions still pass the type check, so keep this order.
Each thread moves its own element, so a launch over a row moves the row one element per thread.
An element wider than 8 bytes is written in several parts, so the write is not atomic.

Bulk Copies
~~~~~~~~~~~

:func:`wp.nvshmem_put() <warp._src.lang.nvshmem_put>` copies a local array into an array on another PE,
and :func:`wp.nvshmem_get() <warp._src.lang.nvshmem_get>` copies an array on another PE into a local array.
:func:`wp.nvshmem_put_signal() <warp._src.lang.nvshmem_put_signal>` copies like ``wp.nvshmem_put()``
and then updates a signal word on the destination PE (see `Signals`_).
The two arrays have the same dtype and shape, with up to four dimensions;
slice them to transfer part of an array, such as one row.
The local array must also be symmetric memory,
as NVSHMEM's `remote memory access reference <https://docs.nvidia.com/nvshmem/api/latest/gen/api/rma.html>`__ requires,
or memory registered with NVSHMEM:
nvshmem4py's ``nvshmem.core.register_external_buffer()`` is a collective that registers a ``cuda.core`` buffer
and returns a new buffer to use in place of the original.
A copy with ``wp.nvshmem_get()`` is complete when the call returns;
a copy with ``wp.nvshmem_put()`` or ``wp.nvshmem_put_signal()`` is complete only after ``wp.nvshmem_quiet()``
or a barrier (see :ref:`nvshmem_ordering`).
The push built-ins ``wp.nvshmem_p()``, ``wp.nvshmem_put()`` and ``wp.nvshmem_put_signal()`` are differentiable
(see :doc:`nvshmem_autodiff`).

Signals
~~~~~~~

A signal word is an element of a symmetric ``wp.array[wp.uint64]``.
:func:`wp.nvshmem_signal_op() <warp._src.lang.nvshmem_signal_op>` updates a signal word on another PE:
:data:`wp.NVSHMEM_SIGNAL_SET <warp.NVSHMEM_SIGNAL_SET>` sets it to the given value,
and :data:`wp.NVSHMEM_SIGNAL_ADD <warp.NVSHMEM_SIGNAL_ADD>` adds the value to it.
:func:`wp.nvshmem_signal_wait_until() <warp._src.lang.nvshmem_signal_wait_until>` waits
until a local signal word satisfies a comparison with a value, and returns the word.
The comparisons are :data:`wp.NVSHMEM_CMP_EQ <warp.NVSHMEM_CMP_EQ>`, :data:`wp.NVSHMEM_CMP_NE <warp.NVSHMEM_CMP_NE>`,
:data:`wp.NVSHMEM_CMP_GT <warp.NVSHMEM_CMP_GT>`, :data:`wp.NVSHMEM_CMP_GE <warp.NVSHMEM_CMP_GE>`,
:data:`wp.NVSHMEM_CMP_LT <warp.NVSHMEM_CMP_LT>` and :data:`wp.NVSHMEM_CMP_LE <warp.NVSHMEM_CMP_LE>`.
Signal updates and waits on the same word are atomic with respect to each other,
as NVSHMEM's `signaling reference <https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html>`__ specifies.

Ordering and Barriers
~~~~~~~~~~~~~~~~~~~~~

:func:`wp.nvshmem_fence() <warp._src.lang.nvshmem_fence>`, :func:`wp.nvshmem_quiet() <warp._src.lang.nvshmem_quiet>`
and :func:`wp.nvshmem_barrier_all() <warp._src.lang.nvshmem_barrier_all>`
order, complete and synchronize the operations above.
:ref:`nvshmem_ordering` describes what each one guarantees.

Two rules apply to every kernel that calls the built-ins:

* **A bulk copy is issued by one thread for the whole array.**
  Call ``wp.nvshmem_put()``, ``wp.nvshmem_get()`` and ``wp.nvshmem_put_signal()`` from one thread per transfer,
  for example from a launch with ``dim=2`` that sends one row to each of two neighbors,
  or slice the arrays so that each thread copies a different part.
  If every thread of a large launch calls one of them with the same arrays, every thread copies the whole array.
* **wp.nvshmem_barrier_all() is called by exactly one thread per PE.**
  NVSHMEM `allows only one thread of a PE in a collective at a time
  <https://docs.nvidia.com/nvshmem/api/latest/gen/api/collectives.html#collective-operations-scopes-and-active-sets>`__,
  so call it from a kernel launched with ``dim=1`` (see :ref:`nvshmem_synchronization`).

.. _nvshmem_ordering:

Ordering and Completion
-----------------------

.. figure:: /img/distributed/ordering_completion-light.svg
    :class: only-light
    :alt: One thread issues put A, a fence, put B, a quiet, a signal and a barrier; brackets show what the fence, the quiet, the signal and the barrier each guarantee.
    :width: 100%

    Ordering and completion

.. figure:: /img/distributed/ordering_completion-dark.svg
    :class: only-dark
    :alt: One thread issues put A, a fence, put B, a quiet, a signal and a barrier; brackets show what the fence, the quiet, the signal and the barrier each guarantee.
    :width: 100%

    Ordering and completion

A put returns once its data has been copied out of the source, so the thread may overwrite the source,
but the data may not have arrived at the destination PE yet.
These calls give stronger guarantees:

* ``wp.nvshmem_fence()`` orders delivery:
  operations issued to a PE before the fence arrive at that PE before operations issued to it after the fence.
  It does not wait for any operation to complete.
* ``wp.nvshmem_quiet()`` waits until the operations are complete, which makes their data visible at the destination PE.
  It is local: no other PE learns that it happened.
* ``wp.nvshmem_put_signal()`` delivers the signal update after its own data,
  so a PE that waits on the signal word sees that copy complete.
* ``wp.nvshmem_barrier_all()`` returns when every PE has reached the barrier
  and all operations that the PEs issued before it are complete.

``wp.nvshmem_fence()`` orders the operations of the calling thread.
``wp.nvshmem_quiet()`` completes the operations of the calling thread
and those that other threads issued before the call, for example the threads of an earlier kernel on the same stream,
and ``wp.nvshmem_barrier_all()`` completes all previously issued operations before it synchronizes the PEs.
This is why a one-thread kernel that calls ``wp.nvshmem_quiet()`` or ``wp.nvshmem_barrier_all()`` after a launch
completes the puts of every thread of that launch.
The signal of a ``wp.nvshmem_put_signal()`` covers only its own data, not earlier puts:
to publish earlier puts that the thread issued to the same PE,
call ``wp.nvshmem_fence()`` before the ``wp.nvshmem_put_signal()``.

Kernel completion does not imply remote completion.
Blocking ordering calls on the host apply only to operations issued from the host,
and calls made in a kernel apply only to operations issued from the device.
For the host to rely on data that kernels put, a quiet or a barrier must run on the device after the puts,
and the host must then synchronize with its device.

.. list-table::
    :header-rows: 1
    :widths: 40 60

    * - I need
      - Call
    * - My puts to be complete and visible at their destinations.
      - ``wp.nvshmem_quiet()`` after the puts,
        in the thread that issued them or in a later kernel on the same stream.
    * - Two puts to the same PE to arrive in the order I issued them.
      - ``wp.nvshmem_fence()`` between them.
    * - To publish a row to another PE with one flag.
      - ``wp.nvshmem_put_signal()`` for the whole row, from one thread;
        the receiving PE waits on the flag with ``wp.nvshmem_signal_wait_until()``.
    * - Every PE to be done with the operations it issued so far.
      - ``wp.nvshmem_barrier_all()`` in a kernel launched with ``dim=1``.
    * - The host to read data that other PEs put into this PE's arrays.
      - A ``wp.nvshmem_barrier_all()`` kernel after the puts,
        then :func:`wp.synchronize_device() <warp.synchronize_device>` before reading.
        Alternatively, ``wp.nvshmem_quiet()`` at the end of the kernels that put,
        then ``wp.synchronize_device()`` and a host barrier (see :ref:`nvshmem_synchronization`).

NVSHMEM's `memory ordering reference <https://docs.nvidia.com/nvshmem/api/latest/gen/api/ordering.html>`__
and its `memory model <https://docs.nvidia.com/nvshmem/api/latest/gen/mem-model.html#ordering-of-operations>`__
give the complete rules.

.. _nvshmem_synchronization:

Synchronizing from the Host or from the Device
----------------------------------------------

A barrier across all PEs can run in three places.

**On the host, blocking.**
Each PE waits for its device, then for the other PEs:

.. code-block:: python

    wp.synchronize_device(device)
    nvshmem_bindings.barrier_all()

This is the simplest form, and the examples use it after allocating and before freeing symmetric memory.
It blocks the host at every barrier, and the device has no work queued while its host waits.
A host barrier completes only operations issued from the host,
so the puts it must cover need a ``wp.nvshmem_quiet()`` on the device first,
at the end of the kernels that put or in a one-thread kernel after them.

**On a stream, enqueued from the host.**
nvshmem4py's host collectives take a stream
(see `Stream Requirement for Collectives
<https://docs.nvidia.com/nvshmem/api/latest/api/language_bindings/python/collectives.html#stream-requirement-for-collectives>`__),
and they accept a :class:`wp.Stream <warp.Stream>` directly:

.. literalinclude:: ../../../warp/tests/distributed/aux_nvshmem_halo_stages.py
    :language: python
    :start-after: # docs: on-stream-barrier begin
    :end-before: # docs: on-stream-barrier end
    :dedent: 4

The call returns immediately, and the barrier runs in stream order, after the work enqueued before it.
NVSHMEM's `collectives reference <https://docs.nvidia.com/nvshmem/api/latest/gen/api/collectives.html>`__
lists ``nvshmemx_barrier_all_on_stream`` under ``nvshmem_barrier_all``,
which "ensures completion of all previously issued memory stores and remote memory updates",
and its notes say that stream-based quiet operations have the effect of a quiet executed on the GPU in stream order.
NVIDIA's reference Jacobi solver,
`nvshmem/jacobi.cu <https://github.com/NVIDIA/multi-gpu-programming-models/blob/master/nvshmem/jacobi.cu>`__
in the multi-GPU programming models repository, relies on that:
its kernel puts elements without a quiet, and a barrier on the stream follows each launch.
The kernels of the guides in this section end their puts with ``wp.nvshmem_quiet()``, so they do not depend on it.
Use this form when a barrier between two launches is all the program needs:
there is no kernel to write, and the host does not wait.
It cannot share a launch with other device work, which the kernel form can.

**In a kernel.**
A kernel launched with ``dim=1`` calls ``wp.nvshmem_barrier_all()`` from its only thread:

.. code-block:: python

    @wp.kernel
    def barrier_all():
        # NVSHMEM's device-side barrier is a collective of one thread per PE: launch with dim=1.
        wp.nvshmem_barrier_all()


    wp.launch(barrier_all, dim=1, device=device)

The barrier runs in stream order and keeps the host out of the loop,
and as an ordinary Warp launch it can be captured into a CUDA graph together with the kernels around it.
The guides in this section use this form wherever the host stays out of the loop.
Because the barrier is a call in a kernel, it can also end a one-thread kernel that does other work,
which saves a launch:
`example_nvshmem_basic.py
<https://github.com/NVIDIA/warp/blob/main/warp/examples/distributed/example_nvshmem_basic.py>`__
puts a value, calls ``wp.nvshmem_quiet()`` and then calls ``wp.nvshmem_barrier_all()``, all in one kernel.

A host-side collective, such as ``nvshmem.buffer()`` or ``nvshmem.free()``,
must not run while a barrier on a stream is still pending,
as NVSHMEM's notes on `CUDA streams <https://docs.nvidia.com/nvshmem/api/latest/cuda-interactions.html>`__ explain.
Call ``wp.synchronize_device()`` before allocating or freeing symmetric memory
in a program that enqueues barriers on a stream or in kernels.

.. _nvshmem_multi_node:

Running Across Nodes
--------------------

Across nodes, launch one process per GPU with ``mpirun`` or ``srun``,
and select each process's GPU from its rank within the node, as in :ref:`nvshmem_initialization`.
NVSHMEM chooses how to reach each PE:
directly between GPUs that can access each other's memory, and through a network transport between nodes.
NVSHMEM's `transport environment variables
<https://docs.nvidia.com/nvshmem/api/latest/gen/env.html#transport-options>`__
constrain that choice when the default does not suit a cluster.

The rules on this page do not change across nodes, but the cost of each operation does.
Between GPUs that can access each other's memory, an element put is a store to the other GPU's memory.
Across nodes, every element put is a network message of its own,
so a row sent one element per thread costs far more than the same row sent in one transfer.
Exchange data across nodes in whole rows,
with ``wp.nvshmem_put()`` or ``wp.nvshmem_put_signal()`` from one thread per row.
To measure element puts on a given system, run
`example_nvshmem_put_bw.py
<https://github.com/NVIDIA/warp/blob/main/warp/examples/distributed/example_nvshmem_put_bw.py>`__
with two PEs on one node and with two PEs on two nodes.

Every PE calls the collectives in the same order as the other PEs:
the host-side ones, such as ``nvshmem.buffer()``, ``nvshmem.free()``, ``nvshmem_bindings.barrier_all()``
and ``nvshmem.finalize()``, and the barriers on streams and in kernels.

.. _nvshmem_errors:

Errors and What They Mean
-------------------------

Warp raises these errors as ``RuntimeError`` when it compiles or loads a module whose kernels call the built-ins,
which happens on the first launch of one of the module's kernels.
The table lists the most common ones, in the order in which Warp checks for them,
and ends with a failure that NVSHMEM reports itself.
Two others are rarer:
``WARP_NVSHMEM_DEVICE_LIBRARY names '<path>', which is not a file.`` when the override names a missing file,
and ``Cannot load a module that uses NVSHMEM built-ins: nvshmemx_cumodule_init failed (<code>).``
when NVSHMEM rejects the module, most often because a device library set with ``WARP_NVSHMEM_DEVICE_LIBRARY``
does not match the loaded host library.
In the messages, ``<paths>``, ``<version>``, ``<path>`` and ``<code>`` stand for values that Warp fills in.

.. list-table::
    :header-rows: 1
    :widths: 50 50

    * - Message
      - Cause and fix
    * - ``This Warp build cannot link NVSHMEM device code (Linux CUDA builds only).``
      - The installed Warp build has no NVSHMEM support.
        Use a Linux build of Warp with CUDA support;
        :func:`wp.is_nvshmem_enabled() <warp.is_nvshmem_enabled>` returns ``True`` for one.
    * - ``Cannot find the NVSHMEM device library needed to compile a module that uses NVSHMEM built-ins.
        Tried: <paths>. Install NVSHMEM 3.9 or newer, whose wheels ship libnvshmem_device.ltoir.fatbin,
        or set WARP_NVSHMEM_DEVICE_LIBRARY to a libnvshmem_device.ltoir.fatbin or raw .ltoir file.``
      - No NVSHMEM 3.9 or newer installation is visible to Warp.
        Install ``nvshmem4py-cu13`` and ``"nvidia-nvshmem-cu13>=3.9"``, or the ``-cu12`` packages,
        to match the CUDA major version of the Warp build (see :ref:`nvshmem_requirements`);
        the message lists the places that Warp searched.
    * - ``Cannot load a module that uses NVSHMEM built-ins because no NVSHMEM host library is loaded in this process.
        Initialize NVSHMEM before loading the module, for example with nvshmem4py's nvshmem.core.init().``
      - The program launched the kernel before it loaded NVSHMEM.
        Call ``nvshmem.init()`` before the first launch, as in :ref:`nvshmem_initialization`.
    * - ``Warp requires NVSHMEM 3.9 or newer, but the loaded host library is NVSHMEM <version>.
        Install a newer nvidia-nvshmem wheel, or set WARP_NVSHMEM_DEVICE_LIBRARY to a device library
        that matches the loaded host library.``
      - The process loaded an NVSHMEM host library older than 3.9.
        Upgrade the ``nvidia-nvshmem-cu13`` or ``nvidia-nvshmem-cu12`` wheel.
        ``WARP_NVSHMEM_DEVICE_LIBRARY`` is meant for custom NVSHMEM builds (see :ref:`nvshmem_requirements`).
    * - ``Cannot load a module that uses NVSHMEM built-ins because NVSHMEM is not initialized.
        Initialize NVSHMEM before loading the module, for example with nvshmem4py's nvshmem.core.init().``
      - The NVSHMEM host library is loaded, for example because the program imported nvshmem4py,
        but ``nvshmem.init()`` has not run yet.
        Call it before the first launch; the failed load leaves nothing behind, so the next launch loads the module.
    * - ``cuda failed with operation would make the legacy stream depend on a capturing blocking stream``
        or ``cuda failed with operation not permitted when stream is capturing``
      - NVSHMEM prints these, and the process exits,
        when the first launch of a kernel that calls the built-ins happens during a CUDA graph capture:
        Warp then loads the module during the capture, and NVSHMEM cannot register it (see :ref:`nvshmem_first_launch`).
        Launch the kernels once eagerly before capturing them,
        or capture with ``wp.ScopedCapture(force_module_load=True)``, which loads the modules before the capture begins.
