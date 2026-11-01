.. _communication_layers:

Choosing a Communication Layer
==============================

.. currentmodule:: warp

Warp runs kernels on the GPU of one process.
Moving data between the processes of a distributed program is the job of a communication library,
which the program calls on the host or, with NVSHMEM, from inside kernels.
This page shows how to give each process its GPU, compares the ways to communicate from a Warp program,
and has a short note for each host-side library on how to hand it a Warp array and what to synchronize.
The libraries' own documentation covers the rest, and the notes link to it.

.. _communication_layers_devices:

Processes and Devices
---------------------

A distributed Warp program runs one process per GPU, started by an external launcher such as ``mpirun`` or ``srun``.
Python multiprocessing can start the workers on a single host, and a cluster launcher starts them across nodes.
Each process selects its GPU before it allocates Warp arrays, loads modules, or captures graphs.

The GPU index comes from the process's rank within its node.
With MPI, split the communicator by node and take the rank in the node-local communicator, as the examples do.
Without MPI, read the local-rank variable that the launcher sets.

.. code-block:: python

    from mpi4py import MPI

    import warp as wp

    local_rank = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED).Get_rank()
    wp.set_device(f"cuda:{local_rank}" if wp.get_cuda_device_count() > 1 else "cuda:0")

``CUDA_VISIBLE_DEVICES`` restricts the GPUs that a process sees, and the GPUs it exposes are numbered from ``cuda:0``.
A launcher that exposes one GPU to each process therefore makes every process see its GPU as ``cuda:0``,
which the conditional above handles.
The CUDA Programming Guide's `CUDA Environment Variables
<https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html>`__ chapter
explains device visibility and enumeration.

Call :func:`wp.set_device() <warp.set_device>` before any library that needs a CUDA context.
For a CUDA device it makes the device's context current as well as setting Warp's default device,
and NVSHMEM's initialization needs that context (see :ref:`nvshmem_initialization`).
If another framework selected the GPU before :func:`wp.init() <warp.init>`,
Warp adopts that framework's current CUDA context as its default device.

.. _communication_layers_table:

The Layers
----------

Three libraries move data between processes, and NVSHMEM offers two ways to call it.
NVSHMEM calls each process a processing element (PE).
The layers differ in who takes part in a transfer, where the call is made, and what each is good at:

.. list-table::
    :header-rows: 1
    :widths: 15 15 20 26 24

    * - Layer
      - Initiated by
      - Called from
      - Suits
      - Example
    * - MPI (mpi4py)
      - Sender and receiver
      - Host, after the GPU work is done
      - Programs that already use MPI, with exchanges and reductions driven from the host
      - `example_jacobi_mpi.py
        <https://github.com/NVIDIA/warp/blob/main/warp/examples/distributed/example_jacobi_mpi.py>`__
    * - NCCL (nccl4py)
      - Sender and receiver
      - Host, on a CUDA stream
      - Collectives and exchanges in stream order with Warp's kernels, also inside a CUDA graph
      - `example_jacobi_nccl.py
        <https://github.com/NVIDIA/warp/blob/main/warp/examples/distributed/example_jacobi_nccl.py>`__
    * - NVSHMEM host API (nvshmem4py)
      - Calling PE (put or get)
      - Host, on a CUDA stream
      - A few large transfers, with no changes to kernels
      - `example_jacobi_nvshmem.py
        <https://github.com/NVIDIA/warp/blob/main/warp/examples/distributed/example_jacobi_nvshmem.py>`__
    * - :ref:`NVSHMEM in kernels <nvshmem_in_kernels>` (``wp.nvshmem_*``)
      - Calling PE (put or get)
      - Kernel threads
      - Fine-grained or data-dependent transfers, and communication fused with computation
      - `example_jacobi_nvshmem_device.py
        <https://github.com/NVIDIA/warp/blob/main/warp/examples/distributed/example_jacobi_nvshmem_device.py>`__

In the two NVSHMEM rows, the PE at the other end of a put or get posts no matching call.

The layers can be combined.
The NCCL example broadcasts its unique ID over MPI,
and the nvshmem4py examples initialize NVSHMEM with the MPI communicator.
Each example solves the same Laplace problem with Jacobi iteration on a row-slab decomposition,
so reading two of them side by side shows what changes from one layer to the next.

.. _communication_layers_mpi:

MPI with mpi4py
---------------

mpi4py's upper-case methods, such as ``Sendrecv`` and ``Allreduce``, accept GPU arrays
only when mpi4py is built against a CUDA-aware MPI library,
and only if the arrays are contiguous and ready when the call starts,
as the `mpi4py tutorial <https://mpi4py.readthedocs.io/en/stable/tutorial.html>`__ describes.
Warp arrays expose ``__cuda_array_interface__`` and DLPack (see :doc:`/user_guide/interoperability`),
which mpi4py accepts,
so a row of a two-dimensional Warp array can be sent as it is and a column cannot.
mpi4py cannot synchronize GPU work for you,
so call :func:`wp.synchronize_device() <warp.synchronize_device>` before handing a buffer to MPI,
or wait for an event recorded after the producing launch
with :func:`wp.synchronize_event() <warp.synchronize_event>`.
mpi4py's `installation guide <https://mpi4py.readthedocs.io/en/stable/install.html>`__ says that
the wheels on PyPI may lack GPU awareness,
and that the ``openmpi`` package on conda-forge has built-in CUDA support that is disabled by default,
so confirm that the MPI library behind mpi4py has CUDA support enabled.
One way to get a CUDA-aware stack is to download `NVIDIA HPC-X <https://developer.nvidia.com/networking/hpc-x>`__
and build mpi4py from source against its ``mpicc``,
following the guide's instructions for
`building from sources <https://mpi4py.readthedocs.io/en/stable/install.html#building-from-sources>`__.
`example_jacobi_mpi.py
<https://github.com/NVIDIA/warp/blob/main/warp/examples/distributed/example_jacobi_mpi.py>`__,
before each exchange, waits for an event recorded after the launch that produced the data.

.. _communication_layers_nccl:

NCCL with nccl4py
-----------------

`nccl4py <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py.html>`__ accepts a ``cuda.core.Buffer``
or any object that implements DLPack or the CUDA Array Interface, which includes Warp arrays.
Its `point-to-point <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/p2p.html>`__
and collective methods take the CUDA stream to run on.
A :class:`wp.Stream <warp.Stream>` can be passed as it is, because it implements the ``__cuda_stream__`` protocol.
Leaving the stream unset runs the operation on the default stream,
so pass the stream that carries the Warp launches,
and the communication then runs in stream order with the kernels.
Communication on another stream needs events to order it against Warp's.
The example creates its communicator from a unique ID that rank 0 generates and broadcasts over MPI,
and it captures the kernels and the communication into one CUDA graph:
`example_jacobi_nccl.py <https://github.com/NVIDIA/warp/blob/main/warp/examples/distributed/example_jacobi_nccl.py>`__.

.. _communication_layers_nvshmem_host:

NVSHMEM Host API with nvshmem4py
--------------------------------

The host API of `nvshmem4py <https://docs.nvidia.com/nvshmem/api/latest/api/language_bindings/python/index.html>`__
moves data between symmetric buffers from the host, on a CUDA stream, and the kernels contain no NVSHMEM calls.
It does not accept Warp arrays:
its `RMA calls <https://docs.nvidia.com/nvshmem/api/latest/api/language_bindings/python/rma.html>`__
take a ``cuda.core.Buffer``, CuPy arrays or Torch tensors,
and a stream, which can be a ``wp.Stream``.
It works on the symmetric buffers that Warp arrays wrap:
allocate with ``nvshmem.buffer()`` and wrap the buffer as a Warp array for the kernels
(see :ref:`nvshmem_symmetric_memory`),
and wrap the byte ranges to transfer with ``cuda.core.Buffer.from_handle()`` for the puts.
Initialize NVSHMEM as in :ref:`nvshmem_initialization`.
Prefer the host API when the kernels need no changes and the exchange is a few large transfers,
and prefer :ref:`the built-ins <nvshmem_in_kernels>` when communication is fine-grained,
depends on data that a kernel computes, or has to be fused with the computation.
`example_jacobi_nvshmem.py
<https://github.com/NVIDIA/warp/blob/main/warp/examples/distributed/example_jacobi_nvshmem.py>`__
shows the host API.
