.. _distributed:

Distributed Computing
=====================

A distributed Warp program runs one process per GPU, started by an external launcher such as ``mpirun`` or ``srun``.
In each process, the program selects the process-local device with :func:`wp.set_device() <warp.set_device>`
before any device work.
Warp then compiles and launches kernels on that device,
and it wraps memory that another library allocated as Warp arrays without taking ownership of it.
Communication between the processes happens either on the host or inside kernels.
On the host, MPI and NCCL accept Warp arrays,
and nvshmem4py works on the symmetric buffers that Warp arrays wrap.
Inside kernels, Warp's NVSHMEM built-ins read and write the arrays of other processes.

Guides in this section
----------------------

* :doc:`Choosing a Communication Layer <distributed/communication_layers>`: Process and device selection,
  a comparison of MPI, NCCL and NVSHMEM, and notes on passing Warp arrays to each.
* :doc:`NVSHMEM in Warp Kernels <distributed/nvshmem>`: Requirements, initialization,
  symmetric memory, the NVSHMEM built-ins, ordering and synchronization rules, and error messages.
* :doc:`Halo Exchange with NVSHMEM <distributed/nvshmem_halo_exchange>`: A Jacobi solver built in stages,
  from element puts and a host barrier to signals, overlapped communication, a device-side reduction and CUDA graphs.
* :doc:`Automatic Differentiation with NVSHMEM <distributed/nvshmem_autodiff>`: Which built-ins carry gradients,
  gradient arrays in symmetric memory, how Warp synchronizes the backward pass, and the rules for correct gradients.

.. toctree::
   :hidden:
   :titlesonly:

   distributed/communication_layers
   distributed/nvshmem
   distributed/nvshmem_halo_exchange
   distributed/nvshmem_autodiff
