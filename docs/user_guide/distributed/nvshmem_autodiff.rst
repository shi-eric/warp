.. _nvshmem_autodiff:

Automatic Differentiation with NVSHMEM
======================================

.. currentmodule:: warp

Kernels that exchange data between GPUs with the NVSHMEM built-ins can be differentiated
when each processing element (PE) pushes data into the symmetric arrays of other PEs
and those PEs read the data in a later launch.
:ref:`nvshmem_halo_exchange` builds such a forward pass, and :ref:`nvshmem_in_kernels` describes the built-ins.
This page covers which built-ins carry gradients, how to allocate gradient arrays that other PEs can read,
how Warp synchronizes the backward pass across PEs, and the rules a program must follow to get correct gradients.

Differentiable Built-ins
------------------------

A write to another PE is a copy, so its adjoint is a copy in the opposite direction:
the backward pass reads the gradient of the destination from the PE that holds it.
The adjoint kernel does not repeat the forward write.

* :func:`wp.nvshmem_p() <warp._src.lang.nvshmem_p>` is differentiable:
  its adjoint reads the gradient of the destination element from PE ``pe`` and adds it to the gradient of ``value``.
* :func:`wp.nvshmem_put() <warp._src.lang.nvshmem_put>` is differentiable:
  its adjoint reads the gradient of ``dest`` from PE ``pe`` one element at a time and adds it to ``source.grad``.
* :func:`wp.nvshmem_put_signal() <warp._src.lang.nvshmem_put_signal>` is differentiable:
  its adjoint is that of ``wp.nvshmem_put()``, and the signal update carries no gradient.
* :func:`wp.nvshmem_g() <warp._src.lang.nvshmem_g>` and :func:`wp.nvshmem_get() <warp._src.lang.nvshmem_get>`
  are not differentiable: no gradient flows back to the remote source,
  and Warp warns when the adjoint of a kernel that calls them is launched.
  Restructure such an exchange so that the PE that owns the data pushes it
  with ``wp.nvshmem_p()`` or ``wp.nvshmem_put()``.
* The other NVSHMEM built-ins carry no gradient.
  The signal, wait, quiet, fence and barrier calls of the forward pass are not executed again in the backward pass,
  which Warp synchronizes between launches instead, as described in :ref:`nvshmem_autodiff_barriers`.

Gradient Arrays in Symmetric Memory
-----------------------------------

The adjoint of a push reads the gradient of the destination on the remote PE at the address of the local gradient array,
and NVSHMEM translates addresses between PEs only within symmetric memory.
An array that a kernel writes on another PE therefore needs a gradient array in symmetric memory,
allocated with nvshmem4py in the same order on every PE, like the array itself
(see :ref:`nvshmem_symmetric_memory`).
``requires_grad=True`` does not provide one, because Warp allocates that gradient array with its own allocator.
Allocate the gradient array yourself and assign it to :py:attr:`array.grad`,
as this helper from the worked example at the end of this page does:

.. literalinclude:: ../../../warp/examples/distributed/example_jacobi_nvshmem_grad.py
    :language: python
    :pyobject: symmetric_array

Freeing symmetric memory is collective as well:
once the backward pass is complete, synchronize the device and call ``nvshmem.free()`` for every buffer on every PE.

Warp checks the gradient arrays when it launches the adjoint of a kernel that calls ``wp.nvshmem_p()``,
``wp.nvshmem_put()``, ``wp.nvshmem_put_signal()`` or ``wp.nvshmem_g()``.
For every array argument in symmetric memory, the gradient array that the launch uses for it, if any,
must be in symmetric memory too; otherwise, the launch raises a ``RuntimeError``.
That gradient array is the adjoint array passed for the argument, or else the argument's ``grad``.
The check applies to every symmetric array argument of such a kernel,
including arrays that the kernel only reads and writes locally.

.. _nvshmem_autodiff_barriers:

Synchronizing the Backward Pass
-------------------------------

.. figure:: /img/distributed/backward_pass-light.svg
    :class: only-light
    :alt: Left, the forward pass: on each of two PEs, step k, a push of the boundary row, and step k+1, with the push going from PE 0 to PE 1. Right, the backward pass: on each PE, the adjoint of step k+1, a barrier, the pull that is the adjoint of the push, which reads the gradient from PE 1 into PE 0, the adjoint of step k, and a final barrier.
    :width: 100%

    The backward pass across PEs

.. figure:: /img/distributed/backward_pass-dark.svg
    :class: only-dark
    :alt: Left, the forward pass: on each of two PEs, step k, a push of the boundary row, and step k+1, with the push going from PE 0 to PE 1. Right, the backward pass: on each PE, the adjoint of step k+1, a barrier, the pull that is the adjoint of the push, which reads the gradient from PE 1 into PE 0, the adjoint of step k, and a final barrier.
    :width: 100%

    The backward pass across PEs

On one GPU, stream order guarantees that an adjoint launch sees the gradients that earlier adjoint launches accumulated.
Across PEs it does not: the adjoint of a push on one PE reads a gradient
that the adjoint launches of another PE may still be accumulating.
Warp therefore enqueues a barrier across all PEs on the launch's stream before every adjoint launch of a kernel
that calls ``wp.nvshmem_p()``, ``wp.nvshmem_put()``, ``wp.nvshmem_put_signal()`` or ``wp.nvshmem_g()``,
directly or in a function it calls.
``wp.nvshmem_g()`` is included because the adjoint kernel calls it again to recompute the value it read.
The barrier is a one-thread kernel that calls ``wp.nvshmem_barrier_all()``, so it does not block the host.
Warp enqueues it for adjoint launches from :meth:`Tape.backward() <warp.Tape.backward>`
and from ``wp.launch(..., adjoint=True)`` alike, including launches whose ``dim`` is zero on some PEs.
If a backward pass enqueued any of these barriers,
:meth:`Tape.backward() <warp.Tape.backward>` enqueues one more after its last adjoint launch,
so that no PE zeroes, reseeds or frees gradient arrays that the last adjoint launch of another PE still reads.
Each barrier adds a small kernel launch and a synchronization of all PEs to the backward pass.

Set :attr:`wp.config.nvshmem_adjoint_barrier <warp.config.nvshmem_adjoint_barrier>` to ``False``
to turn off both kinds of barriers, for example when the PEs do not run the same sequence of adjoint launches
and the program synchronizes the backward pass itself.
Warp reads the setting at each adjoint launch.

Warp loads the barrier kernel on a device the first time an adjoint launch on that device needs it,
and it cannot do so while the launch stream is being captured into a CUDA graph.
Run one backward pass before capturing a backward pass into a CUDA graph,
which most graph-capture code already does as a warm-up;
otherwise, the adjoint launch raises a ``RuntimeError``.

Usage Rules
-----------

The gradients are correct when the program follows these rules:

* Data that a PE writes to another PE in one launch is read only in later launches.
  Reading it in the same launch is not supported; see :ref:`nvshmem_autodiff_same_launch`.
* Each element is written once: use one array per time step, as differentiable single-GPU simulations do,
  instead of overwriting an array.
* An array that is the destination of ``wp.nvshmem_p()``, ``wp.nvshmem_put()`` or ``wp.nvshmem_put_signal()``
  has a gradient array in symmetric memory, allocated in the same order on every PE.
* Every PE runs the same sequence of adjoint launches of the kernels that Warp synchronizes.
  A PE whose part of a launch is empty still launches the kernel, with a ``dim`` of zero,
  so that it takes part in the launch's barrier.
  Otherwise, the program hangs in a barrier, unless it turns the barriers off and synchronizes the backward pass itself.
* No PE allocates or frees symmetric memory from the host while a backward pass is still running on its stream.
  NVSHMEM does not allow a host-side collective, such as an allocation, to run concurrently with a barrier on a stream,
  so call :func:`wp.synchronize_device() <warp.synchronize_device>` first.
* A program that launches adjoint kernels with ``wp.launch(..., adjoint=True)``
  instead of using a :class:`wp.Tape <warp.Tape>` enqueues a barrier after its last adjoint launch itself,
  for example by launching a kernel that calls ``wp.nvshmem_barrier_all()`` with ``dim=1``.
* Differentiable kernels call the NVSHMEM built-ins directly in the kernel body,
  not in a :func:`@wp.func <warp.func>` helper.
  The adjoint kernel calls user functions again to recompute their results, as described in :ref:`dynamic_loops`,
  so NVSHMEM calls inside a helper run a second time in the backward pass.
  For example, a ``wp.nvshmem_signal_op()`` with ``wp.NVSHMEM_SIGNAL_ADD`` inside a helper is applied twice.
  Warp also does not see NVSHMEM built-ins called inside a custom gradient or replay function
  (:func:`@wp.func_grad <warp.func_grad>`, :func:`@wp.func_replay <warp.func_replay>`)
  or through :func:`wp.grad() <warp.grad>`,
  so they add neither barriers nor diagnostics to the kernel's adjoint launches.

.. _nvshmem_autodiff_same_launch:

Dependencies Within One Launch
------------------------------

A kernel in which threads wait for data that another PE writes in the same launch cannot be differentiated:

.. code-block:: python

    @wp.kernel
    def fused_step(u: wp.array[wp.float32], sig: wp.array[wp.uint64], nx: int, n_rows: int, pe: int, step: int):
        tid = wp.tid()
        ...
        if iy == n_rows:
            wp.nvshmem_put_signal(u, ..., sig, 0, wp.uint64(step), wp.NVSHMEM_SIGNAL_SET, pe)  # write to neighbor
        wp.nvshmem_signal_wait_until(sig, 0, wp.NVSHMEM_CMP_EQ, wp.uint64(step))  # wait for neighbor
        v = ... u[0 * nx + ix] ...  # read what neighbor wrote

In the backward pass, the gradient of the values read from the neighbor is accumulated in the same adjoint launch
in which the neighbor's adjoint reads it, and no barrier between launches can order the two.
Warp warns when the adjoint of a kernel that writes to another PE before waiting on a signal is launched.
The forward pass of such a kernel also depends on all thread blocks of all PEs running at the same time.

Move the wait to the start of the next launch instead, so that each launch reads only data that earlier launches wrote.
The push moves to a launch of its own after the step, in which one thread pushes the whole boundary row
with a single ``wp.nvshmem_put_signal()``:

.. code-block:: python

    @wp.kernel
    def pipelined_step(u: wp.array[wp.float32], u_new: wp.array[wp.float32], sig: wp.array[wp.uint64], ...):
        tid = wp.tid()
        ...
        if iy == 1:
            wp.nvshmem_signal_wait_until(sig, 0, wp.NVSHMEM_CMP_GE, wp.uint64(step))  # wait for the previous launch
        v = ... u[0 * nx + ix] ...  # read what the neighbor wrote in the previous launch
        u_new[iy * nx + ix] = v


    @wp.kernel
    def push_row(u_new: wp.array[wp.float32], sig: wp.array[wp.uint64], nx: int, n_rows: int, pe: int, step: int):
        # The last owned row goes into the neighbor's ghost row, whose signal flag is set after the row has arrived.
        wp.nvshmem_put_signal(
            u_new[0:nx], u_new[n_rows * nx : (n_rows + 1) * nx], sig, 0, wp.uint64(step + 1), wp.NVSHMEM_SIGNAL_SET, pe
        )
        wp.nvshmem_quiet()  # complete the push, so that the push of the next step cannot overtake it


    for step in range(n_steps):
        wp.launch(pipelined_step, dim=n_rows * (nx - 2), inputs=[u[step], u[step + 1], sig, ...])
        wp.launch(push_row, dim=1, inputs=[u[step + 1], sig, nx, n_rows, pe, step])

:ref:`nvshmem_halo_signals` explains why the signal flag is set once per row and why the push ends with a quiet.
The signals and waits need no changes for the backward pass, which Warp synchronizes with its own barriers.

Diagnostics
-----------

Warp reports the following cases when it launches an adjoint kernel,
never when it compiles a kernel or launches one forward, so programs that only run forward passes see none of them.

.. list-table::
    :header-rows: 1
    :widths: 40 60

    * - Situation
      - Diagnostic
    * - A kernel calls ``wp.nvshmem_g()`` or ``wp.nvshmem_get()``.
      - Warning, once per kernel:
        ``gradients do not flow through wp.nvshmem_g() or wp.nvshmem_get() in kernel "..."``
    * - A kernel writes to another PE before it waits on a signal.
      - Warning, once per kernel:
        ``kernel "..." writes to a remote PE before waiting on a signal``
    * - An array argument of a kernel that calls ``wp.nvshmem_p()``, ``wp.nvshmem_put()``,
        ``wp.nvshmem_put_signal()`` or ``wp.nvshmem_g()`` is in symmetric memory and its gradient array is not.
      - ``RuntimeError``:
        ``Warp: argument "..." of kernel "..." is in NVSHMEM symmetric memory but its gradient array is not``
    * - The first adjoint launch on a device that needs a barrier happens while its stream is being captured.
      - ``RuntimeError``:
        ``Warp: the NVSHMEM adjoint barrier kernel is not loaded yet and the stream is capturing``

Worked Example
--------------

`example_jacobi_nvshmem_grad.py
<https://github.com/NVIDIA/warp/blob/main/warp/examples/distributed/example_jacobi_nvshmem_grad.py>`__
puts these pieces together on two GPUs.
It records a fixed number of Jacobi steps on a tape, with one symmetric array and gradient array per step,
and its backward pass computes the gradient of the sum of squares of the final field's interior, over all PEs,
with respect to the initial field.
Each PE prints its share of that loss and the norm of the gradient with respect to its own rows of the initial field.
Those rows also feed the neighbors' ghost rows, so their gradient includes the paths through the neighbors.

Run the example with one process per GPU:

.. code-block:: sh

    mpirun -np 2 uv run --with mpi4py --with nvshmem4py-cu13 --with "nvidia-nvshmem-cu13>=3.9" \
        python -m warp.examples.distributed.example_jacobi_nvshmem_grad
