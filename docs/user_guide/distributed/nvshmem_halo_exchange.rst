.. _nvshmem_halo_exchange:

Halo Exchange with NVSHMEM
==========================

.. currentmodule:: warp

This guide builds a distributed Jacobi solver for the Laplace equation in stages,
from the simplest correct halo exchange to one that overlaps communication with computation,
keeps the host out of the step loop, and replays its steps from a CUDA graph.
It is the solver that NVIDIA's `multi-GPU programming models
<https://github.com/NVIDIA/multi-gpu-programming-models>`__ repository implements with MPI, NCCL and NVSHMEM,
with row slabs, periodic neighbors, and fixed values in the left and right columns.

The first stage is
`example_jacobi_nvshmem_device.py
<https://github.com/NVIDIA/warp/blob/main/warp/examples/distributed/example_jacobi_nvshmem_device.py>`__,
and the last is
`example_jacobi_nvshmem_overlap.py
<https://github.com/NVIDIA/warp/blob/main/warp/examples/distributed/example_jacobi_nvshmem_overlap.py>`__.
Only these two ship as examples.
The four stages in between are included from
`aux_nvshmem_halo_stages.py
<https://github.com/NVIDIA/warp/blob/main/warp/tests/distributed/aux_nvshmem_halo_stages.py>`__,
a program that the test suite runs on two PEs and checks against the same steps on one GPU.
The setup that all stages share appears once, in :ref:`nvshmem_halo_decomposition`.
Each later stage shows its kernels and its step loop, and says what changed from the stage before and why.

:ref:`nvshmem_in_kernels` is the reference for the built-ins this guide uses,
including the ordering and synchronization rules that the stages rely on.

.. _nvshmem_halo_one_sided:

From Two-Sided to One-Sided
---------------------------

In MPI, moving a halo row takes a call on each side.
The receiver posts a receive that names the buffer, the sender posts a send, and both wait for completion;
``MPI_Sendrecv`` combines the two for an exchange.
MPI writes the row into the receiver's buffer only after the receiver has posted the receive,
so the receiver decides when its buffer may be overwritten.

With NVSHMEM, the sending PE writes the row into the receiver's array itself, and the receiver posts nothing.
Every PE allocates the array at the same offset in its symmetric heap (see :ref:`nvshmem_symmetric_memory`),
so the sender names the destination by the array, an index and the receiver's PE index.
The two jobs that the receive did in MPI become the program's.
The receiver needs to know that the row has arrived:
a signal that the put delivers after the data, or a barrier, tells it.
The sender needs to know that the receiver has finished reading the previous contents of the rows it overwrites.
The stages below meet both needs, first with barriers and then with signals.

.. figure:: /img/distributed/two_sided_one_sided-light.svg
    :class: only-light
    :alt: Moving one row from PE 0 to PE 1. With MPI, PE 0 posts Isend and PE 1 posts Irecv, and both then wait. With NVSHMEM, PE 0 calls wp.nvshmem_put_signal while PE 1 waits in wp.nvshmem_signal_wait_until until the signal arrives after the data.
    :width: 100%

    Two-sided and one-sided exchange

.. figure:: /img/distributed/two_sided_one_sided-dark.svg
    :class: only-dark
    :alt: Moving one row from PE 0 to PE 1. With MPI, PE 0 posts Isend and PE 1 posts Irecv, and both then wait. With NVSHMEM, PE 0 calls wp.nvshmem_put_signal while PE 1 waits in wp.nvshmem_signal_wait_until until the signal arrives after the data.
    :width: 100%

    Two-sided and one-sided exchange

The MPI calls of a halo exchange map to the built-ins as follows.
No row needs the receiving PE to post a receive.

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - MPI
      - Warp with NVSHMEM
    * - ``MPI_Sendrecv`` of a halo row
      - A put of the row into the neighbor's ghost row:
        :func:`wp.nvshmem_put_signal() <warp._src.lang.nvshmem_put_signal>` from one thread,
        or :func:`wp.nvshmem_p() <warp._src.lang.nvshmem_p>` from one thread per element.
    * - ``MPI_Wait`` on my sends
      - :func:`wp.nvshmem_quiet() <warp._src.lang.nvshmem_quiet>`:
        my puts are complete and visible at their destinations.
        The source of a put can be reused as soon as the put returns.
    * - ``MPI_Wait`` on my receives
      - :func:`wp.nvshmem_signal_wait_until() <warp._src.lang.nvshmem_signal_wait_until>`
        on the flag that the neighbor's put sets, or a barrier after the neighbor's puts.
    * - ``MPI_Barrier``
      - :func:`wp.nvshmem_barrier_all() <warp._src.lang.nvshmem_barrier_all>` in a kernel launched with ``dim=1``,
        or a barrier on the host.
    * - ``MPI_Allreduce``
      - A local reduction with :func:`wp.atomic_add() <warp._src.lang.atomic_add>`,
        then a put of the partial sum into an array on every PE, between two barriers
        (see :ref:`nvshmem_halo_reduction`).

.. _nvshmem_halo_decomposition:

Decomposition and Allocation
----------------------------

.. figure:: /img/distributed/halo_slabs-light.svg
    :class: only-light
    :alt: Two row slabs, PE 0 above PE 1, each with a ghost row above and below its owned rows. A put copies the last owned row of PE 0 into the top ghost row of PE 1, and another copies the first owned row of PE 1 into the bottom ghost row of PE 0.
    :width: 100%

    Row slabs and ghost rows

.. figure:: /img/distributed/halo_slabs-dark.svg
    :class: only-dark
    :alt: Two row slabs, PE 0 above PE 1, each with a ghost row above and below its owned rows. A put copies the last owned row of PE 0 into the top ghost row of PE 1, and another copies the first owned row of PE 1 into the bottom ghost row of PE 0.
    :width: 100%

    Row slabs and ghost rows

The mesh has ``NY`` rows and ``NX`` columns.
Its ``NY - 2`` interior rows are split into one slab of consecutive rows per PE, as evenly as possible,
and the rows wrap around, so the PE above PE 0 is the last PE.
Each PE stores its slab between two ghost rows:
row 0 holds a copy of the last row of the PE above, and the row after the slab a copy of the first row of the PE below.
The left and right columns hold fixed boundary values, so a step updates columns 1 to ``NX - 2`` of the owned rows.

A symmetric allocation must have the same size on every PE,
so every PE allocates the largest slab plus two ghost rows, and a PE with a smaller slab leaves its last row unused.
Each field is a one-dimensional array of ``alloc_rows * NX`` elements, which the kernels index as ``iy * nx + ix``.
A PE names a neighbor's ghost row by its row index in the neighbor's slab:
the bottom ghost row of the PE above is row ``slab_rows(top_neighbor) + 1``, after that PE's own rows,
and the top ghost row of the PE below is row 0.

The shared setup initializes NVSHMEM as in :ref:`nvshmem_initialization`,
computes the decomposition, and allocates the fields and the arrays that later stages use:

.. literalinclude:: ../../../warp/examples/distributed/example_jacobi_nvshmem_device.py
    :language: python
    :start-after: # docs: setup begin
    :end-before: # docs: setup end

The values that the kernels receive about the neighbors travel in the ``decomposition`` tuple,
which the function that runs a step loop unpacks first, with
``num_local_rows, top_pe, top_halo_row, bottom_pe, bottom_halo_row = decomposition``.
Warp's module hash currently does not let a kernel parameter shadow a module global of the same name,
so a global whose value differs per PE would give each PE a module of its own, which each PE compiles separately.
The tuple avoids that.

The host writes the initial field into both arrays, ghost rows included,
so that the first step reads correct ghost rows without an exchange.
It then synchronizes its device and waits at a host barrier,
so that no PE puts into an array that its owner is still initializing (see :ref:`nvshmem_symmetric_memory`).
``flags`` and ``partials`` are used from :ref:`nvshmem_halo_signals` and :ref:`nvshmem_halo_reduction` on.

.. _nvshmem_halo_simple:

The Simplest Correct Exchange
-----------------------------

Each thread of the first stage's stencil kernel updates one cell of the owned rows.
The threads of the first and last owned rows also write their new value into the neighbor's ghost row
with ``wp.nvshmem_p()``, one element per thread, as soon as they have computed it:

.. literalinclude:: ../../../warp/examples/distributed/example_jacobi_nvshmem_device.py
    :language: python
    :start-after: # docs: stage-3-kernel begin
    :end-before: # docs: stage-3-kernel end

The puts go into ``a_new`` on the neighbor.
The neighbor writes that array in the same step, but only its owned rows, and it reads ghost rows only from ``a``,
so the puts can arrive at any time during the step.
After the step, the two arrays swap roles, and the next step reads the rows that arrived.

Before the kernel ends, each thread that put an element calls ``wp.nvshmem_quiet()``.
A kernel's completion does not complete the puts that its threads issued,
and a host barrier completes only operations that the host issued (see :ref:`nvshmem_ordering`).
The quiet completes each thread's put, so once the host has waited for its device, this PE's puts have arrived.
The host then waits for every PE at a barrier after each step:

.. literalinclude:: ../../../warp/examples/distributed/example_jacobi_nvshmem_device.py
    :language: python
    :start-after: # docs: stage-3-loop begin
    :end-before: # docs: stage-3-loop end
    :dedent: 4

No PE starts step ``k + 1`` before every PE has finished step ``k`` and its puts have arrived.
That covers both directions of the exchange:
the rows of step ``k`` have arrived before any PE reads them,
and no PE puts the rows of step ``k + 1`` into an array whose ghost rows a neighbor still reads in step ``k``.

This form is correct on any transport and makes a good first version.
It has two costs.
The host is in the loop: it waits for its device at every step, and the device has no work queued while the host waits.
And every element is a message of its own,
which is cheap between GPUs that access each other's memory and expensive across nodes (see :ref:`nvshmem_multi_node`).

.. _nvshmem_halo_device_barrier:

Taking the Host out of the Loop
-------------------------------

The host leaves the loop when the barrier moves onto the stream.
A kernel launched with ``dim=1`` calls ``wp.nvshmem_barrier_all()`` from its only thread
(see :ref:`nvshmem_synchronization`):

.. literalinclude:: ../../../warp/tests/distributed/aux_nvshmem_halo_stages.py
    :language: python
    :start-after: # docs: stage-4-kernels begin
    :end-before: # docs: stage-4-kernels end

The stencil kernel stays as it was, and a step becomes two launches with no host synchronization:

.. literalinclude:: ../../../warp/tests/distributed/aux_nvshmem_halo_stages.py
    :language: python
    :start-after: # docs: stage-4-loop begin
    :end-before: # docs: stage-4-loop end
    :dedent: 4

The barrier returns on a PE when every PE has reached it and the operations that the PEs issued before it are complete
(see :ref:`nvshmem_ordering`),
so it orders the steps as the host barrier did,
and it would complete the stencil kernel's puts without their quiet,
because an earlier launch on the same stream issued them.
The kernel keeps its quiet because this stage reuses the first stage's kernel unchanged.
The host now only enqueues work: it runs ahead of the device, and the stream runs the steps back to back.

Because the loop makes no host synchronization, it can also be captured into a CUDA graph (see :ref:`graphs`)
and replayed with one call, which saves the host the cost of launching its kernels one at a time.
Warp loads the kernels' module and registers it with NVSHMEM on the first launch,
which cannot happen during a capture (see :ref:`nvshmem_first_launch`),
so run the first step eagerly before capturing the rest,
or capture with ``wp.ScopedCapture(force_module_load=True)``, which loads the module before the capture begins.
:ref:`nvshmem_halo_together` shows a captured loop.

The loop has no convergence check.
Reading the norm of the update on the host after every step would put the host back in the loop
with a device-to-host copy per step;
:ref:`nvshmem_halo_reduction` computes it every ``CHECK_EVERY`` steps instead.

What remains is the barrier itself.
Every PE waits for the slowest PE at every step, although it needs only the rows of its two neighbors,
and every row still travels as one message per element.

.. _nvshmem_halo_signals:

Signals Instead of Barriers
---------------------------

A signal lets a PE wait for exactly the rows it needs.
``wp.nvshmem_put_signal()`` copies a row and then updates a signal word on the destination PE,
which arrives after the row (see :ref:`nvshmem_ordering`),
and the destination waits on that word with ``wp.nvshmem_signal_wait_until()``.
Each PE has two signal words in the symmetric array ``flags``:
``flags[0]`` guards its top ghost row and ``flags[1]`` its bottom one.
Each holds the number of the step that reads the row that last arrived.

The stencil kernel, ``jacobi_rows``, no longer puts.
The threads that read a ghost row wait until the flag that guards it reaches the current step.
After the step, ``push_rows`` copies each boundary row as a whole into the neighbor's ghost row
and sets the neighbor's flag to the step that reads the row,
with :data:`wp.NVSHMEM_SIGNAL_SET <warp.NVSHMEM_SIGNAL_SET>`:

.. literalinclude:: ../../../warp/tests/distributed/aux_nvshmem_halo_stages.py
    :language: python
    :start-after: # docs: stage-5-kernels begin
    :end-before: # docs: stage-5-kernels end

One thread copies each row, so the push is a launch with ``dim=2`` (see :ref:`nvshmem_builtins`).
It is a launch of its own because a row is complete
only when every thread of the stencil launch has written its element,
and a launch that follows on the same stream starts only after that.
No step waits at a barrier:

.. literalinclude:: ../../../warp/tests/distributed/aux_nvshmem_halo_stages.py
    :language: python
    :start-after: # docs: stage-5-loop begin
    :end-before: # docs: stage-5-loop end
    :dedent: 4

Four details of the kernels matter:

* **One flag per row.**
  If every thread of a boundary row put its element and set the flag,
  the first element to arrive would satisfy the wait while the rest of the row was still in flight,
  and the neighbor would read a partly updated row.
  One ``wp.nvshmem_put_signal()`` per row sets the flag once, after the whole row has arrived.
  The row also travels as one transfer, which suits network transports (see :ref:`nvshmem_multi_node`).
* **Only the threads that read a ghost row wait.**
  The other threads read only rows that this PE wrote in the previous step, so they have nothing to wait for,
  and they update their cells while the first and last rows wait.
* **The wait compares with** :data:`wp.NVSHMEM_CMP_GE <warp.NVSHMEM_CMP_GE>`.
  A neighbor can run up to one step ahead.
  By the time this PE's step ``k`` waits, the neighbor may have finished its own step ``k``
  and set the flag to ``k + 1`` with the push after it.
  A wait for equality with ``k`` would then never return.
  The push that sets the flag to ``k + 1`` starts only after the push that set it to ``k`` has completed,
  so any value of ``k`` or more means that the row for step ``k`` has arrived.
  The flags start at zero, and the host wrote the ghost rows that step 0 reads, so the waits of step 0 pass at once.
* **The push ends with a quiet.**
  Without it, the push after the next step could overtake this one:
  its signal could arrive first and satisfy a wait for this push's row,
  and this push's signal, arriving later, would set the flag back.

The flags also keep a push from overwriting a ghost row that the neighbor still reads.
The push after step ``k`` writes into the array whose ghost rows the neighbor read in step ``k - 1``.
This PE's step ``k`` cannot finish before the neighbor's push after step ``k - 1`` has arrived,
and the neighbor ran that push after its step ``k - 1`` had finished,
so the push after this PE's step ``k`` starts only once the neighbor is done with those ghost rows.

.. note::

    NVSHMEM's notes on `CUDA interactions <https://docs.nvidia.com/nvshmem/api/latest/cuda-interactions.html>`__
    warn that a kernel that waits for a signal can deadlock
    when the signal comes from a kernel that runs at the same time:
    a waiting thread holds its GPU's resources,
    and CUDA does not guarantee that the threads that would set the signal get to run.
    NVSHMEM's collective launch, which Warp's launches do not use, exists for such kernels.
    The stages on this page avoid the situation.
    Every wait is for a row that the neighbor pushes in a launch of its own after its previous step,
    so the push never waits for the kernel that waits for it,
    and with one PE per GPU, the push runs on a different GPU from the wait.
    :ref:`nvshmem_autodiff_same_launch` shows the pattern to avoid,
    in which threads wait for data that another PE writes in the same launch.

.. _nvshmem_halo_overlap:

Overlapping Communication and Computation
-----------------------------------------

To hide the push behind computation, a step needs work that does not depend on the push:
the interior rows, which read no ghost row.
This stage splits the stencil into a kernel for the two boundary rows and a kernel for the interior rows.
On one stream that is not enough.
A put returns only after its data has been copied out, and the push ends with a quiet,
so a launch of the interior rows behind the push on the same stream would start only after the rows were delivered.
Overlap needs a second stream.

.. figure:: /img/distributed/jacobi_step_timeline-light.svg
    :class: only-light
    :alt: Top, on two streams per PE: the compute stream waits for the flags and runs the boundary rows, then the interior rows, while the communication stream pushes the boundary rows with a signal and a quiet; events fork the push off and join it back, and each PE's push releases the other PE's next wait. Bottom, on one stream per PE: a stencil with element puts, then barrier_all, in which each PE waits for the slower one.
    :width: 100%

    Two streams versus a barrier per step

.. figure:: /img/distributed/jacobi_step_timeline-dark.svg
    :class: only-dark
    :alt: Top, on two streams per PE: the compute stream waits for the flags and runs the boundary rows, then the interior rows, while the communication stream pushes the boundary rows with a signal and a quiet; events fork the push off and join it back, and each PE's push releases the other PE's next wait. Bottom, on one stream per PE: a stencil with element puts, then barrier_all, in which each PE waits for the slower one.
    :width: 100%

    Two streams versus a barrier per step

The compute stream runs the boundary rows and records an event.
The communication stream, ``comm_stream``, waits for that event and runs the push,
while the compute stream goes on with the interior rows.
The compute stream then waits for an event recorded after the push, and the step ends.
The threads of the boundary kernel wait on the flags inside the kernel, as in :ref:`nvshmem_halo_signals`;
the figure draws the wait as a box of its own in front of the boundary rows.
The push is the ``push_rows`` kernel of that section; the two stencil kernels are new:

.. literalinclude:: ../../../warp/tests/distributed/aux_nvshmem_halo_stages.py
    :language: python
    :start-after: # docs: stage-6-kernels begin
    :end-before: # docs: stage-6-kernels end

The loop creates the second stream and the events before its first step and orders each step with them
(see :ref:`cuda_events`):

.. literalinclude:: ../../../warp/tests/distributed/aux_nvshmem_halo_stages.py
    :language: python
    :start-after: # docs: stage-6-loop begin
    :end-before: # docs: stage-6-loop end
    :dedent: 4

The wait for ``push_done`` at the end of each step joins ``comm_stream`` back into the compute stream,
for two reasons.
First, the push reads rows 1 and ``n_rows`` of ``a_new``, which the boundary kernel of the step after next overwrites,
and the join orders the two on this PE's own streams.
The flags happen to order them as well, through the chain of waits between the PEs:
that boundary kernel waits for a row that the neighbor pushed only after this push's row had arrived.
The join keeps the reasoning local, so that the program's correctness does not rest on that chain.
Second, a CUDA graph capture must join every stream that it forked back into the capture stream before the capture ends
(see :ref:`synchronization_guidance`),
which the join of the last step does, so the loop can be captured as it is.
Replaying a captured loop also needs the step number in device memory, which :ref:`nvshmem_halo_together` shows.

The push and the interior rows now run at the same time,
so the step no longer waits for the push as long as the interior rows take longer than it.
In exchange, a step has one more launch than in :ref:`nvshmem_halo_signals`, and it records and waits for two events.
For small slabs, where the push takes less time than a launch, there is nothing to hide,
and the one-stream form of that section is the simpler choice.

The stencil kernel of :ref:`nvshmem_halo_simple` overlaps in its own way:
its boundary threads put elements while other threads compute.
It cannot publish a row with one signal, though, and element-sized messages are the wrong unit across nodes,
so the stages from :ref:`nvshmem_halo_signals` on push whole rows.

.. _nvshmem_halo_reduction:

The Global Reduction
--------------------

A solver checks convergence with the norm of the update,
which needs a sum over all PEs: an ``MPI_Allreduce`` in an MPI program.
This stage computes it on the device every ``CHECK_EVERY`` steps.
Each PE sums its squared update into a one-element array with ``wp.atomic_add()``,
then puts that partial sum with ``wp.nvshmem_p()`` into slot ``my_pe`` of the symmetric array ``partials`` on every PE,
itself included.
Every PE then holds all partial sums, and its host adds them up:

.. literalinclude:: ../../../warp/tests/distributed/aux_nvshmem_halo_stages.py
    :language: python
    :start-after: # docs: stage-7-kernels begin
    :end-before: # docs: stage-7-kernels end

``exchange_partials`` runs in one thread and completes its puts with ``wp.nvshmem_quiet()``.
The PEs exchange partial sums instead of adding into one remote value because NVSHMEM has no floating-point atomic add:
its `atomic add <https://docs.nvidia.com/nvshmem/api/latest/gen/api/amo.html>`__ takes integer types only.

The loop is the one of :ref:`nvshmem_halo_overlap`, with the check after every ``CHECK_EVERY``-th step:

.. literalinclude:: ../../../warp/tests/distributed/aux_nvshmem_halo_stages.py
    :language: python
    :start-after: # docs: stage-7-loop begin
    :end-before: # docs: stage-7-loop end
    :dedent: 4

Two barrier kernels surround the exchange:

* **The barrier before the exchange** keeps a fast PE from putting its next sums into ``partials``
  while a slow PE has yet to read the previous ones.
  Each host reads ``partials`` before it enqueues the next steps,
  so every PE reaches this barrier only after its host has read the previous sums.
  The flags keep only neighbors within one step of each other, and a PE ``d`` hops away can be ``d`` steps ahead,
  so with enough PEs, a PE far from a slow one could otherwise reach its next exchange
  before the slow PE has read the previous sums.
* **The barrier after the exchange** completes the sums.
  When it returns, the puts of every PE into this PE's ``partials`` are complete,
  and ``partials.numpy()``, which copies on the same stream, reads all of them.

The host reads ``partials`` at every check, so it waits for its device once every ``CHECK_EVERY`` steps
and only enqueues work in between.

On the host, nvshmem4py's ``nvshmem.core.reduce()`` reduces symmetric arrays on a stream,
but it takes CuPy arrays or Torch tensors that nvshmem4py allocated, not Warp arrays;
the `nvshmem4py collectives reference
<https://docs.nvidia.com/nvshmem/api/latest/api/language_bindings/python/collectives.html>`__ describes it.
A program that uses MPI anyway can also synchronize its device, read its partial sum and call ``mpi_comm.allreduce()``,
as ``example_jacobi_nvshmem_device.py`` does for its final norm.
The device form keeps the reduction in stream order, where it can be captured with the steps.

.. _nvshmem_halo_together:

Putting It Together
-------------------

`example_jacobi_nvshmem_overlap.py
<https://github.com/NVIDIA/warp/blob/main/warp/examples/distributed/example_jacobi_nvshmem_overlap.py>`__
combines the stages of :ref:`nvshmem_halo_signals`, :ref:`nvshmem_halo_overlap` and :ref:`nvshmem_halo_reduction`,
and it captures blocks of steps into a CUDA graph.
Its kernels are those of the stages, except that the step number comes from a one-element device array,
``step_counter``, instead of a kernel argument, and a new kernel, ``advance_step``, increments it.

.. literalinclude:: ../../../warp/examples/distributed/example_jacobi_nvshmem_overlap.py
    :language: python
    :start-after: # docs: kernels begin
    :end-before: # docs: kernels end

A captured graph replays its launches with the arguments they had when they were captured.
A step number passed as an argument would keep its captured value,
and every replay would wait for and signal the same steps.
Reading the step from device memory lets one graph run every block:
``boundary_rows`` waits until the flag reaches ``wp.uint64(step_counter[0])``,
``push_rows`` signals ``step_counter[0] + 1``,
and ``advance_step``, the last launch of each step, increments the counter.

``run_block()`` issues one block: ``CHECK_EVERY`` steps, then the residual and the exchange of the partial sums.
The join before ``advance_step`` now also orders the push's read of the counter before the increment,
which nothing else orders.
``CHECK_EVERY`` is even, so after a block ``a`` and ``a_new`` have their original roles again,
and one graph serves every block.

.. literalinclude:: ../../../warp/examples/distributed/example_jacobi_nvshmem_overlap.py
    :language: python
    :start-after: # docs: step begin
    :end-before: # docs: step end

The first block runs eagerly, through the same ``run_block()``:
its first launches load the module and register it with NVSHMEM, which cannot happen during a capture
(``wp.ScopedCapture(force_module_load=True)`` would load the module before the capture instead).
The capture then records the launches of one block without running them, and each replay runs the next block.
``--iterations`` must therefore be a multiple of ``CHECK_EVERY``.
After every block, the host reads ``partials``, which waits for the block to finish:

.. literalinclude:: ../../../warp/examples/distributed/example_jacobi_nvshmem_overlap.py
    :language: python
    :start-after: # docs: capture begin
    :end-before: # docs: capture end

After installing the packages in :ref:`nvshmem_requirements`,
and mpi4py built against the MPI library that runs the job (see :ref:`nvshmem_initialization`),
run the example with one process per GPU:

.. code-block:: sh

    mpirun -np 2 python -m warp.examples.distributed.example_jacobi_nvshmem_overlap

Checklist
~~~~~~~~~

A halo exchange in Warp kernels follows these rules:

* Allocate symmetric arrays with the same size on every PE, in the same order: the largest slab plus two ghost rows.
* Zero symmetric memory before its first use, then synchronize and wait at a barrier before any PE puts into it.
* Issue each bulk transfer, ``wp.nvshmem_put()``, ``wp.nvshmem_put_signal()`` or ``wp.nvshmem_get()``, from one thread.
* Call ``wp.nvshmem_barrier_all()`` from a kernel launched with ``dim=1``.
* Have every PE call every collective in the same order, the barrier kernels included.
* Read the data pushed in step ``k`` in step ``k + 1`` or later, never in the launch that pushes it.
* Publish each row with one signal, and wait for the step number with ``wp.NVSHMEM_CMP_GE``.
* Call ``wp.nvshmem_quiet()`` after a push that a later push could overtake.
* Call ``wp.nvshmem_quiet()`` in the threads that put before relying on a host barrier.
* Join the communication stream back into the compute stream before the arrays that the push reads are written again,
  and before a capture ends.
* Launch every kernel once before capturing, or capture with ``wp.ScopedCapture(force_module_load=True)``,
  and keep the step number in device memory in a captured loop.
