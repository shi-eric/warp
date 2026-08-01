Runtime Optimization Examples
=============================

Warp programs with the same observable result can have different repeated
end-to-end runtime. The runtime optimization corpus provides paired,
synthetic examples with explicit correctness contracts and device- and
workload-specific evidence. It excludes compilation, JIT caching, module
hashing, build time, and cold-start optimization.

Runtime taxonomy
----------------

The examples organize runtime costs by the boundary or resource they affect:

* host/device transfers and unnecessary host synchronization;
* kernel-launch amortization and compatible kernel fusion;
* repeated allocation and buffer lifetime;
* data layout, work decomposition, and on-chip reuse;
* autodiff execution and memory/runtime tradeoffs; and
* device-native operations and zero-copy interoperability.

Kernel fusion is more than placing code in one function. A useful fusion
performs multiple compatible operations between one global read and one global
write, reducing launches and global-memory round trips. It remains conditional:
required intermediate observations, barriers, divergent work, larger halos, or
register pressure can make separate kernels preferable.

Initial corpus index
--------------------

The impact labels below apply only to each card's exact measured device and
default workload. ``Unverified`` means that the corpus makes no runtime claim
for that platform.

Host/device transfer elimination
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``device-resident-spectral-transform``
   **Status:** recommended. CUDA: improved on NVIDIA RTX PRO 6000 Blackwell
   Server Edition; CPU: unverified.

   **Default workload:** ``{"batch": 256, "iterations": 20, "seed":
   20260730, "size": 256}``.

   **Runtime mechanism:** Keep tiled FFT state on CUDA to remove the repeated
   device-to-host transform path, host temporaries, synchronization, and
   host-to-device copy.

Kernel fusion
^^^^^^^^^^^^^

``fused-elementwise-pipeline``
   **Status:** recommended. CUDA: improved on NVIDIA RTX PRO 6000 Blackwell
   Server Edition; CPU: improved on AMD EPYC 9B45.

   **Default workload:** ``{"iterations": 50, "seed": 20260729, "size":
   1048576}``.

   **Runtime mechanism:** Perform three compatible pointwise operations
   between the global reads and final write, removing two launches and two
   full-size intermediate round trips.

``expanded-halo-fusion``
   **Status:** rejected. CUDA: harmful on NVIDIA RTX PRO 6000 Blackwell Server
   Edition; CPU: unverified.

   **Default workload:** ``{"height": 2048, "iterations": 20, "radius": 2,
   "seed": 20260730, "width": 2048}``.

   **Runtime mechanism:** Removing the separable intermediate expands each
   output to a product halo whose redundant loads and arithmetic outweigh the
   saved launch and global-memory round trip.

   **Rejected evidence:** the paired CUDA candidate/baseline 95% confidence
   interval was ``[1.5462755904604266, 1.5603224785792487]``, entirely above
   parity, so the predeclared radius-4 follow-up was not run.

Allocation reuse
^^^^^^^^^^^^^^^^

``reused-iteration-workspace``
   **Status:** recommended. CUDA: improved on NVIDIA RTX PRO 6000 Blackwell
   Server Edition; CPU: neutral on AMD EPYC 9B45.

   **Default workload:** ``{"iterations": 100, "seed": 20260730, "size":
   1048576}``.

   **Runtime mechanism:** Reuse fixed scratch storage to remove repeated
   allocator bookkeeping and temporary-array lifetime churn without changing
   either kernel.

Autodiff strategy
^^^^^^^^^^^^^^^^^

``native-autodiff-rollout``
   **Status:** recommended. CUDA: improved on NVIDIA RTX PRO 6000 Blackwell
   Server Edition; CPU: unverified.

   **Default workload:** ``{"seed": 20260730, "size": 262144, "steps": 32}``.

   **Runtime mechanism:** Record the complete device-resident rollout on one
   Warp Tape to remove per-step host-framework callbacks and repeated array
   mapping.

   **Optional requirement:** CUDA-enabled PyTorch from the CUDA-matched Warp
   environment is required for the baseline and correctness comparison.

Memory/runtime tradeoffs
^^^^^^^^^^^^^^^^^^^^^^^^

``gradient-safe-intermediate-lifetime``
   **Status:** recommended. CUDA: improved on NVIDIA RTX PRO 6000 Blackwell
   Server Edition; CPU: improved on AMD EPYC 9B45.

   **Default workload:** ``{"derivative_depends_on_state": false, "seed":
   20260730, "size": 262144, "steps": 64}``.

   **Runtime mechanism:** For the declared constant-derivative recurrence,
   ping-pong forward and manual-adjoint buffers avoid retaining a unique
   primal and gradient array for every step.

``direct-tape-without-checkpointing``
   **Status:** recommended. CUDA: improved on NVIDIA RTX PRO 6000 Blackwell
   Server Edition; CPU: unverified.

   **Default workload:** ``{"seed": 20260730, "segment_length": 8, "size":
   131072, "steps": 64}``.

   **Runtime mechanism:** When the bounded full Tape passes the capacity
   guard, retain it directly to avoid segmented forward recomputation, copies,
   and repeated Tape construction.

Interoperability
^^^^^^^^^^^^^^^^

``device-resident-torch-exchange``
   **Status:** recommended. CUDA: improved on NVIDIA RTX PRO 6000 Blackwell
   Server Edition; CPU: unverified.

   **Default workload:** ``{"iterations": 20, "seed": 20260730, "size":
   1048576}``.

   **Runtime mechanism:** Reuse one zero-copy Warp view of PyTorch-owned CUDA
   storage to remove per-iteration host staging and cross-framework
   allocation.

   **Optional requirement:** CUDA-enabled PyTorch from the CUDA-matched Warp
   environment is required.

Evidence gate
-------------

Every benchmark checks all declared outputs first. It then measures the
complete repeated path, including required transfers and synchronization, with
alternating paired trials. Compilation and one-time initialization happen
outside the timed region.

A CUDA result qualifies as improved only when the upper bound of the paired
95% confidence interval for candidate/baseline runtime is below ``1.0``. CUDA
and CPU impact are labeled independently. A published positive claim names its
supporting evidence records and exact device and workload scope; a neutral
claim requires the complete interval to fit inside a predeclared equivalence
band. Conditional and rejected results remain visible, and no result is
generalized beyond its measured scope.

Each new record binds the exact workload, benchmark protocol, output
tolerances, Warp and device compatibility, and hashes of the measured sources
in an immutable contract. The record stores a normalized correctness margin
for every declared output. Legacy or stale records remain auditable history
but cannot support a current claim.

See the `runtime optimization corpus README
<https://github.com/NVIDIA/warp/blob/main/warp/examples/optimizations/README.md>`_
for card anatomy and commands. The `corpus design
<https://github.com/NVIDIA/warp/blob/main/design/runtime-optimization-example-corpus.md>`_
describes the correctness, measurement, classification, and clean-room
contracts.
