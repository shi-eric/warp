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
