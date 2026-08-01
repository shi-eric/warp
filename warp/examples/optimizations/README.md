# Runtime Optimization Examples

This corpus compares semantically equivalent Warp programs whose repeated
runtime behavior differs. It covers steady-state execution after normal
initialization. It does not cover compilation, JIT caching, module hashing,
build time, or cold-start optimization.

Results are evidence for a specific device and workload, not universal speed
claims. Measure the complete repeated path in your application before adopting
a transformation.

## Example cards

Each card contains:

- `before.py`: a correct baseline with an intentional runtime cost;
- `after.py`: the candidate runtime transformation;
- `benchmark.py`: `build_case(device, workload)`, which constructs matching
  variants for the shared harness;
- `test_correctness.py`: deterministic checks for declared outputs;
- `explanation.md`: the mechanism, preconditions, and contraindications;
- `manifest.json`: searchable metadata, workload defaults, tolerances,
  structured scoped claims, status, and separate CUDA and CPU impact labels;
  and
- `evidence.json`: append-only correctness results, raw paired samples,
  confidence intervals, environment facts, measured contracts, and
  limitations.

The examples are independently authored from synthetic, domain-neutral
specifications using public Warp APIs. They do not derive code, structure,
identifiers, equations, constants, or provenance from private sources. A local
deny-pattern file can add a clean-room review backstop without placing those
patterns in the repository.

## Initial corpus index

The impact labels below apply only to the exact measured device and default
workload shown for each card. `Unverified` means that the corpus makes no
runtime claim for that platform.

### Host/device transfer elimination

#### [`device-resident-spectral-transform`](transfer_elimination/device_resident_spectral_transform)

- **Status:** recommended. CUDA: improved on NVIDIA RTX PRO 6000 Blackwell
  Server Edition; CPU: unverified.
- **Default workload:** `{"batch": 256, "iterations": 20, "seed": 20260730,
  "size": 256}`.
- **Runtime mechanism:** Keep tiled FFT state on CUDA to remove the repeated
  device-to-host transform path, host temporaries, synchronization, and
  host-to-device copy.

### Kernel fusion

#### [`fused-elementwise-pipeline`](kernel_fusion/fused_elementwise_pipeline)

- **Status:** recommended. CUDA: improved on NVIDIA RTX PRO 6000 Blackwell
  Server Edition; CPU: improved on AMD EPYC 9B45.
- **Default workload:** `{"iterations": 50, "seed": 20260729, "size":
  1048576}`.
- **Runtime mechanism:** Perform three compatible pointwise operations between
  the global reads and final write, removing two launches and two full-size
  intermediate round trips.

#### [`expanded-halo-fusion`](kernel_fusion/expanded_halo_fusion)

- **Status:** rejected. CUDA: harmful on NVIDIA RTX PRO 6000 Blackwell Server
  Edition; CPU: unverified.
- **Default workload:** `{"height": 2048, "iterations": 20, "radius": 2,
  "seed": 20260730, "width": 2048}`.
- **Runtime mechanism:** Removing the separable intermediate expands each
  output to a product halo whose redundant loads and arithmetic outweigh the
  saved launch and global-memory round trip.
- **Rejected evidence:** the paired CUDA candidate/baseline 95% confidence
  interval was `[1.5462755904604266, 1.5603224785792487]`, entirely above
  parity, so the predeclared radius-4 follow-up was not run.

### Allocation reuse

#### [`reused-iteration-workspace`](allocation_reuse/reused_iteration_workspace)

- **Status:** recommended. CUDA: improved on NVIDIA RTX PRO 6000 Blackwell
  Server Edition; CPU: neutral on AMD EPYC 9B45.
- **Default workload:** `{"iterations": 100, "seed": 20260730, "size":
  1048576}`.
- **Runtime mechanism:** Reuse fixed scratch storage to remove repeated
  allocator bookkeeping and temporary-array lifetime churn without changing
  either kernel.

### Autodiff strategy

#### [`native-autodiff-rollout`](autodiff/native_autodiff_rollout)

- **Status:** recommended. CUDA: improved on NVIDIA RTX PRO 6000 Blackwell
  Server Edition; CPU: unverified.
- **Default workload:** `{"seed": 20260730, "size": 262144, "steps": 32}`.
- **Runtime mechanism:** Record the complete device-resident rollout on one
  Warp Tape to remove per-step host-framework callbacks and repeated array
  mapping.
- **Optional requirement:** CUDA-enabled PyTorch from the CUDA-matched Warp
  environment is required for the baseline and correctness comparison.

### Memory/runtime tradeoffs

#### [`gradient-safe-intermediate-lifetime`](autodiff/gradient_safe_intermediate_lifetime)

- **Status:** recommended. CUDA: improved on NVIDIA RTX PRO 6000 Blackwell
  Server Edition; CPU: improved on AMD EPYC 9B45.
- **Default workload:** `{"derivative_depends_on_state": false, "seed":
  20260730, "size": 262144, "steps": 64}`.
- **Runtime mechanism:** For the declared constant-derivative recurrence,
  ping-pong forward and manual-adjoint buffers avoid retaining a unique primal
  and gradient array for every step.

#### [`direct-tape-without-checkpointing`](memory_tradeoffs/direct_tape_without_checkpointing)

- **Status:** recommended. CUDA: improved on NVIDIA RTX PRO 6000 Blackwell
  Server Edition; CPU: unverified.
- **Default workload:** `{"seed": 20260730, "segment_length": 8, "size":
  131072, "steps": 64}`.
- **Runtime mechanism:** When the bounded full Tape passes the capacity guard,
  retain it directly to avoid segmented forward recomputation, copies, and
  repeated Tape construction.

### Interoperability

#### [`device-resident-torch-exchange`](interoperability/device_resident_torch_exchange)

- **Status:** recommended. CUDA: improved on NVIDIA RTX PRO 6000 Blackwell
  Server Edition; CPU: unverified.
- **Default workload:** `{"iterations": 20, "seed": 20260730, "size":
  1048576}`.
- **Runtime mechanism:** Reuse one zero-copy Warp view of PyTorch-owned CUDA
  storage to remove per-iteration host staging and cross-framework allocation.
- **Optional requirement:** CUDA-enabled PyTorch from the CUDA-matched Warp
  environment is required.

## Reading runtime evidence

Correctness is checked before any benchmark runs. The harness then warms both
variants outside the timed region and alternates their order in synchronized
paired trials. A CUDA result is improved only when the upper bound of the
paired 95% confidence interval for candidate/baseline runtime is below `1.0`.
A favorable point estimate alone is inconclusive.

Each output records the applied absolute and relative tolerances and the
maximum normalized error
`abs_error / (atol + rtol * abs(reference))`. The harness derives correctness
from finite outputs with normalized error at most `1.0`; an unbounded metric is
stored as JSON `null`.

End-to-end runtime includes work that remains in the repeated application path:
kernel and library launches, required transfers, synchronization, allocation,
and framework crossings. Moving a transfer or synchronization outside the
measurement is valid only when the application can move it outside the
repeated path too. Compilation and one-time initialization remain excluded.

Kernel fusion means performing multiple compatible operations between one
global read and one global write. It can reduce launches and global-memory
traffic, but it is conditional when intermediate values must be observed,
stages require barriers, or resource pressure offsets the gain.

CUDA and CPU impact are labeled independently as `improved`, `neutral`,
`harmful`, or `unverified`. A positive label must be derived from an eligible
structured claim that names its supporting record IDs and exact measured
device and workload. `neutral` additionally requires a confidence interval
inside the card's predeclared equivalence band. `conditional` cards are
supported only for their explicit bounded device or workload region.
`rejected` cards retain incorrect, harmful, or inconclusive attempts so the
same unsafe advice is not repeated.

New evidence records include an immutable measured-contract envelope. It binds
the exact workload and protocol, output tolerances, Warp and device contract,
CPU model and topology where applicable, and SHA-256 hashes of the measured
card and shared runner sources. Legacy records remain visible and valid as
history, but cannot support a current claim. A record also becomes
non-supporting when the current workload, protocol, output contract,
compatibility, Warp version, source hashes, or evidence-age policy no longer
matches its envelope.

## Commands

Run commands from a Warp source checkout or installed environment:

```console
python -m warp.examples.optimizations.run list
python -m warp.examples.optimizations.run check \
  --example EXAMPLE_ID --device cuda:0
python -m warp.examples.optimizations.run benchmark \
  --example EXAMPLE_ID --device cuda:0 \
  --set size=1048576 --pairs 20 --resamples 10000 \
  --output /tmp/runtime-optimization-evidence.json
python -m warp.examples.optimizations.run validate
python -m warp.examples.optimizations.run validate \
  --deny-pattern-file /local/path/deny-patterns.txt
```

`--set` accepts declared workload keys and deterministic JSON-scalar values,
such as integers, floats, `true`, `false`, `null`, or strings. Benchmark output
is always explicit; the runner never selects a checked-in evidence file for
you. Exploratory output is validated for intrinsic integrity without requiring
publication claims. Checked-in `validate` additionally verifies claim support,
classification, and current compatibility before a card can be recommended.

Raw timings and confidence intervals vary with the device, driver, Warp
version, workload shape, iteration count, and surrounding system activity.
Recheck correctness and paired runtime on the environment that matters.
