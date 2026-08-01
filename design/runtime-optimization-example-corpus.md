# Runtime Optimization Example Corpus

**Status**: Proposed

## Motivation

Warp users can express the same computation through programs with very
different end-to-end runtime behavior. Common performance losses include
unnecessary host/device transfers, host synchronization, repeated allocation,
excessive kernel launches, avoidable global-memory traffic, and execution
strategies that prevent useful work from remaining on the device.

General GPU advice identifies these classes of problems, but it does not tell a
Warp user whether a specific transformation is correct, supported, or faster
for their workload. Individual examples also tend to report a favorable timing
without preserving the environment, workload, correctness tolerance, raw
samples, or an unsuccessful alternative. That makes the result difficult to
reproduce and unsafe to automate.

This feature adds a clean-room corpus of paired Warp programs. Each pair
contains an intentionally inefficient runtime pattern and a semantically
equivalent optimized form. A shared evidence harness verifies correctness and
measures end-to-end steady-state runtime on representative devices. Structured
metadata makes the corpus usable both by people and by a future optimization
skill that can recognize, rank, apply, and validate candidate transformations.

The corpus is educational and empirical. Inclusion does not mean that a
transformation is universally beneficial. Each entry records the conditions
under which it was measured, its device-specific impact, and cases where the
strategy should not be applied.

## Requirements

| ID | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| R1 | Every example must be independently authored from a synthetic, domain-neutral workload | Must | No private application code, identifiers, structure, constants, equations, layouts, comments, or provenance |
| R2 | Every optimization must preserve the declared observable behavior within an explicit tolerance | Must | Correctness is checked before performance |
| R3 | Performance evidence must use paired, synchronized, steady-state end-to-end measurements | Must | Warm-up and compilation occur outside timed regions |
| R4 | A recommended optimization must have a scoped structured claim backed by eligible paired evidence whose 95% confidence interval is entirely below runtime parity on at least one CUDA GPU | Must | The supporting record IDs, exact default workload, and measured device scope are explicit |
| R5 | Every entry must label CUDA and CPU impact independently from eligible referenced evidence | Must | Values are `improved`, `neutral`, `harmful`, or `unverified`; neutral uses a predeclared equivalence band |
| R6 | The corpus must include conditional and rejected transformations | Must | Negative evidence helps prevent unsafe automation |
| R7 | Metadata must describe recognition signals, preconditions, contraindications, compatibility, and evidence | Must | This is the retrieval contract for future tooling |
| R8 | Shared harness code must validate manifests, correctness results, benchmark samples, and environment records | Must | Invalid or incomplete evidence cannot produce a recommendation |
| R9 | Examples must use public Warp APIs | Must | Examples should remain useful to external users |
| R10 | The initial corpus must prioritize strategies with direct empirical support before adding broader GPU advice | Must | Broader entries require their own correctness and benchmark evidence |
| R11 | Timed regions must reflect the runtime path a user would experience after normal initialization | Must | Required transfers, launches, synchronization, allocation, and framework crossings remain timed |
| R12 | Example status and evidence must remain auditable as Warp and hardware evolve | Should | Staleness metadata and periodic revalidation support this |
| R13 | The harness should make evidence generation convenient on both CUDA and CPU devices | Should | CPU results do not replace the CUDA inclusion gate |
| R14 | The directory should be runnable as a focused example suite and discoverable from Warp's example documentation | Should | Exact documentation integration is an implementation detail |

**Non-goals**:

- Optimizing compilation, code generation, JIT cache behavior, build time,
  module hashing, or cold-start latency.
- Claiming that any transformation is universally faster.
- Encoding application-specific algorithms or publishing the provenance of
  observations from non-public code.
- Building the autonomous optimization skill in the same change. This design
  defines the corpus interface that such a skill can consume.
- Replacing application-level profiling with a static checklist.
- Using timing assertions in Warp's standard test suite.
- Introducing checkpointing solely as a precaution. It is considered only when
  measured memory pressure or capacity constraints justify it.

## Terminology

- **Baseline**: The valid but intentionally inefficient program in an example.
- **Candidate**: The semantically equivalent program containing the proposed
  runtime optimization.
- **Example card**: The complete directory for one paired transformation,
  including programs, metadata, tests, explanation, benchmark, and evidence.
- **Evidence run**: One correctness and performance evaluation for a fixed
  workload and environment.
- **Parity**: An optimized/baseline paired runtime ratio of `1.0`.
- **Recommended**: Correct and measurably faster for the declared scope.
- **Conditional**: Correct and measurably faster only under explicit
  preconditions or for a bounded workload/device region.
- **Rejected**: Incorrect, slower, inconclusive, or impractical for the tested
  scope. Rejected entries remain useful counterexamples.

## Design

### Approach

The feature uses structured example cards backed by a shared evidence harness.
Human-readable explanations answer why a transformation can improve runtime,
while machine-readable manifests answer when it may apply. Evidence files
record what was actually measured rather than turning a general heuristic into
an unsupported recommendation.

The proposed layout is:

```text
warp/examples/optimizations/
├── README.md
├── schema/
│   └── example.schema.json
├── harness/
│   ├── benchmark.py
│   ├── correctness.py
│   ├── environment.py
│   └── statistics.py
└── <category>/
    └── <example-name>/
        ├── manifest.json
        ├── before.py
        ├── after.py
        ├── test_correctness.py
        ├── benchmark.py
        ├── explanation.md
        └── evidence.json
```

The baseline and candidate expose matching setup, execution, synchronization,
and result-extraction interfaces. The harness owns trial ordering,
synchronization, sample collection, statistical analysis, and environment
capture so individual examples do not subtly redefine what counts as a speedup.

### Clean-Room Construction

Every example is authored from a short, generic behavior specification rather
than by editing or anonymizing application code. Permitted inputs are:

- Public Warp APIs, documentation, and examples.
- Public specifications for interoperating libraries.
- General GPU programming knowledge.
- Abstract performance smells, such as a device array being copied to a host
  library and back during an iterative calculation.

The following material is prohibited:

- Private application names or provenance.
- Application identifiers, file structure, control-flow structure, equations,
  constants, data layouts, comments, or test fixtures.
- Cosmetic rewrites of application source.
- Example descriptions specific enough to reconstruct an originating
  application.

Each manifest contains a clean-room declaration. Review checks also scan for a
denylist of known private identifiers. The denylist is a backstop, not a
substitute for independent authorship.

Synthetic workloads use neutral concepts such as vector transforms, tiled
matrix operations, generic stencil updates, or spectral filtering. Sizes and
parameters are selected for benchmark coverage rather than inherited from an
application.

### Example Card Contract

`manifest.json` is the stable retrieval interface for people and future tools.
JSON avoids adding a YAML runtime dependency to the example harness. Its schema
includes:

```json
{
  "schema_version": 1,
  "id": "device-resident-spectral-transform",
  "title": "Keep a spectral transform on the device",
  "category": "host-device-transfer-elimination",
  "status": "recommended",
  "summary": "Replace an iterative host round trip with a device-native transform.",
  "recognition": {
    "signals": [
      "device_to_host_copy_inside_iteration",
      "host_transform_between_device_kernels",
      "host_to_device_copy_inside_iteration"
    ]
  },
  "applicability": {
    "preconditions": ["supported_transform_shape", "supported_element_type", "reusable_device_workspace"],
    "contraindications": [
      "transform_is_executed_only_on_cpu",
      "unsupported_shape_or_precision",
      "transfer_is_outside_repeated_runtime_path"
    ]
  },
  "semantics": {
    "observable_outputs": ["result_array"],
    "tolerance": {"relative": 1.0e-5, "absolute": 1.0e-6}
  },
  "impact": {
    "cuda": "improved",
    "cpu": "unverified",
    "mechanism": [
      "avoids_host_device_transfer",
      "avoids_host_synchronization",
      "keeps_intermediate_data_device_resident"
    ]
  },
  "claims": {
    "cuda": [
      {
        "impact": "improved",
        "supporting_record_ids": ["CURRENT_CUDA_RECORD_ID"],
        "scope": {
          "workload": {"iterations": 50, "seed": 20260729, "size": 1048576},
          "device": {
            "class": "cuda",
            "name": "Measured CUDA device",
            "architecture": "Measured architecture",
            "total_memory_bytes": 1024,
            "cpu_model": null,
            "logical_cpu_count": null,
            "affinity_cpu_count": null
          }
        }
      }
    ],
    "cpu": []
  },
  "benchmark": {
    "warmups": 3,
    "pairs": 20,
    "bootstrap_seed": 1729,
    "resamples": 10000,
    "workload": {"iterations": 50, "seed": 20260729, "size": 1048576},
    "equivalence_band": {"low": 0.98, "high": 1.02}
  },
  "compatibility": {
    "warp": ">=1.17",
    "devices": ["cuda"],
    "limitations": ["Example-specific limitations are explicit here."]
  },
  "artifacts": {
    "baseline": "before.py",
    "candidate": "after.py",
    "correctness": "test_correctness.py",
    "benchmark": "benchmark.py",
    "explanation": "explanation.md",
    "evidence": "evidence.json"
  },
  "clean_room": {
    "synthetic": true,
    "derived_from_private_source": false,
    "declaration": "Independently authored from the abstract pattern."
  }
}
```

The exact serialization details may change during implementation, but these
semantic groups are required:

1. Identity and taxonomy.
2. Classification status.
3. Static and dynamic recognition signals.
4. Preconditions and contraindications.
5. Observable behavior and correctness tolerance.
6. CUDA and CPU impact, derived from structured claims.
7. Performance mechanism.
8. Supporting evidence IDs with exact workload and device scopes.
9. Version, device, type, shape, and workload compatibility.
10. Artifact locations.
11. Clean-room declaration.

The schema uses enums where the vocabulary must remain stable and free-form
notes only where examples need explanatory context. Schema versions allow a
future skill to reject unsupported manifests instead of guessing.

### Runtime Optimization Taxonomy

The taxonomy classifies the limiting resource or boundary, not the syntax of
the proposed edit.

#### 1. Host/Device Transfer Elimination

Keep intermediate values on the device and use a device-native operation when
the baseline crosses to a host-only implementation. The expected gain comes
from removing transfer bandwidth, host synchronization, host dispatch, and
temporary conversion costs.

#### 2. Synchronization Avoidance

Remove host-visible scalar reads, per-stage synchronization, or other barriers
that are not required for correctness. The candidate preserves dependency
ordering on the device and synchronizes only at an actual observation boundary.

#### 3. Kernel Fusion and Memory-Traffic Reduction

Combine compatible operations so a value is loaded once, transformed through
several arithmetic steps, and stored once. The goal is to reduce global-memory
round trips and launch overhead. Fusion is conditional when it increases
register pressure, duplicates work, complicates synchronization, or expands a
halo enough to reduce occupancy.

#### 4. Launch Amortization

Move enough independent work into each launch to make useful GPU work large
relative to dispatch overhead. This includes batched operations, persistent
device-side iteration where appropriate, and tile primitives that replace
sequences of tiny kernels.

#### 5. Allocation and Buffer Reuse

Allocate scratch arrays and workspaces at a stable lifetime boundary rather
than inside a repeated runtime path. The candidate must preserve aliasing and
lifetime semantics, and it must not retain unbounded memory.

#### 6. Autodiff Execution Strategy

Use Warp-native recording and backward execution when a Python callback bridge
would cause repeated framework transitions, synchronization, or conversions.
This category also distinguishes safe primal buffer reuse from intermediates
that must remain unique for correct gradients.

#### 7. Data Layout and Coalescing

Arrange or access data so neighboring GPU threads use contiguous, aligned
memory and avoid unnecessary gathers. Entries must include enough shape and
access-pattern evidence to avoid generalizing a layout that helps one kernel
but hurts its consumers.

#### 8. Shared-Memory and Register Reuse

Use tile, shared-memory, or register-local reuse when multiple threads or
operations consume the same values. The benefit must exceed staging and
synchronization overhead for the measured tile and device.

#### 9. Work Decomposition

Choose block, tile, and thread responsibilities that expose sufficient
parallelism without excessive divergence, atomics, or redundant work. Entries
describe the workload region for which the chosen decomposition applies.

#### 10. Device-Native Algorithm Substitution

Replace a correct but boundary-heavy algorithm with a public Warp primitive or
device-native formulation. The candidate may change the computational method
only when the observable-behavior contract and numerical tolerance remain
valid.

#### 11. Interoperability and Zero Copy

Exchange device-resident arrays through a supported interoperability mechanism
instead of staging through host memory. Ownership, lifetime, contiguity,
device, stream, and synchronization requirements are part of the preconditions.

#### 12. Memory/Runtime Tradeoffs

Reduce retained intermediates only when memory pressure is measured and the
extra recomputation improves feasibility or end-to-end runtime. Checkpointing
belongs here as a conditional strategy, never as an unconditional best
practice.

### Initial Corpus

The first implementation phase establishes examples closest to the strongest
available evidence:

| Example | Category | Intended lesson |
| --- | --- | --- |
| Device-resident spectral transform | Transfer elimination; device-native substitution | A supported tiled transform can avoid iterative host round trips |
| Fused elementwise pipeline | Kernel fusion | Perform multiple simple operations between one global read and one global write |
| Reused iteration workspace | Allocation reuse | Move fixed-shape scratch allocation outside a repeated step |
| Native autodiff rollout | Autodiff strategy | Avoid a per-step Python gradient bridge when Warp can record the rollout |
| Gradient-safe intermediate lifetime | Autodiff strategy; buffer reuse | Reuse buffers only when overwritten values are not needed by backward |
| Direct tape without checkpointing | Memory/runtime tradeoff | Do not pay recomputation overhead when the full tape fits comfortably |
| Rejected over-fusion | Kernel fusion | Retain separate kernels when fusion increases resource or halo costs enough to lose |
| Device-resident framework exchange | Interoperability | Preserve a device buffer across a supported framework boundary |

Example names, calculations, shapes, parameters, and code are created
independently during implementation. This table defines only abstract
performance lessons.

Later entries may cover the remaining taxonomy, but each requires independent
correctness and performance evidence. Taxonomy coverage alone is not a reason
to publish an optimization as recommended.

### Correctness Protocol

Correctness precedes benchmarking:

1. Create identical seeded inputs for baseline and candidate.
2. Run both through the same declared workload.
3. Synchronize before observing results.
4. Compare every declared observable output.
5. Apply an example-specific absolute and relative tolerance.
6. For each output, record the maximum normalized error
   `abs_error / (atol + rtol * abs(reference))` and derive the pass result from
   a finite value no greater than `1.0`.
7. Store an unbounded normalized or absolute/relative metric as JSON `null`,
   never `NaN` or infinity, so rejected evidence remains serializable.
8. Reject non-finite output unless non-finite values are explicitly part of the
   contract.
9. Check important boundary shapes and parameter regimes separately from the
   benchmark size.
10. For autodiff examples, compare both primal outputs and requested gradients.
11. For stateful examples, compare the final state and any externally visible
   iteration results.

The correctness test is a normal deterministic unit test. It does not assert a
runtime threshold. Benchmark evidence is recorded separately because shared CI
load and device variation make timing assertions unsuitable for standard
tests.

### Benchmark Protocol

The benchmark measures a fixed runtime path after normal one-time
initialization:

1. Capture the environment and validate that the workload is supported.
2. Construct baseline and candidate using identical inputs.
3. Compile and warm up both outside the timed region.
4. Run at least three additional untimed warm-up iterations.
5. Execute at least ten paired trials, alternating which variant runs first.
6. Synchronize immediately before and after each timed region.
7. Keep every operation required by the repeated user-visible runtime path
   inside the timed region.
8. Record raw baseline and candidate samples for each pair.
9. Report medians, median absolute deviations, the paired ratio distribution,
   and a paired 95% bootstrap confidence interval.
10. Use a fixed bootstrap seed and at least 10,000 resamples.

Alternating order reduces bias from thermal behavior, clocks, and other drift.
Pairing preserves the relationship between measurements taken under similar
conditions. The primary statistic is the optimized/baseline runtime ratio.

For ratio samples \(r_i = t_{\mathrm{candidate},i} /
t_{\mathrm{baseline},i}\), a CUDA result qualifies as improved only when the
upper endpoint of the paired 95% confidence interval is below `1.0`. A
confidence interval crossing parity is inconclusive, even when the point
estimate is favorable.

An example may optionally define a minimum effect threshold stricter than
parity to screen out changes too small to justify added complexity. The
threshold and rationale are stored in the manifest before evaluating the final
result.

### What Is Timed

The timed boundary is explicit in each benchmark.

Included when they occur in the repeated workload:

- Kernel and library operation launches.
- Required host/device and device/device transfers.
- Required synchronization.
- Repeated allocation and deallocation.
- Repeated framework conversion or interoperability setup.
- Any recomputation introduced by a memory-saving strategy.

Excluded:

- Import and process startup.
- Warp compilation and code generation.
- One-time module loading.
- One-time input generation.
- One-time setup explicitly intended to live at an application endpoint.
- Benchmark reporting and statistical analysis.

Moving work outside the timed region is valid only when the candidate also
moves that work outside the real repeated runtime path. The explanation must
state the new lifetime boundary.

### Classification

An evidence run produces one of these classifications:

| Status | Correctness | Runtime evidence | Other conditions |
| --- | --- | --- | --- |
| `unverified` | Not yet evaluated on CUDA or unsupported by current eligible CUDA records | None or inconclusive | It has no positive structured CUDA claim and cannot be shown as a recommendation; CPU impact remains independent |
| `recommended` | Passes | Referenced CUDA 95% CI upper bound is below parity | The exact claim scope matches the declared default workload |
| `conditional` | Passes | Referenced CUDA CI is below parity only for an explicit workload or device region | The structured claim has an exact bounded scope and precise contraindications |
| `rejected` | Fails or passes | Regression, inconclusive result, or improvement outweighed by resource/maintenance constraints | Failure reason is retained |

A strategy may have multiple evidence runs with different classifications.
The card's current status is the most conservative accurate summary of its
supported region. Publication validation derives status and impact only from
eligible record IDs referenced by the structured CUDA and CPU claims.
`improved` and `harmful` labels require matching record classifications.
`neutral` requires the entire referenced interval to fall within the
predeclared equivalence band. `unverified` may retain absent or inconclusive
evidence but cannot satisfy a positive claim. CPU measurements cannot satisfy
the CUDA inclusion gate.

Rejected entries preserve the attempted hypothesis and measurements. They are
not mixed into positive user-facing guidance, but a future skill consults them
before proposing a similar transformation.

### Evidence Record

`evidence.json` contains immutable records for each evaluated environment and
workload. New records use evidence format version 2 and carry a
`measured_contract` envelope with a canonical SHA-256 digest. The envelope
binds:

- Exact measured and declared workloads.
- Exact measured protocol and the manifest's minimum protocol requirements.
- Per-output absolute and relative tolerances.
- The measured Warp version, its declared compatibility specifier, supported
  device classes, predeclared equivalence band, and exact measured device.
- SHA-256 hashes and paths for the baseline, candidate, benchmark, correctness
  entry point, every shared harness module, and the runner.

The remainder of each record includes:

- Corpus schema and example version.
- Git revision and dirty-state indicator.
- Warp, Python, CUDA Toolkit, CUDA driver, and relevant library versions.
- GPU name, architecture, memory capacity, and device ordinal.
- Public CPU model, logical CPU count, affinity-visible CPU count, and
  operating system when CPU is evaluated.
- Workload dimensions, dtypes, iteration counts, parameters, and seeds.
- Warm-up count, pair count, trial order, and synchronization method.
- Raw paired timing samples and units.
- Median and median absolute deviation for each variant.
- Paired ratios, bootstrap seed, resample count, confidence level, and interval.
- Correctness tolerances, finite-state flag, maximum observed error, and
  normalized margin for every output.
- Peak or retained memory measurements when relevant.
- CUDA and CPU impact classification.
- Preconditions, limitations, and rejection reason.
- Timestamp and evidence format version.

Validation separates three decisions:

1. Intrinsic integrity checks the record against its own stored contract,
   recomputes its digest and statistics, and derives correctness and result.
2. Current compatibility and staleness compare the envelope with the current
   manifest, sources, Warp version, and age policy.
3. Claim eligibility requires a current, clean-source, correctly scoped record
   whose classification supports the claimed impact.

Exploratory benchmark output needs intrinsic integrity but not a publication
claim. Checked-in corpus validation applies all three layers. Legacy records
without an envelope remain intrinsically valid and byte-for-byte visible as
historical evidence; they are stale and cannot support current claims.

Evidence is append-only at the logical record level. Revalidation adds a new
record after the existing history instead of overwriting it. Only a newly
appended record must match the current manifest and source contract.

### Harness Responsibilities

The shared harness is intentionally small and example-agnostic:

- `correctness.py` provides deterministic input seeding, output comparison,
  tolerance reporting, and non-finite checks.
- `benchmark.py` performs warm-up, alternating paired trials, device
  synchronization, and raw sample capture.
- `statistics.py` computes descriptive statistics and the paired bootstrap
  interval with deterministic resampling.
- `environment.py` captures software, hardware, revision, and workload
  metadata without embedding machine-specific policy in example code.
- `evidence.py` builds measured contracts, validates intrinsic integrity,
  reports current staleness, and determines structured-claim eligibility.
- `run.py` preflights workload, protocol minimums, and the explicit output
  parent before importing the selected card. A hidden registry-root override
  isolates synthetic subprocess fixtures from production discovery.

The harness validates rather than hides benchmark boundaries. Each card must
declare the callable representing its repeated workload, and reviewers can see
what setup remains outside it.

### Future Optimization Skill Interface

A future skill consumes the corpus through manifests and evidence, using this
workflow:

1. Create an isolated worktree for the target program and assign a unique
   `WARP_CACHE_PATH`.
2. Establish baseline correctness and a representative end-to-end benchmark.
3. Inspect transfers, synchronization, launches, allocation, global-memory
   traffic, data layout, autodiff lifetimes, and interoperability boundaries.
4. Retrieve recommended, conditional, and rejected cards whose recognition
   signals match.
5. Rank candidates by expected impact, evidence confidence, applicability,
   invasiveness, and validation cost.
6. Apply one candidate at a time.
7. Run correctness checks and revert any regression.
8. Run the paired benchmark and keep the change only when its 95% confidence
   interval is entirely below parity.
9. Record rejected candidates so the same session does not repeat them.
10. Benchmark the combined retained changes against the original baseline.
11. Report retained and rejected transformations, evidence, limitations, and
    remaining opportunities in plain language.

The skill must not recommend compilation-time changes from this corpus. It must
not infer that a CUDA result applies to CPU, introduce checkpointing without
measured need, push a branch, or open a pull request without user
authorization. If GPU detection fails in a restricted environment, it retries
in an environment with GPU access before classifying CUDA as unavailable.

### Compatibility and Staleness

Every card declares a supported Warp version range, public APIs used, device
classes, dtypes, shapes, and other relevant constraints. Evidence is not
silently generalized beyond those declarations.

An evidence record becomes stale when:

- Its measured Warp version differs from the current Warp version or falls
  outside the card's current compatibility window.
- Any measured baseline, candidate, benchmark, correctness, shared-harness, or
  runner source hash differs.
- The benchmark protocol, declared workload, measured workload, output names,
  tolerances, device contract, or equivalence band changes.
- A newer device architecture invalidates an important performance assumption.
- A fixed maximum evidence age adopted by project policy expires.

Stale evidence remains visible but cannot support a current claim.
Revalidation creates a new record. Legacy evidence without a measured
contract is always treated as stale rather than invalid.

### Alternatives Considered

#### Narrative Examples Only

Standalone prose and code are easy to read but cannot be reliably retrieved,
validated, ranked, or applied by future tooling. They also encourage claims
whose benchmark context is missing. Human-readable explanations remain part of
each card, but structured metadata is the source of the operational contract.

#### A Static Optimization Checklist

A checklist is useful for review but treats heuristics as universal. It cannot
encode negative evidence, device impact, statistical confidence, or precise
preconditions. The corpus can generate a checklist view without making the
checklist the underlying data model.

#### Positive Examples Only

Omitting failed transformations would make automation repeat known mistakes.
Rejected and conditional cards are retained so applicability boundaries are
first-class information.

#### Microbenchmarks Only

Isolated kernel timings help explain a mechanism but can hide transfers,
synchronization, allocation, and framework crossings. End-to-end steady-state
runtime is the inclusion metric. A card may add lower-level diagnostic timings,
but they cannot replace the paired end-to-end result.

#### Universal GPU Recommendations

Hardware, Warp versions, shapes, dtypes, and neighboring operations change the
outcome. The design records bounded evidence and requires validation against a
user's actual workload.

#### Implement the Skill and Corpus Together

Building autonomous rewriting before the evidence interface stabilizes would
couple recognition and editing logic to ad hoc examples. The corpus and harness
are delivered first; the skill is a separate feature that consumes the stable
contract.

## Testing Strategy

### Per-Change Tests

- Validate every manifest against `example.schema.json`.
- Import every baseline, candidate, correctness test, and benchmark module.
- Run deterministic correctness tests on CPU where supported.
- Run deterministic correctness tests on CUDA where available.
- Run a short benchmark smoke test that validates control flow and record
  production without asserting speed.
- Validate checked-in evidence structure and recompute its summary statistics
  from raw samples.
- Recompute version 2 contract digests, derived correctness, impact
  classification, staleness, and structured-claim eligibility.
- Retain legacy records as intrinsically valid but non-supporting history.
- Exercise boundary correctness sizes independently from the benchmark
  default.
- Scan manifests, explanations, and sources for prohibited private
  identifiers.

### GPU Evidence Tests

Full evidence generation runs on a controlled CUDA worker rather than as a
timing assertion in ordinary CI. It:

- Uses the shared paired protocol.
- Requires the declared minimum pair count.
- Captures a complete environment record.
- Captures an immutable measured contract and clean runtime-source state.
- Confirms correctness in the same run.
- Regenerates statistics from raw samples.
- Applies the classification gate.

Evidence updates are reviewed like source changes. A faster point estimate
without a confidence interval below parity does not justify promotion to
`recommended`.

### Compatibility Tests

Examples with CPU support run on CPU and record independent impact. CUDA
coverage includes the architectures available to project CI, while evidence
manifests state exactly which devices were measured. Version checks ensure that
an unsupported public API yields a clear skip or incompatibility result rather
than a misleading performance classification.

### Periodic Audit

A scheduled audit reports stale evidence, schema drift, missing device
coverage, and examples whose current result conflicts with their classification.
The audit does not silently delete historical evidence.

## Delivery

### Phase 1: Corpus Foundation

- Add the schema, harness, documentation, and validation tests.
- Add the initial clean-room examples with the strongest available runtime
  evidence.
- Record both successful and rejected transformations.
- Integrate the focused suite with the default Warp test entry point where
  appropriate.

### Phase 2: Evidence-Backed Expansion

- Add independently measured examples across the remaining taxonomy.
- Expand device, dtype, and shape coverage.
- Add compatibility and staleness automation.

### Phase 3: Optimization Skill

- Design and implement the separate skill against the stable manifest and
  evidence contract.
- Validate its recognition, ranking, editing, rollback, and reporting behavior
  on public or synthetic programs.

Phases 1 and 2 are the implementation scope following this design. Phase 3 is
documented here only to ensure the corpus provides the data an autonomous tool
will need.
