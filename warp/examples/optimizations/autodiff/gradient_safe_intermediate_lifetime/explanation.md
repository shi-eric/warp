# Reuse only gradient-safe intermediates

This synthetic card compares two lifetimes for the same affine rollout and its
input gradient. The CPU and CUDA recommendations are limited to the published
devices, workload, source revision, and protocol below. The transformation
applies only when every step derivative is independent of the overwritten
primal values.

## Baseline

The baseline allocates one unique float32 state for the input and every rollout
step. It marks each state differentiable, records the complete forward rollout
with `wp.Tape`, seeds the final-state gradient with ones, and runs Tape
backward. Each trial resets the input and every retained gradient before
timing. Output states are overwritten completely by their forward launch.

The affine recurrence is:

```python
y[i] = 0.99 * x[i] + 0.0025
```

Its derivative with respect to `x[i]` is the constant `0.99`.

## Candidate

The candidate ping-pongs exactly two preallocated forward buffers. It then
ping-pongs two independently allocated adjoint buffers with an independently
authored kernel:

```python
previous[i] = 0.99 * adjacent[i]
```

Backward does not read a primal state, so overwriting the forward buffers
cannot change this affine input gradient. Trial preparation resets the same
synthetic input, clears both adjoint buffers, and seeds the endpoint adjoint
outside timing.

This candidate is not a general replacement for Tape recording. It trades
general autodiff support for a manually specified derivative whose
value-independence has to be established before construction.

## Runtime and reset boundary

Timed work includes all 64 forward steps and all 64 backward steps. The
baseline additionally includes Tape construction, launch recording, and Tape
backward dispatch. The candidate includes both Python ping-pong loops and all
manual adjoint launches.

Initial array allocation, compilation, input reset, gradient clearing, final
gradient seeding, and NumPy observation are outside timing for both variants.
The harness synchronizes after preparation and after each timed run.

## Memory accounting

For float32 state, the analytical retained forward-intermediate storage is:

```text
baseline = (steps + 1) * size * 4 bytes
candidate = 2 * size * 4 bytes
```

At the default size 262,144 and 64 steps, the baseline retains 68,157,440
bytes (65 MiB) of primal states. The candidate retains 2,097,152 bytes
(2 MiB) in its two forward buffers. This is a retained-forward-state
comparison; it does not pretend that adjoints or simultaneously constructed
variants occupy no storage.

The manifest's conservative 201,326,592-byte (192 MiB) peak covers both
variants alive in one `OptimizationCase`: 65 baseline primal buffers and 65
paired baseline gradient buffers (130 MiB), two candidate forward and two
candidate adjoint buffers (4 MiB), and the shared host reset values (1 MiB).
The remaining 57 MiB allowance covers four retained float32 output snapshots,
float64 baseline and candidate conversions, absolute, relative, and normalized
error arrays, ufunc temporaries, and Python array-object overhead in the v2
correctness harness. The standard workload stays below the 256 MiB corpus
limit and uses no checkpointing.

## Correctness and nonlinear counterexample

Correctness observes only `final_state` and `input_gradient`. Both must match
with absolute tolerance `2e-6` and relative tolerance `2e-5`. The correctness
entry point also runs two trials per variant and requires bitwise-repeatable
outputs after untimed resets.

A separate small `sin(x)` counterexample computes the actual Tape gradient and
the constant manual formula. They disagree because the derivative is
`cos(x)`, which depends on an overwritten primal value. This counterexample is
not timed and supports no runtime claim. Construction rejects
`derivative_depends_on_state=True` with `UnsupportedWorkload` instead of
silently applying the affine manual adjoint.

## Published evidence

Records `93e2fd0d20ab4b32922e127e020fb48a` and
`67e403c24bcf4a80ba7aa41be4082cf0` measured clean repository and runtime
sources at revision `e4eb24ae31645001dcf49a1d3e6314ebeb2ed602`.
Both used the declared workload of 262,144 values, 64 steps, seed 20260730,
and `derivative_depends_on_state=false`. The paired protocol used three
warmups, 20 alternating pairs, 10,000 bootstrap resamples, and bootstrap seed
271828. Both records passed `final_state` and `input_gradient` correctness
with zero maximum absolute, relative, and normalized error.

On CUDA device `cuda:0`, an NVIDIA RTX PRO 6000 Blackwell Server Edition with
architecture 120 and 101,973,950,464 bytes of memory, the environment used
CUDA driver 13.0, CUDA toolkit 13.0, and Warp `1.17.0.dev0`. The baseline
median was 1,545,245 ns with MAD 4,660.5 ns. The candidate median was
1,160,785 ns with MAD 2,880 ns. The paired candidate/baseline median ratio was
`0.751833932509828`, with 95% confidence interval
`[0.7490799565418221, 0.7543300662817362]`. This supports an `improved` CUDA
classification and `recommended` card status only for that exact contract.

On the recorded x86_64 CPU, an AMD EPYC 9B45 with 192 logical and affinity
CPUs and 760,420,753,408 bytes of total memory, the baseline median was
68,473,757.5 ns with MAD 48,040.5 ns. The candidate median was 34,770,468 ns
with MAD 50,795.5 ns. The paired candidate/baseline median ratio was
`0.507113537696794`, with 95% confidence interval
`[0.5068548583955204, 0.5083326087114441]`. This independently supports an
`improved` CPU classification only for that exact contract.

The retained-forward-state byte counts are analytical consequences of the
array shapes, dtype, and lifetimes, not device-memory measurements in either
runtime record. The 65 MiB versus 2 MiB comparison is valid only for the
declared affine value-independent derivative precondition and workload.

## When not to apply it

Do not reuse overwritten primal storage when a nonlinear derivative, custom
gradient, intermediate observation, or other backward consumer needs the
discarded values. Keep the unique-state Tape baseline or design and validate a
different lifetime strategy. Runtime and memory conclusions from this affine
precondition must not be generalized to arbitrary recurrences.

The runtime classifications also do not generalize beyond the recorded
devices and complete default workload. Other devices, sizes, step counts, or
manual adjoints require their own correctness and paired measurements.
