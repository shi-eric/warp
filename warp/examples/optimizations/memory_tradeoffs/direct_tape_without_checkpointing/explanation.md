# Keep a bounded rollout on one Tape

This synthetic CUDA card compares semantically correct segmented
recomputation with one direct Warp Tape for a bounded nonlinear rollout.
Published evidence supports an improved CUDA result for the exact measured
device and bounded workload. CPU behavior is unsupported and unverified.

## Baseline

The baseline first runs the complete forward rollout and stores the input plus
one primal state at every segment boundary. It uses two scratch arrays for
this original forward and copies each completed boundary into persistent
storage.

Backward visits segments in reverse. For each segment it restores the saved
starting boundary into a reusable set of unique differentiable states,
recomputes the segment under a fresh `wp.Tape`, seeds the segment endpoint
with the propagated boundary adjoint, calls `tape.backward()`, and copies the
preceding boundary adjoint into one reusable propagation buffer. The fresh
Tape and its recorded launches are discarded before the next segment. This is
checkpointing with valid primal reconstruction, not an approximation.

## Candidate

The candidate allocates one unique differentiable output for every step,
records the complete original forward under one `wp.Tape`, and calls
`tape.backward()` once. It does not checkpoint or recompute any forward step.

Both variants use exactly:

```python
y[i] = 0.8 * wp.tanh(x[i]) + 0.02
```

They start from the same deterministic float32 input and expose only
`final_state` and `input_gradient`.

## Runtime, allocation, and zeroing boundaries

Stable Warp arrays, their required gradient arrays, the final-gradient seed,
and the deterministic NumPy reset values are allocated during case
construction, after the capacity guard passes. Compilation occurs during
untimed warm-up. `prepare_trial()` restores the input before the harness
synchronizes and starts the timer.

Timed work includes all original forward steps and all backward work. It also
includes every gradient clear required for reuse. The candidate clears every
retained-state gradient at the start of `run()`. The baseline performs its
original forward first, then clears its reusable segment-state gradients
immediately before each segment recomputation. In both cases, each clear
serves the current trial while leaving the completed trial's outputs
observable. The candidate also includes complete Tape construction, recording,
final-gradient assignment, and one backward call.

The baseline additionally includes segment-boundary copies, segment-state
restoration, recomputed forward launches, one fresh Tape per segment,
endpoint-adjoint assignment, and preceding-boundary-adjoint copies. These are
unavoidable orchestration differences created by segmented recomputation.
Final NumPy observation occurs after timing.

## Capacity accounting

Let:

```text
B = size * 4 bytes
```

For the default `size=131072`, `B=524288` bytes, or 0.5 MiB. The direct Tape
retains:

```text
primal states = (steps + 1) * B
              = 65 * 524288
              = 34078720 bytes
              = 32.5 MiB

paired gradients = (steps + 1) * B
                 = 34078720 bytes

endpoint seed = B
              = 524288 bytes

direct-Tape estimate = (2 * (steps + 1) + 1) * B
                     = 68681728 bytes
                     = 65.5 MiB
```

Before generating the host input or constructing either variant, the card
reads `resolved_device.free_memory`, takes integer one quarter of that value,
and rejects the workload when the direct-Tape estimate is larger. The
construction log reports the exact estimate, free-memory reading, budget, and
combined-peak estimate. The default needs at least 274726912 bytes of free
device memory to pass this rule. A nominal 16 GiB device has a 4 GiB
quarter-memory budget when all memory is free, leaving substantial headroom.
The actual guard always uses current free memory, not nominal capacity.

At the default `steps=64` and `segment_length=8`, there are eight segments.
The baseline's persistent device storage is:

```text
original forward scratch = 2 * B
saved boundaries = (8 + 1) * B
segment primals and gradients = 2 * (8 + 1) * B
propagated boundary adjoint = B
baseline total = 30 * B = 15728640 bytes = 15 MiB
```

Both live variants therefore use a conservative known device total of:

```text
direct candidate + segmented baseline
= (131 + 30) * B
= 84410368 bytes
= 80.5 MiB
```

The manifest's 134217728-byte (128 MiB) conservative peak adds one host reset
array, four retained float32 correctness snapshots, 20 state-sized buffers for
float64 comparison arrays and ufunc temporaries, and 35 MiB of fixed headroom
for Tape metadata, array objects, and allocator bookkeeping. This analytical
construction-time bound is distinct from runtime timing and is below the
corpus's 256 MiB standard-test ceiling.

## Correctness contract

The declared workload is `size=131072`, `steps=64`, `segment_length=8`, and
`seed=20260730`. Segment length must divide the step count exactly. CUDA
correctness is also registered at `size=1024`, `steps=8`, and
`segment_length=2`.

Both observable arrays must match with absolute tolerance `3e-5` and relative
tolerance `3e-4`. The card's correctness entry point runs two prepared trials
of each variant and requires bitwise repeatability before applying the shared
normalized-error check.

## Published evidence

Evidence record `d63b7583a4a44f4f8b0609e2d8dd3614` measured clean
repository and runtime sources at revision
`45ba6892d042abcee032ae663d5b03203d7e1213`. The measured device was CUDA
device `cuda:0`, an NVIDIA RTX PRO 6000 Blackwell Server Edition with
architecture 120 and 101973950464 bytes of total memory. The environment used
CUDA driver 13.0, CUDA toolkit 13.0, and Warp `1.17.0.dev0`.

The exact default workload used 131072 float32 values, 64 steps, segment
length 8, and seed 20260730. The paired protocol used three warmups, 20
alternating pairs, 10000 bootstrap resamples, and bootstrap seed 141421.
Correctness passed with zero maximum absolute, relative, and normalized error
for both `final_state` and `input_gradient`.

The segmented baseline median was 2160034 ns with MAD 5955 ns. The direct-Tape
candidate median was 1434825 ns with MAD 5415 ns. The paired
candidate/baseline median ratio was `0.6644388246355657`, with a 95%
confidence interval of `[0.6619549932317912, 0.667304729146921]`. The entire
interval is below parity, so the record supports an `improved` CUDA
classification and `recommended` card status only for this exact measured
contract. CPU impact remains `unverified`.

The benchmark's pre-allocation capacity log reported 68681728 direct-Tape
bytes, 101388648448 free device bytes, a 25347162112-byte quarter-memory
budget, and a 134217728-byte conservative combined peak. These are the
construction-time analytical estimates and free-memory reading for the
recorded run, not a measured peak-memory trace.

## When checkpointing is appropriate

Checkpointing reduces retained intermediates but adds forward arithmetic,
kernel launches, copies, Tape construction, and Python orchestration. A direct
Tape is preferable only while its complete required storage fits the declared
capacity rule with headroom and measured memory pressure does not displace
other work.

Use segmented recomputation when `full_tape_exceeds_memory_budget`, when
measured allocator or application pressure makes the direct Tape infeasible,
or when the rollout is not predictably bounded. The capacity guard is a
construction precondition, not evidence that every co-resident application
will have enough memory.
