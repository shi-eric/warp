# Differentiate a rollout with one Warp Tape

This synthetic card compares two ways to differentiate the same repeated Warp
kernel on CUDA. The CUDA recommendation is limited to the published device,
workload, and protocol below. CPU performance remains unverified.

## Baseline

The baseline defines a PyTorch `autograd.Function` for one rollout step. Every
forward callback maps its PyTorch input with `wp.from_torch()`, allocates a
Warp output, launches the recurrence kernel, and maps the output back with
`wp.to_torch()`. The rollout chains one callback per step.

During backward, every callback makes its incoming gradient contiguous, maps
it to Warp, allocates an input-gradient array, and launches the same public
Warp kernel in adjoint mode. Backward starts from the sum of the final state,
so the observable input gradient is the derivative of that scalar endpoint.

## Candidate

The candidate maps the input once at the rollout endpoint and creates one
unique Warp output array and its gradient storage per step during setup. It
clears Tape gradients during untimed trial preparation, records all forward
launches under one `wp.Tape`, and invokes `tape.backward()` once with an
all-ones final-state gradient. The final state and input gradient are mapped
back to PyTorch only at the opposite endpoint.

Both variants use this exact recurrence:

```python
y[i] = wp.sin(0.7 * x[i]) + 0.05 * x[i]
```

The candidate does not change this kernel, its launch count, its arithmetic,
or the rollout's forward and backward semantics. The runtime comparison
includes Python autograd callback dispatch and repeated framework mapping. It
also includes the baseline's repeated per-callback output and input-gradient
allocation and zeroing, while the candidate moves stable output and gradient
allocation plus Tape-gradient clearing outside the timed loop.

## Runtime boundary

Timed work includes the complete forward rollout and backward differentiation
for both variants. It includes every callback, mapping, launch, intermediate
output allocation required by the baseline, Tape recording, endpoint-gradient
assignment, and backward dispatch. The harness synchronizes the device before
stopping each timer.

Setup, initial arrays, stable candidate output and gradient arrays, gradient
seed storage and initialization, candidate Tape-gradient clearing, trial input
reset, and compilation are outside timing. Assignment of the stored seed to the
endpoint adjoint occurs inside the timed `tape.backward()` call. Trial
preparation restores the same synthetic input and clears accumulated gradients
for each variant before the harness starts the timer. Final NumPy observation
occurs after timing.

The default workload has 262,144 float32 values and 32 steps. Its conservative
128 MiB peak estimate includes retained forward states and gradients for both
constructed variants, below the 256 MiB corpus limit. It retains every state
needed by backward and does not use checkpointing.

## Correctness contract

Correctness observes `final_state` and `input_gradient`. Both must match with
absolute tolerance `2e-5` and relative tolerance `2e-4`. The deterministic
correctness entry point also runs two trials of each variant and requires
identical results, verifying that untimed resets prevent gradient
accumulation.

## Published evidence

Evidence record `7195a562d19e4eb993d9e1f15ecced62` measured clean
repository and runtime sources at revision
`32470c48c7bfe9ac0f4954a5d9373e03f3905373`. The measured device contract is
CUDA device `cuda:0`, an NVIDIA RTX PRO 6000 Blackwell Server Edition with
architecture 120 and 101,973,950,464 bytes of memory. The environment used
CUDA driver 13.0, CUDA toolkit 13.0, and Warp `1.17.0.dev0`.

The default workload used 262,144 values, 32 steps, and seed 20260730. The
paired protocol used 3 warmups, 20 pairs, 10,000 bootstrap resamples, and
bootstrap seed 161803. Correctness passed with zero error for `final_state` and
`input_gradient`.

The baseline median was 3,032,415.0 ns and the candidate median was 750,125.0
ns. The baseline MAD was 30,695.0 ns and the candidate MAD was 40,495.0 ns.
The candidate/baseline median ratio was `0.24331105599122876`, with paired 95%
confidence interval
`[0.23144665755251131, 0.2576835578152612]`. This supports an `improved` CUDA
classification only for that exact device, default workload, source revision,
runtime configuration, and protocol. Other CUDA configurations require their
own measurements. CPU performance remains `unverified` and has no claim.

## When not to apply it

Keep the host-framework callbacks when operations between rollout steps must
remain in that framework or intermediate states are genuine framework
consumers. Do not replace custom gradients whose behavior Warp does not
support. A full Tape also retains rollout states for backward; when measured
memory pressure makes that unsuitable, evaluate a separately specified
memory/runtime strategy rather than adding checkpointing to this comparison.

The CUDA impact is `improved` only within the published scope above. CPU is
outside this card's supported standard-test devices and remains `unverified`
with no performance claim.
