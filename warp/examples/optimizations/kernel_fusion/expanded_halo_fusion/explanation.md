# Avoid fusion that expands a separable halo

This synthetic CUDA card compares a separable two-dimensional stencil with a
mathematically equivalent product stencil. Paired evidence rejects the fused
candidate for the exact measured device and default workload because its
complete confidence interval is above parity.

## Baseline

The baseline performs one horizontal pass into a full-size temporary field,
then one vertical pass into the next state:

```python
wp.launch(horizontal_pass, inputs=[source, weights], outputs=[temporary])
wp.launch(vertical_pass, inputs=[temporary, weights], outputs=[result])
```

For radius `r`, each pass reads `2r + 1` values per output. The two passes
therefore issue two launches, read two one-dimensional neighborhoods, and
write and reread one full-size intermediate field.

## Candidate

The candidate substitutes the horizontal expression directly into the
vertical expression. One kernel visits the complete product neighborhood:

```python
for row_offset in range(-radius, radius + 1):
    for column_offset in range(-radius, radius + 1):
        result += source[clamped_row, clamped_column] * row_weight * column_weight
```

This is fusion in the read/operate/write sense: a thread performs more
arithmetic between its source reads and final global write, removes a launch,
and eliminates the intermediate field. It also expands work. The direct form
loads `(2r + 1)^2` source values per output instead of the split form's two
sets of `2r + 1` values. Neighboring threads request overlapping halos, and
the larger expression can increase instruction count and resource demand.

Fusion is therefore a tradeoff, not an unconditional rule. Removing a global
intermediate helps only when that benefit outweighs redundant halo loads,
extra arithmetic, and any occupancy effect.

## Matching semantics

Both variants start every prepared trial from the same deterministic
float32 field, use the same normalized triangular weights, apply the same
number of iterations, and clamp row and column indices to the same boundary
cells. Only the final `result` is observable. It must match with absolute
tolerance `3e-5` and relative tolerance `3e-4`.

Stable arrays and weights are allocated during case construction.
`prepare_trial()` restores the first input state before timing. The timed
region includes every stencil launch and all stencil arithmetic, but excludes
allocation, compilation, reset, synchronization before the timer, and NumPy
observation.

## Declared workloads

The default evidence workload is a `2048` by `2048` field, 20 iterations,
radius 2, and seed `20260730`. Radius 2 gives five taps per separable pass and
25 direct source loads per fused output.

The source also predeclares exactly one contingent follow-up before any
measurement: the same shape, iteration count, and seed with radius 4. Radius
4 gives nine taps per separable pass and 81 direct source loads per fused
output. The follow-up is authorized only if the default candidate's complete
95% confidence interval is below parity. No other shape, radius, or iteration
retuning is part of the evidence plan.

CPU execution is unsupported and unverified. The manifest's
`134217728`-byte estimate covers the 112 MiB of known device arrays retained
by both live variants at the default shape, plus 16 MiB of headroom for
weights, module state, and allocator bookkeeping. The standard workload stays
below the 256 MiB device-memory ceiling.

## How to use the result

Evidence record `d60e3a7fdd2c454bba3fd7a60fbb84a0` measured clean source
revision `8303eb71e81f720333c5bb0bb616aff4a3cbae52`. The CUDA device was an
NVIDIA RTX PRO 6000 Blackwell Server Edition with architecture 120 and
101973950464 bytes of total memory. The environment used CUDA driver 13.0,
CUDA toolkit 13.0, Warp `1.17.0.dev0`, and Python 3.12.13.

For the exact default workload, correctness passed with maximum absolute
error `5.21540641784668e-08` and maximum normalized error
`0.0008941981503807623`. The paired protocol used three warmups, 20
alternating pairs, 10000 bootstrap resamples, and bootstrap seed 161803.

The separable baseline median was 754515 ns with MAD 1000 ns. The
expanded-halo candidate median was 1169659.5 ns with MAD 1875 ns. The paired
candidate/baseline median ratio was `1.5508574073813775`, with a 95%
confidence interval of
`[1.5462755904604266, 1.5603224785792487]`. The complete interval is above
parity, so CUDA impact is `harmful` and the candidate is `rejected` for this
exact scope. CPU impact remains `unverified`.

Because the default candidate was not faster, the predeclared radius-4
contingency did not trigger and no larger-radius timing was collected. This
result is evidence against this specific fusion shape, not against kernel
fusion generally. Compatible pointwise pipelines can still benefit by doing
more operations between global reads and writes when fusion does not expand
the data halo or create excessive resource demand.
