# Fuse a compatible elementwise pipeline

This synthetic card compares two equivalent ways to execute three pointwise
arithmetic stages. Paired evidence supports the recommendation for the
declared workload only when a current structured claim references an eligible
evidence record.

## Before

The baseline launches one kernel per stage and retains two full-size
intermediate arrays:

```python
wp.launch(affine_stage, dim=size, inputs=[x, bias, intermediate])
wp.launch(bounded_stage, dim=size, inputs=[intermediate, bounded])
wp.launch(polynomial_stage, dim=size, inputs=[bounded, result])
```

Each iteration performs three launches, writes and rereads both intermediates
through global memory, and finally stores the observable result.

## After

The candidate performs the same arithmetic in one kernel:

```python
affine = x[index] * 1.25 + bias[index]
bounded = wp.tanh(affine)
result[index] = bounded * bounded + 0.1 * bounded
```

The inputs are loaded once per element, multiple operations occur between the
global reads and the final global write, and intermediate values can remain in
registers. The candidate needs one launch per iteration and no intermediate
arrays. Stable allocations and trial preparation stay outside the timed
`run()` function for both variants.

## Why it can help

The candidate reduces launch overhead and removes global-memory traffic for
two full-size intermediates. These benefits are workload- and device-specific;
CPU impact is classified independently from CUDA impact.

## When not to fuse

Keep stages separate when they require barriers between stages or when callers
must observe an intermediate. Fusion can also lose when it creates excessive
register pressure, combines substantially divergent stages, or expands a data
halo enough to increase redundant work or reduce occupancy.

## Declared scope

The default workload uses one-dimensional `float32` arrays. Correctness
requires the final `result` to match with absolute tolerance `2e-6` and
relative tolerance `2e-5`.

The first four retained records are historical. The first two predate the
measured-contract envelope, while the next two version-2 records hash an
earlier shared harness source contract. None supports a current structured
claim. Fresh evidence was generated from clean revision
`9c01a9f1db7efca1a3584513efae3eb261f93481` with 20 pairs and 10,000
bootstrap resamples.

On an NVIDIA RTX PRO 6000 Blackwell Server Edition (`sm_120`) with Warp
`1.17.0.dev0`, CUDA Toolkit 13.0, and CUDA driver 13.0, the baseline median was
1,251,125 ns and the candidate median was 448,310 ns. The median paired ratio
was `0.35798423218992614`, with paired 95% confidence interval
`[0.35610791210584825, 0.3613193853079908]`. The complete interval is below
parity, so CUDA impact is `improved` and the card is `recommended` for the
exact default scope. Record `b21fce5d5550491ca22d3803158dde44` supports the
CUDA claim.

On an AMD EPYC 9B45 CPU with 192 logical and affinity-visible CPUs, the
baseline median was 1,014,407,394.5 ns and the candidate median was
966,347,418 ns. The median paired ratio was `0.9532197821540516`, with paired
95% confidence interval `[0.9516633758055284, 0.9547122141526531]`. That
interval is also below parity, so CPU impact is independently `improved`.
Record `d19540be9046429380e42dfc63e9d3be` supports the CPU claim.
