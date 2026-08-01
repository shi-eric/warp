# Reuse fixed iteration workspace

## Original pattern

The baseline applies two unchanged Warp kernels in each iteration. It creates a
new scratch array immediately before those launches, then releases the Python
array object when the loop moves to the next iteration. The result array is
reset once during untimed trial preparation.

This pattern can be easy to write because the scratch lifetime is visibly
local. In a repeated path, however, the program also repeats array-object
construction, allocator bookkeeping, and storage lifetime transitions.

## Candidate pattern

The candidate allocates one scratch array while constructing the variant and
reuses it for every iteration. It launches the exact same two kernel objects
with the same inputs, outputs, order, and iteration count as the baseline. The
candidate does not fuse the stages or move result reset into the measured
region, so the comparison isolates workspace lifetime.

## Why it can run faster

Reuse can remove repeated allocator and array-lifetime work from the
steady-state path. CUDA memory pools may already make storage acquisition
cheap, but reuse can still avoid host-side bookkeeping. On CPU, allocator
behavior and the relative cost of the two kernels may produce a different
effect, so CPU and CUDA evidence are classified independently.

The optimization does not reduce the number of kernel launches or the array
traffic performed by either kernel. It is an allocation-lifetime
transformation, not kernel fusion.

## Correctness contract

Both variants start from the same synthetically generated float32 input and
reset their observable output before each trial. The scratch contents are
overwritten by the first stage on every iteration. Only the accumulated energy
array is observable.

Reuse is valid when:

- scratch shape, dtype, and device remain fixed;
- calls that share one workspace do not overlap; and
- scratch contents neither escape nor become observable between iterations.

Use separate workspace per concurrent call or retain per-shape storage when
the operation legitimately has multiple stable configurations.

## When not to apply it

Do not blindly retain one scratch array when shape, dtype, or device changes.
Do not share mutable workspace across overlapping streams or concurrent calls
without a lifetime and synchronization design. Reuse is also inappropriate
when downstream work retains the scratch result, or when holding the storage
crowds out higher-value device work.

If the active allocator already amortizes allocation and the paired confidence
interval crosses runtime parity, keep the rejected result instead of changing
allocator policy or enlarging the benchmark after observing it.

## Evidence scope

Evidence was recorded from clean source revision
`5b81e209f648664838a64a3b77984620e5262d0a` with 1,048,576 float32 values,
100 iterations, three warm-ups, 20 alternating pairs, and 10,000 bootstrap
resamples.

On an NVIDIA RTX PRO 6000 Blackwell Server Edition, the baseline median was
2,335,704.5 ns and the candidate median was 1,601,350 ns. The paired median
candidate/baseline ratio was `0.6841226631313866`, with a 95% confidence
interval of `[0.6821658749457802, 0.688900549762065]`. Correctness passed with
zero observed error. Record `1fca6467a02e414a934af068b6ef9d41` supports the
CUDA `improved` label and the card's `recommended` status for this exact
scope.

On an AMD EPYC 9B45 CPU, the baseline median was 238,020,310 ns and the
candidate median was 236,045,255 ns. The paired median ratio was
`0.9918275034775634`, with a 95% confidence interval of
`[0.9915450749735908, 0.9920071699976571]`. Correctness again passed with zero
observed error. Although the raw interval is below runtime parity, it lies
entirely within the predeclared equivalence band of `[0.98, 1.02]`; record
`f9f5dc2406c649a9b1b5b21a7aed49b1` therefore supports the CPU `neutral`
label.

These classifications apply only to the recorded device and workload scopes.
They do not imply that workspace reuse benefits every allocator or array size.
