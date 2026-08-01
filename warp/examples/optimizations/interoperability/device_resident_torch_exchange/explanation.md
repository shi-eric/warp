# Preserve a CUDA buffer across frameworks

This synthetic CUDA card compares two ways to apply the same in-place Warp
transform to PyTorch data. Paired evidence supports the device-resident
candidate for the exact CUDA device, workload, source, framework versions, and
protocol reported below. CPU execution and impact are unsupported and
unverified.

## Baseline

Every timed iteration stages the current CUDA PyTorch tensor through host
NumPy storage, allocates and copies into a new Warp CUDA array, launches the
in-place transform, copies the Warp result back to a new host NumPy array, and
stages that result into a new CUDA PyTorch tensor:

```python
host = torch_values.detach().cpu().numpy()
warp_values = wp.array(host, dtype=float, device=device)
wp.launch(transform_values, inputs=[warp_values], outputs=[warp_values])
host_result = warp_values.numpy()
torch_values = torch.from_numpy(host_result).to(torch_device)
```

All transfers, allocations, view construction, and framework crossings in
that sequence remain inside every timed iteration. The synchronous host
observations also provide the lifetime boundary after which superseded
staging objects can be released safely.

## Candidate

The candidate creates one contiguous float32 CUDA PyTorch tensor and wraps it
once with `wp.from_torch()`. The Torch tensor remains the storage owner, and
the Warp array is a zero-copy view with the same CUDA device and data pointer.
Every transform launch mutates that shared storage in place. Only after the
last launch does `wp.to_torch()` create the endpoint observation view.

The PyTorch owner, Warp view, endpoint PyTorch view, and converted stream stay
alive through trial synchronization. The correctness entry point asserts
device and pointer equality while those objects are live.

## Matching runtime semantics

Both variants begin every prepared trial from the same deterministic float32
values and apply this transform for the same iteration count:

```python
output[i] = 0.5 * input[i] + sin(input[i])
```

Both run in place at the Warp kernel boundary, synchronize their owned CUDA
stream at the same timer boundary, and expose the final PyTorch values only
after timing. Trial reset, initial allocation, compilation, synchronization
before the timer, and final NumPy observation are untimed.

The default workload contains 1,048,576 values and 20 iterations. Correctness
uses absolute tolerance `2e-6` and relative tolerance `2e-5`. The
`67108864`-byte peak estimate covers both live variants, their stable reset
storage, the baseline's current host and device staging objects, and
allocator headroom. It remains below the 256 MiB corpus limit.

## Stream ordering and lifetime

Zero-copy view creation shares storage but does not order work. Each variant
therefore owns a non-default PyTorch CUDA stream, converts it with
`wp.stream_from_torch()`, and enters both `torch.cuda.stream()` and
`wp.ScopedStream()` while PyTorch and Warp operate on the values. A correctness
probe schedules a real PyTorch producer, the Warp transform, and a PyTorch
consumer on that same non-blocking stream before synchronizing it.

A non-default PyTorch stream remains non-blocking when Warp wraps it. Every
owner and view used by asynchronous work must remain alive until that stream
finishes. Applications using different streams must establish equivalent
event or stream-wait ordering before either framework consumes the shared
buffer.

## Published evidence

Evidence record `4dae121b697440f0a1250100949a1fab` measured clean repository
and runtime sources at revision
`cfda1dfa89d3338a6bd6823d100cce34ffa67a1b`. The environment used Python
3.12.13, PyTorch `2.13.0+cu130`, Warp `1.17.0.dev0`, CUDA toolkit 13.0, and
CUDA driver 13.0. The measured `cuda:0` device was an NVIDIA RTX PRO 6000
Blackwell Server Edition with architecture 120 and 101,973,950,464 bytes of
memory.

The default workload used 1,048,576 values, 20 iterations, and seed 20260730.
The paired protocol used 3 warmups, 20 alternating pairs, 10,000 bootstrap
resamples, and bootstrap seed 161803. Correctness passed with zero absolute,
relative, and normalized error for `values`.

The baseline median was 14,810,222.5 ns with a median absolute deviation of
19,185 ns. The candidate median was 210,275 ns with a median absolute
deviation of 13,950 ns. The candidate/baseline median ratio was
`0.014207820228111913`, with paired 95% confidence interval
`[0.012920709267062404, 0.01474594813987275]`. Because the complete interval
is below runtime parity, the mechanical CUDA classification is `improved`.
That classification applies only to the exact environment, device, workload,
source revision, and protocol above. CPU impact remains `unverified` and has
no claim.

## Preconditions and contraindications

Apply this pattern only to a compatible contiguous float32 CUDA tensor on the
same device as Warp. The source tensor must own its storage for every Warp
use, and the source, Warp view, endpoint view, and stream wrappers must remain
alive through completion. The application must accept aliasing and in-place
mutation, and producer, Warp, and consumer operations must be ordered.

Keep an explicit copy when dtype or strides are unsupported, a consumer
requires independent storage, mutation is not allowed, owner lifetime cannot
be guaranteed, or stream ordering cannot be established. Pointer equality
alone is not a lifetime or synchronization mechanism.

## How to use the result

The candidate removes repeated host transfers and allocation but still pays
for the identical transform launches and required synchronization. Recheck
correctness and runtime before applying the result to another PyTorch or Warp
version, device, workload, tensor layout, or stream topology.
