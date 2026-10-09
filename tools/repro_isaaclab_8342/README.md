# Isaac Lab #8342 investigation

Investigated on 2026-10-07–08. **The reported allocation hang was reproduced on
this NVIDIA A40 with driver 595.97**, using the stock 8,192-camera task at the
reported Isaac Lab commit and its public OVRTX/OVStage pins. The reporter's
source builds are not required to reproduce the symptom here.

**A procedural sphere scene now reproduces the captured allocation hang with
only Warp, NumPy, and OVRTX installed.** Isaac Lab, Newton, MuJoCo, Torch,
OVStage, robot assets, and downloaded USD files have been removed. The latest
parameterized probe is `probe_ovrtx_capture.py`. The smallest confirmed script is
`minimal_ovrtx_alloc_capture.py`, a single file with its three dependency pins.
It authors the entire scene as an inline USD string and uses OVRTX's standalone
APIs. Its environment inventory is `evidence/ovrtx-no-stage-packages.txt`;
the parameterized probe also rejects environments where Torch or OVStage is importable.
The observed configuration has 8,192 camera partitions and 32 generated spheres
per partition, publishes GPU transforms, renders/maps/releases a frame, then
captures one Warp kernel and a 32 KiB allocation. One of three fresh processes
hung in round 1; the other two passed all three rounds. Repeated Python stacks
are saved in `evidence/logs/ovrtx-no-stage-hang.txt`.

Further confirmed reductions remove image mapping/copying/unmapping and retain
the render products through capture. Removing the prefix kernel also reproduced
the same native allocation stall. Run `probe_ovrtx_capture.py --no-map --no-prefix`
for this configuration: the first captured operation is the 32 KiB allocation.
Its stack is saved in `evidence/ovrtx-no-prefix-native-unwind.txt`. GPU transform
publication remains in this confirmed reduction; the static scene passed once.
The compact script independently reproduced the same stall without loading
`prefix_kernels.py` or allocating its counter array. Capture contains only
`wp.empty(8192, dtype=wp.int32)`, with the verification fill outside capture.
Repeated Python samples and native unwind are saved in
`evidence/logs/minimal-ovrtx-alloc-hang.txt` and
`evidence/minimal-ovrtx-native-unwind.txt`.

The native stack of the stock task and an independent exported-scene replay
reaches `cuMemAllocAsync` from `wp_alloc_device_async` and blocks in a Windows
driver call. The standalone OVRTX reproduction has been natively unwound to
the same call in `evidence/ovrtx-no-stage-native-unwind.txt`. Its native module
list contains no OVStage or Torch DLLs (`evidence/ovrtx-no-stage-module-check.txt`).
The renderer turned out not to be required; see the CUDA Python reproduction
below. A fresh nonblocking stream
completed once and subsequently stalled at the same allocation, so that change
is not a reliable workaround. A Warp library defect has not been established.

Issue: <https://github.com/isaac-sim/IsaacLab/issues/8342>.
The issue's attached reproduction archive was downloaded and inspected, including
its PowerShell harness, scene configuration, logs, and sampled Python stacks.

## Root cause: UVM-Lite VA window placement

**A CUDA Python script reproduces the stall without OVRTX, Warp, NumPy, or
Vulkan.** The stall is not a wait for GPU work. During a live OVRTX hang the
GPU was idle (P8, about 1% utilization, 4 GiB used). Meanwhile the capturing
thread ran at 100% CPU in kernel mode. Between two dumps of the stock task taken
154 seconds apart, it gained 150 seconds of kernel time and no user time. It is
inside a CUDA driver loop:

1. The capture's first stream-ordered allocation needs a new 96 GiB VA window
   for graph allocations.
2. Managed memory has initialized UVM-Lite, so the driver reserves
   `{base, 96 GiB}` with `\Device\UVMLiteController*` (IOCTL `0x22e004`). It
   then reserves the same range in the GPU address space.
3. When the second reservation fails, the driver releases the UVM-Lite range
   (IOCTL `0x22e008`). That release takes 1 to 3.8 seconds of kernel CPU on this
   A40. The driver then retries 2 MiB higher.

The window's first-fit base is directly after OVRTX's own 52 GiB managed
window, at `0x4c00000000`. Renderer allocations made later occupy that range,
so the driver walks it 2 MiB at a time. This costs about one second per MiB, and
several GiB is effectively a hang.

All seven decoded hang dumps are blocked in the same release IOCTL, with a 96 GiB
length and a base just above `0x4c00000000`. They cover the stock Isaac Lab
task, its nonblocking-stream variant, the exported-scene replay, and the
standalone OVRTX scripts. The doubly dumped process had advanced 58 steps of
2 MiB in between (`evidence/hang-dump-ioctl-args.txt`). A hooked OVRTX hang
(`evidence/ovrtx-uvm-ioctls.txt`) shows the whole sequence:

- At 128 seconds, an OVRTX thread initializes UVM-Lite and commits a 0x4820-byte
  managed allocation.
- Placing OVRTX's 52 GiB managed window fails seven times, at about 1.5 seconds
  each.
- The capture then issues repeated 96 GiB reserve/release pairs.

This accounts for earlier observations:

- Captures that "passed" in about two seconds were a single failed placement.
- Larger scenes put more renderer VA in the window.
- The CUDA/Vulkan interop probes never allocated managed memory. Without UVM-Lite,
  the same collision costs about a millisecond.
- The driver sometimes places a new window at an unrelated base, which bypasses
  the collision. The CUDA Python reproduction shows this directly, which likely
  explains why the OVRTX hang is intermittent.

Warp only issues the allocation. Any first allocation from a new stream-ordered
pool or graph-allocation window is affected, whether captured or not.

### CUDA Python reproduction

`minimal_uvm_va_capture.py` depends only on `cuda-bindings`. It reserves
foreign VA through `D3DKMTReserveGpuVirtualAddress` with ctypes. From the
repository root:

```powershell
uv run --no-project tools/repro_isaaclab_8342/minimal_uvm_va_capture.py
```

It allocates from the default pool, then makes one managed allocation. It
reserves 16 MiB of GPU VA at the first free 96 GiB range, where CUDA places its
next window, and then captures one 32 KiB `cuMemAllocAsync`. That allocation
takes about 16 seconds instead of about a millisecond. `--obstacle-mib` scales the
stall by about one second per MiB. In 14 of 18 fresh processes, the allocation
took 15 to 18 seconds. In the others, the driver chose an unrelated base, and the
script printed `INCONCLUSIVE`. Controls with the same obstacle:

- `--no-managed`: 0.001 seconds.
- `--warmup`, which captures one allocation before the foreign VA exists:
  0.000 seconds.
- `--vulkan` replaces D3DKMT with one 2 MiB `vkAllocateMemory` block. Run it with
  `uv run --no-project --with vulkan==1.3.275.1 ... --vulkan --obstacle-mib 2`.
  It was still stalled after 140 seconds, when it was terminated. With
  `--no-managed`, it took 0.001 seconds.

Run results are in `evidence/minimal-uvm-va-runs.txt`. To list the driver's
placement attempts, run the script under `trace_uvm_ioctls.py`. It uses only
ctypes, so it also works in the OVRTX environment:

```powershell
uv run --no-project --with cuda-bindings==12.9.5 python tools/repro_isaaclab_8342/trace_uvm_ioctls.py --log uvm.txt tools/repro_isaaclab_8342/minimal_uvm_va_capture.py
```

`evidence/cuda-python-uvm-trace.txt` is one such trace. It has eight
reserve/release pairs 2 MiB apart, and each release took 1.7 to 2.5 seconds.

### Mitigation

Reserving the graph-allocation window before foreign VA occupies its first-fit
base avoids the collision. One captured allocation, with its graph destroyed, was
enough: later captures reused the window. This held whether the warm-up ran
before or after the managed allocation. `cuDeviceGraphMemTrim` did not release
the window: a traced capture after a trim made no new reservation.

The stock Isaac Lab benchmark confirms this (`evidence/isaaclab-warmup-ab.txt`).
`run_isaaclab.py --warmup-capture early` captures one 4-byte `wp.empty` when
`gym.make` is entered, and `--warmup-capture solver` does so from
`NewtonManager._invalidate_graph`, where Newton captured before #8058. Runs were
interleaved and traced:

| Variant | Hung | Graph-window placement |
| --- | --- | --- |
| Baseline (including two runs with `ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS` unset, which defaults to `1`) | 6 of 8 | Hangs started at `0x4c00000000` when capture began and failed 34–41 times in 90 s. Both passes used an alternative base. |
| `early` | 0 of 6 | Reserved 13–17 s after start, before UVM-Lite existed; no UVM-Lite placement at all |
| `solver` | 0 of 6 | Reserved at 132–145 s on the first attempt, at `0x4c00000000` in 5 of 6 runs; none at capture |

Zero of 12 warm-up runs hung against 6 of 8 baseline runs (one-sided Fisher
exact p = 0.0007). The traces also show why each run passed. This also explains
the #8058 regression: Newton used to capture at the `solver` point, after
UVM-Lite initialization but before foreign VA filled the next window's base.
Capturing at the first step moved the first graph allocation past that point.

The `solver` point works only because it precedes that foreign VA. The `early`
point precedes UVM-Lite entirely and is the more robust choice.

`ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS=0` does not help in this configuration:
both runs hung. With OVStage, the managed allocation comes from `ovstage.dll`
through `usdrt.hierarchy.plugin.dll` and `omni.cubric.plugin.dll`. In the
standalone OVRTX script without OVStage, `read_gpu_transforms=False` removed it.

Graph allocations exceeding the window, or a new device or memory pool, would
still require a new window.

### Open questions for the driver team

- The IOCTL roles are inferred. Their parameter layouts are consistent with UVM's
  `RESERVE_VA`, `RELEASE_VA`, and `REGION_COMMIT`.
- Releasing an unused 96 GiB UVM-Lite range costs about two seconds of kernel
  CPU.
- Placement retries in 2 MiB steps. Without UVM-Lite, the same collision is
  resolved in about a millisecond.
- The failing GPU VA call is resolved dynamically. It is not visible to the
  import hook.

## Environment

| Item | This machine | Reported machines |
| --- | --- | --- |
| OS | Windows 11, build 26100 | Windows 11 Enterprise, build 26200 |
| GPU | NVIDIA A40, 48 GiB, sm_86, WDDM | RTX 6000 Ada / L40 |
| Driver | 595.97; CUDA driver API 13.2 | 596.72 / 596.86 |
| Warp | 1.17.0 wheel; 1.19.0.dev0 checkout | 1.17.0 |
| Embedded Warp CUDA Toolkit | 12.9 wheel; 13.4 checkout | Not established by the report |
| MuJoCo / MuJoCo-Warp | Both 3.12.0, isolated temporary environment | MuJoCo-Warp 3.12.0 |
| Renderer | Public OVRTX 0.5.0.377615 / OVStage 0.2.0.377349 probed | Source-built OVRTX 0.6.0 / OVStage 0.3.0 |

Checkout: `d3c56b3a65cc98eed3f557be63a89e84e6f0675b`.
Python: 3.12.10. NumPy: 2.5.3. CUDA memory pools are enabled and supported.
The installed system Toolkit is 13.4; no native libraries were rebuilt.
An isolated checkout of Isaac Lab at the reported commit was installed with
`uv sync --frozen --extra ovrtx`, including Newton 1.6.1, Torch 2.12.0+cu130,
Warp 1.17.0, and MuJoCo-Warp 3.12.0. Its environment uses Python 3.12.15 and
NumPy 2.5.1. The standalone probes use a separate environment.

The exact renderer source builds are not included in the attached archive and
were not available on public PyPI. The public renderer versions above are the
pins from Isaac Lab's reported commit, but the reporter explicitly did not test
them. They are not an equivalent reproduction environment.

## Results

| Probe | Result |
| --- | --- |
| Warp 1.17.0 allocation matrix | 72/72 processes passed; 7,200 captures; 230,400 allocations |
| Warp 1.19.0.dev0 allocation matrix | 72/72 processes passed; 7,200 captures; 230,400 allocations |
| Small 32 KiB candidate probe | 100 captures/replays passed on each Warp version |
| MuJoCo-Warp 3.12.0, Warp 1.17.0, no eager warmup | 100 captures/replays passed for 8,192 worlds |
| Same, with eager warmup | 3/3 processes passed, 100 captures/replays per process |
| MuJoCo-Warp 3.12.0, current Warp, relaxed/nonblocking capture with warmup | 100 captures/replays passed for 8,192 worlds |
| Warp 1.17.0 captured allocation with pending legacy-stream host callback | Returned before the callback was released; capture/replay passed |
| Official OVRTX v0.5.0 minimal example | Passed; rendered and saved an image; cold startup took 297 seconds |
| Public OVRTX/OVStage + Warp 1.17.0, populated 64×64 camera stage | 100 render/capture/replay rounds passed; image dimensions and buffer values checked |
| Same with MuJoCo-Warp steps for 8,192 worlds | 3/3 fresh processes passed, 100 rounds each, with Warp 1.17.0 |
| Same with the current Warp checkout | 100 rounds passed |
| Stock Isaac Lab physics-only task, 8,192 environments | Passed, exit 0; three warmup and three measured steps |
| Stock Isaac Lab camera task, 8,192 environments, public renderer pins | Reproduced: no capture return; repeated stacks in `make_constraint -> wp.empty -> allocate`; terminated after preserving diagnostics |
| Same task, capture target reduced to one 32 KiB allocation/fill | Passed; capture returned in 2.031 seconds; this replaces physics and is a diagnostic only |
| Same task, capture target reduced to MuJoCo-Warp's step | Same allocation stall; Newton's collision/contact conversion is not required inside capture |
| Same task, capture target reduced to MuJoCo-Warp kinematics and one allocation | Same allocation stall; constraint construction and the solver are not required inside capture |
| Same task, capture target reduced to one generic Warp kernel and one allocation | Same allocation stall; captured MuJoCo/Newton physics is not required; full initialized renderer scene retained |
| Same task, generic Warp kernel then direct `cuMemAllocAsync`/`cuMemFreeAsync` | Passed once, both CUDA calls returned success; capture returned in 1.938 seconds; one pass cannot rule out an intermittent stall |
| Same task, original physics target on a fresh nonblocking stream | 2/3 fresh processes passed (capture returned in 1.891 and 2.250 seconds); one stalled at the same allocation and was terminated after saving diagnostics; thread-local mode throughout |
| Same stock camera task, one environment | Passed, exit 0; cold startup took 841 seconds |
| Standalone sphere scene cloned to 8,192 camera partitions, generic Warp kernel then allocation, GPU image mapping | Three render/capture/replay rounds passed; image dimensions, allocation values, and kernel output checked; no Isaac Lab or MuJoCo |
| Procedural 32-sphere scene, 8,192 cameras, GPU transforms and hierarchy, shading mode 3, Torch, CUDA mapping and product release | Reproduced in round 1: repeated Python stacks at captured `wp.empty(8192, dtype=wp.int32)`; timed out at 360 seconds |
| Same procedural configuration, one sphere per environment | Ten rounds passed in one fresh process (35.375 seconds); a negative control, not proof that 32 spheres are necessary |
| Same 32-sphere configuration, Torch initialization omitted in the original environment | Ten rounds passed in one process (144.984 seconds) |
| Same 32-sphere configuration in a fresh environment with only Warp, NumPy, OVRTX, and OVStage installed | One of three fresh processes hung in round 1 at captured allocation; two repeated Python samples and native unwind confirm the stall; the other two passed three rounds each |
| Inline USD scene, same camera/sphere counts and GPU transforms, only Warp, NumPy, and OVRTX installed | One of three fresh processes hung in round 1 at captured allocation; native unwind confirms `cuMemAllocAsync`, with no OVStage/Torch DLLs loaded; the first two passed three rounds each |
| Same standalone OVRTX scene under Nsight Systems CUDA/Vulkan tracing | Hung in round 1; repeated Python stacks at the captured allocation; trace completed at its 480-second limit |
| Same scene under a native CUPTI API-parameter logger | Hung in round 1; `cuMemAllocAsync` entry for 32 KiB has no exit; native unwind confirms the driver stall |
| Same scene with image mapping/copy/unmap omitted and render products retained | Hung in round 1; native unwind confirms the same `cuMemAllocAsync` stall |
| Same no-mapping scene with the prefix kernel removed from capture | Hung at the first captured operation, `wp.empty(8192)`; native unwind confirms the same driver stall |
| Compact standalone script: generated USD, two GPU transform writes, one render, capture only 32 KiB allocation | Confirmed the same native driver stall; no kernel import/preload, counter array, image mapping, or captured fill; terminated after saving diagnostics at 392.547 seconds |
| Static authored scene, no transform publication and no image mapping | One process passed three rounds; a negative control, not proof that transform publication is necessary |
| No-mapping allocation-only scene reduced to 4,096 cameras, 32 spheres per partition | One process passed three rounds (78.328 seconds); the compact confirmed script retains 8,192 cameras |
| CUDA Python + Warp, 128 MiB copy with cross-stream temporary allocation/free and event handoffs | Three fresh processes passed, 100 rounds each; no renderer or graphics API calls |
| Vulkan + CUDA Python + Warp, 128 MiB shared buffer with matched external semaphore handoffs | Three fresh processes passed, 100 rounds each; no OVRTX |
| Vulkan + CUDA Python + Warp, 128 MiB shared RGBA8 image, CUDA-array copy into temporary storage, cross-stream async free, then capture | Three fresh processes passed, 100 rounds each; no OVRTX |
| Same image/staging model with the traced 5,824 x 5,824 extent and CUDA color-attachment flag | Three fresh processes passed, 100 rounds each |
| Same model with 1,024 queued Vulkan writes per frame | Three fresh processes passed, 20 rounds each; producer event was pending before every capture |
| Six independent CUDA/Vulkan producers, non-dedicated 16 MiB shared buffers, no image copy or Warp stream wait | Three fresh processes passed, 20 rounds each; all six producer events were pending before every capture |
| Exported renderer scene replay, no Isaac Lab or physics imports, GPU transforms and Torch | Reproduced in round 1; native unwind confirms `cuMemAllocAsync`; a subsequent instrumented 20-round process passed, so the failure is intermittent |
| Same full task, generic kernel and direct CUDA driver allocation/free, two additional fresh processes | Both passed; three passing processes total, insufficient to establish immunity to an intermittent bug |
| Warp + Torch, 8 GiB of live Warp allocations, generic kernel then captured allocation, no renderer | Three fresh processes passed, 20 capture/replay rounds each |
| Pending legacy-stream host callback, generic Warp kernel then allocation | Passed with allocation returning before the callback was released |
| Live standalone OVRTX hang, GPU and thread sampling | GPU idle in P8 at about 1% utilization; main thread gained 32.6 s of kernel time in 31 s |
| Same, `DeviceIoControl` import of `nvcuda64.dll` hooked | Hung; repeated 96 GiB UVM-Lite reserve/release pairs 2 MiB apart, 1.9–3.8 s per release |
| CUDA Python only: managed allocation, 16 MiB of foreign GPU VA at the next window, captured 32 KiB allocation | 14 of 18 fresh processes took 15–18 s; the others placed the window elsewhere |
| Same, without the managed allocation | 0.001 s |
| Same, with one 2 MiB Vulkan allocation instead of D3DKMT | Still stalled at 140 s and terminated; 0.001 s without the managed allocation |
| Same, warm-up captured allocation before the foreign VA | 0.000 s |
| Stock Isaac Lab camera task, traced, interleaved | Baseline hung in 6 of 8 runs; warm-up capture at `gym.make` or at `NewtonManager._invalidate_graph` hung in 0 of 12 |
| Same, `ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS=0` | Hung in 2 of 2; OVStage still makes the managed allocation |

Each allocation matrix covers:

- `THREAD_LOCAL`, `RELAXED`, and `GLOBAL` capture modes.
- Warp's default blocking stream (driver flags 0) and a driver-created
  nonblocking stream (flags 1).
- With and without GPU kernel work queued on the legacy stream before capture.
- Warp `wp.empty`/fill/free and direct `cuMemAllocAsync`/`cuMemFreeAsync` calls.
- Three fresh processes per combination, each making 100 captures with 32
  allocations per capture. Warp allocations alternate between temporaries freed
  during capture and arrays retained beyond capture. Each graph is replayed
  twice, with a counter checked against the expected value.

The matrix allocations are approximately 4 MiB each. MuJoCo's constraint buffer
and the smaller candidate probe allocate 8,192 int32 entries (32 KiB). The
MuJoCo probe uses a sphere contacting a plane with the Newton solver, executes
`mjw.step`, checks time advancement, and checks finite positions. It exercises
the reported `make_constraint -> wp.empty` path, without reproducing the Kuka
scene, Newton integration, or camera workload.

Early renderer probes timed out at 120 seconds during attachment, with both
renderer-first and stage-first initialization. That cutoff was inadequate for
cold shader compilation: the official example subsequently completed at 297
seconds, and the populated-stage rendering/capture probe completed at 16 seconds
with the cache warmed. Those early timeouts are **not** evidence of a renderer
deadlock and are **not** reproductions of #8342.

Native minidumps of the stock camera task show its main thread waiting in
renderer attachment, while worker instruction pointers move through
`nvgpucomp64.dll`, `nvrtum64.dll`, and the NVIDIA graphics driver and process CPU
time increases. Candidate stack pointers were scanned without a symbolized
unwind; they must not be presented as precise native call stacks.

After renderer initialization, the instrumented stock task logged
`GRAPH_CAPTURE_ENTER device=cuda:0 relaxed=False` and never logged its return.
Repeated Python samples matched the issue at `make_constraint`, line 4873,
and Warp's allocator, line 5044. Unlike shader compilation, native samples at
this point had no active compiler threads. DbgHelp `StackWalk64`, supplied with
the dump's memory and the local PE exception tables, subsequently unwound the
main thread through:

```text
NtDeviceIoControlFile
DeviceIoControl
NVIDIA driver internal frames
cuMemAllocAsync + 0x25
Warp's embedded CUDA Runtime frames (without private symbols)
wp_alloc_device_async + 0xdb
ctypes
```

Only exported names are available; large offsets from neighboring exports are
not meaningful function names. This establishes that the observed stall is
inside CUDA allocation, rather than Warp's later graph-node queries. It does
not by itself establish whether Warp, the renderer, or the driver is at fault.

An initial matrix harness error called `cuStreamCreate` without a current CUDA
context, producing error 201 in nine processes. The harness was corrected to
set the context and both complete matrices were rerun. Only those final results
are retained here. An initial inline USDA syntax error was also corrected before
the renderer attachment timeout was measured.

## What the source establishes

At the reported Isaac Lab commit
[`1d5a8fc`](https://github.com/isaac-sim/IsaacLab/blob/1d5a8fce0eb3fdf2ba4eb0f5a441c61794c3fed1/source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py),
`NewtonManager._capture_graph` selects relaxed capture only when
`has_kit() and (sim.has_gui or sim.has_offscreen_render)` is true. Kitless OVRTX
therefore takes thread-local capture on the existing Warp stream. The relaxed
branch also creates a separate nonblocking Torch stream. Mode and stream flags
both differ between those paths.

MuJoCo-Warp 3.12.0's `make_constraint`, line 4873, allocates
`wp.empty((d.nworld,), dtype=int)` on every step. Eager warmup does not remove that
allocation from a subsequent recording. This agrees with the reporter's stack.

Warp 1.17.0's allocator at `context.py:5044` enters
`wp_alloc_device_async`, whose implementation calls `cudaMallocAsync` before
tracking the graph allocation node. A Python stack sampled at that ctypes call
does not distinguish a stall in CUDA allocation, CUDA context management, or
the following node queries. A native stack is needed to identify the blocking
operation precisely.

CUDA explicitly supports graph capture of stream-ordered allocation/free calls.
Allocations during recording are not inherently a Warp bug. Separately, CUDA
prohibits use of the legacy stream while a blocking stream in the same context
is capturing. The public
[CUDA graph documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
and
[stream synchronization documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html)
describe these rules. Relaxed mode does not change stream flags or eliminate
implicit legacy-stream dependencies.

An earlier working hypothesis pointed at CUDA/graphics synchronization during
capture. It is superseded by the root cause above. The stall is a driver-side
VA placement loop, not a wait on GPU work, which is also why stream flags made
no difference.

The generic kernel reduction is in `prefix_kernels.py`. `--kernel-only`
preloads that module, then records only the following work (the full environment
is still initialized before this point):

```python
values = NewtonManager._solver.mjw_data.qpos
wp.launch(prefix_kernels.prefix, dim=values.shape, inputs=[values], device=device)
temporary = wp.empty(8192, dtype=wp.int32, device=device)  # Stalls here.
retained_allocations.append(temporary)
temporary.fill_(42)
```

`evidence/kernel-only-python.txt` records the stalled Python stack. This does
not require MuJoCo's constraint kernel inside capture; the input array is still
obtained from the initialized fixture. The initial stock-task allocation-only
variant passed once. Later standalone OVRTX reductions reproduced the hang
without the preceding launch; a captured prefix kernel is not required.

`--raw-allocation` retains the same generic kernel, then calls the CUDA driver
allocator/free functions through ctypes with the current context and stream,
bypassing Warp's allocation wrapper and its embedded CUDA Runtime. Its three
passing processes are a useful comparison, not evidence that either API is immune to
this intermittent bug. The nonblocking control already demonstrates why a
single passing process is insufficient.

## Runnable diagnostics

### Renderer-free CUDA/Vulkan candidates

`probe_cuda_interop_capture.py` is a **tested candidate, not a confirmed
reproduction**. All 18 fresh processes across six tested configurations
passed, totaling 1,320 captures. OVRTX, OVStage, and Torch were absent from the
environment, and the script rejects environments where those packages are
importable. The `--backend cuda` control does not import Vulkan or create
graphics resources. The Windows Vulkan backends replace rendering with a
buffer fill or image clear and use matched external semaphore signals/waits,
queue ownership transfers, and checked image contents.

Create a separate environment and test the image/staging configuration:

```powershell
uv venv "$env:TEMP/warp-8342-cuda-interop"
uv pip install --python "$env:TEMP/warp-8342-cuda-interop/Scripts/python.exe" -r tools/repro_isaaclab_8342/requirements-cuda-interop.txt
uv run --no-project --python "$env:TEMP/warp-8342-cuda-interop/Scripts/python.exe" python tools/repro_isaaclab_8342/probe_cuda_interop_capture.py --backend vulkan-image --staging --rounds 100 --image-mib 128
```

Use `--backend cuda --staging` for the CUDA allocation/copy/free model and
`--backend vulkan` without `--staging` for the shared-buffer model. The image
model uses a 4,096-by-8,192 RGBA8 image, imports its Vulkan allocation as a
CUDA mipmapped array, copies to temporary linear CUDA memory, makes a retained
Warp copy, and queues the temporary free on the producer stream after a Warp
event. It destroys those temporary events before the original Warp capture.
Image data, graph allocation values, and the kernel counter are all verified.
The installed inventory is `evidence/cuda-interop-packages.txt`; summaries are
`evidence/cuda-staging128.json`, `evidence/vulkan-handoff128.json`, and
`evidence/vulkan-image-staging128.json`.

Additional controls match the traced image dimensions/flags, keep Vulkan work
pending through capture, and queue six producers independently of the Warp
stream. They also passed. `--producer-repeats` queues alternating writes with
write-after-write barriers, ending with the checked value; `--report-pending`
queries each producer event before capture. `--background` removes image copies
and Warp stream waits, then joins the producers after capture/replay. The six
buffer producers each have their own Vulkan device, whereas the OVRTX trace
does not establish that topology. These are scheduling models, not an exact
replay of the renderer's internal commands.

```powershell
uv run --no-project --python "$env:TEMP/warp-8342-cuda-interop/Scripts/python.exe" python tools/repro_isaaclab_8342/probe_cuda_interop_capture.py --backend vulkan-image --image-side 5824 --staging --producer-repeats 1024 --report-pending --rounds 20
uv run --no-project --python "$env:TEMP/warp-8342-cuda-interop/Scripts/python.exe" python tools/repro_isaaclab_8342/probe_cuda_interop_capture.py --backend vulkan --image-mib 16 --buffers 6 --background --non-dedicated --producer-repeats 1024 --report-pending --rounds 20
```

Summaries are `evidence/vulkan-image-matched5824.json`,
`evidence/vulkan-pending5824.json`, and `evidence/vulkan-background16.json`.

The models were selected from a trace of the actual OVRTX hang, rather than a
known isolated driver trigger. Nsight Systems 2026.3.2 collected CUDA and Vulkan
software traces, with CUDA event tracing and CPU sampling disabled. Its captured
stdout/stderr contains repeated allocator samples in
`evidence/logs/nsys-ovrtx-hang.txt`. The trace shows external memory/semaphore
imports, external semaphore waits, a mapped mipmapped array, an asynchronous
3D copy, event handoffs, and `cudaFreeAsync` immediately before capture begins.
The last completed CUDA calls include the prefix launch and allocator context
setup; capture never returns. `evidence/nsys-ovrtx-handoff.txt` saves the relevant
API counts and final completed calls. Incomplete calls are not represented by
completed API records. Raw `.nsys-rep`/SQLite data remains under the temporary
investigation directory.

A separate native CUPTI callback logger captured the CUDA arguments during
another confirmed OVRTX hang. `evidence/cupti-ovrtx-parameters.txt` records
dedicated opaque-Win32 image memory (137,166,848 bytes), an RGBA8 5,824 x 5,824
array with flags 10 (surface access and color attachment), seven non-dedicated
16 MiB shared buffers, and six opaque-Win32 binary semaphore imports. All 58
external waits use a zero fence value and zero flags. The final runtime
`cudaMallocAsync` and driver `cuMemAllocAsync` entries request 32,768 bytes on
the Warp capture stream and have no exit. The native unwind is in
`evidence/cupti-ovrtx-native-unwind.txt`.

`trace_cuda_handoff.cpp` implements the logger, and `trace_cuda_handoff.py`
loads it before running the target. The callback reads API parameters and writes
log records; it never calls CUDA. Build the DLL from an x64 Visual Studio native
tools prompt, with `CUDA_PATH` pointing to a Toolkit containing CUPTI:

```cmd
cl /nologo /LD /MT /O2 /EHsc /I"%CUDA_PATH%\include" /I"%CUDA_PATH%\extras\CUPTI\include" tools\repro_isaaclab_8342\trace_cuda_handoff.cpp /Fo"%TEMP%\trace_cuda_handoff.obj" /Fe"%TEMP%\trace_cuda_handoff.dll" /link /LIBPATH:"%CUDA_PATH%\extras\CUPTI\lib\x64" cupti.lib
```

Run from the diagnostic directory using the renderer-only interpreter:

```powershell
uv run --no-project --python "$env:TEMP/warp-8342/renderer-no-stage/Scripts/python.exe" python trace_cuda_handoff.py --dll "$env:TEMP/trace_cuda_handoff.dll" --cupti "$env:CUDA_PATH/extras/CUPTI/lib/x64" --log "$env:TEMP/cuda-handoff.txt" probe_ovrtx_capture.py
```

This is enough evidence to construct targeted probes, but it does not establish
which renderer operation, parameter, resource lifetime, or pending GPU workload
is necessary. The matching CUDA API patterns alone did not reproduce the hang.
The interop setup follows NVIDIA's
[CUDA/Vulkan interop description](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/graphics-interop.html)
and [CUDA driver bindings](https://nvidia.github.io/cuda-python/cuda-bindings/12.9.0/module/driver.html).

### Confirmed standalone OVRTX reproduction

The smallest confirmed reproduction is a single file with three inline package
pins. From the repository root:

```powershell
$env:OMNICLIENT_HUB_MODE = 'disabled'
uv run --no-project tools/repro_isaaclab_8342/minimal_ovrtx_alloc_capture.py
```

The file is also runnable with the already-installed three-package interpreter:

```powershell
uv run --no-sync python tools/repro_isaaclab_8342/run_probe.py --python "$env:TEMP/warp-8342/renderer-no-stage/Scripts/python.exe" --label minimal-ovrtx-repeat --timeout 900 --trials 3 minimal_ovrtx_alloc_capture.py
```

Its captured work is exactly:

```python
with wp.ScopedCapture(capture_mode=wp.CaptureMode.THREAD_LOCAL) as capture:
    allocation = wp.empty(8192, dtype=wp.int32)  # Confirmed cuMemAllocAsync stall.
```

No image is mapped, copied, or unmapped, and products are kept through capture.
The generated scene still has 8,192 camera partitions and 32 spheres per
partition; GPU transform publication and OVRTX rendering remain. A passing run
replays the graph, fills the allocation outside capture, and checks its contents.
`evidence/minimal-ovrtx-alloc.json` records the observed process.
The inline-dependency uv invocation installed exactly those three packages and
completed the script's checks in a fresh environment. Its redirected PowerShell
command reported exit 1 despite the final `Passed` message, so it is recorded
as completed script checks rather than a clean process pass. The raw log,
inventory, and limitation are in `evidence/minimal-ovrtx-uv.json` and its associated
log/package files. Cold initialization took substantially longer in that fresh
environment; the confirmed allocation hang above used the existing environment.

To run the **confirmed reproduction without OVStage or Torch** from the Warp
repository root, using the environment already installed on this machine:

```powershell
$env:OMNICLIENT_HUB_MODE = 'disabled'
uv run --no-sync python tools/repro_isaaclab_8342/run_probe.py --python "$env:TEMP/warp-8342/renderer-no-stage/Scripts/python.exe" --label ovrtx-no-stage-repeat --timeout 900 --trials 3 probe_ovrtx_capture.py --cameras 8192 --objects 32 --rounds 3 --inline-environments
```

To create this three-package environment elsewhere:

```powershell
uv venv "$env:TEMP/warp-8342-renderer-no-stage"
uv pip install --python "$env:TEMP/warp-8342-renderer-no-stage/Scripts/python.exe" -r tools/repro_isaaclab_8342/requirements-renderer-no-stage.txt
```

Use that interpreter with the command above. The pins are Warp 1.17.0,
OVRTX 0.5.0.377615, and NumPy 2.5.1; the tested interpreter is Python 3.12.15.
No scene or robot files are needed. OVRTX's standalone USD API is deprecated
in this version but remains functional. Its bundled OpenUSD, USDRT, graphics,
and CUDA libraries are still involved; removing the separate OVStage package
does not remove those renderer internals.

The first two processes passed in 155.000 and 267.156 seconds. The third rendered
its frame and stalled at `wp.empty(8192, dtype=wp.int32)` inside thread-local
capture. Repeated Python samples and a native unwind confirmed the allocation
stall before that process was manually terminated at 421.015 seconds. Native diagnostics used
the dump's thread memory and local DLL exception tables, without private
symbols. The two-camera version also passed three rounds.

Allow several minutes for cold shader startup. A trace timer firing during
USD population or transform writes is not itself evidence of this bug. Earlier
attempts using legacy cloning/relationship writes failed inside rendering
before capture, with a frame submission semaphore timeout; those failures are
not counted as allocation reproductions. Authoring all partitions and camera
relationships directly in USD avoided that separate renderer failure.

To run the **confirmed Torch-free procedural reproduction** with the
already-installed interpreter on this machine, from the Warp repository root:

```powershell
$env:OMNICLIENT_HUB_MODE = 'disabled'
uv run --no-sync python tools/repro_isaaclab_8342/run_probe.py --python "$env:TEMP/warp-8342/renderer-only/Scripts/python.exe" --label renderer-only-repeat --timeout 900 --trials 3 probe_renderer_capture.py --cameras 8192 --objects 32 --rounds 3 --kernel-first --gpu-map --hierarchy GPU_INCREMENTAL --release-products --shading-mode 3 --gpu-transforms
```

The `renderer-only` environment contains exactly the four packages in
`requirements-renderer.txt`: Warp 1.17.0, OVRTX 0.5.0.377615, OVStage
0.2.0.377349, and NumPy 2.5.1. It uses Python 3.12.15. This script imports no
Isaac Lab or physics packages. No scene files are needed. The optional
`--torch` switch remains available for comparisons and is omitted above.
The hang occurs after `Round 1: before captured allocation`; periodic Python
stack samples identify the blocking allocation. This is an intermittent bug,
so a passing process does not invalidate the observed stall. In the Torch-free
batch, the first and third processes passed and the second hung. Its native snapshot
contained no Torch DLLs or CUDA 13 runtime. The additional CUDA runtime loaded
by Torch is therefore not required to reproduce this symptom.

To create a separate environment from the four-package dependency file:

```powershell
uv venv "$env:TEMP/warp-8342-renderer-only"
uv pip install --python "$env:TEMP/warp-8342-renderer-only/Scripts/python.exe" -r tools/repro_isaaclab_8342/requirements-renderer.txt
```

Use that interpreter with the command above. Allow several minutes for cold
renderer startup. The first completed run in the new environment took
687.766 seconds, mostly before renderer attachment returned. An earlier
240-second startup timeout was insufficient and was not an allocation hang.
We also copied the same renderer version's existing shader/derived-data caches
to shorten recompilation; two locked Vulkan cache files could not be copied.
This did not introduce USD scene or robot inputs.

`minimal_alloc_capture.py` is a small **candidate probe**, not a confirmed
reproduction. It repeatedly captures the same-sized allocation as the reported
constraint buffer, with progress messages and a traceback timer. Run it on an
affected machine first:

```powershell
uv run tools/repro_isaaclab_8342/minimal_alloc_capture.py
```

To test the reported package versions independently of the checkout:

```powershell
uv venv "$env:TEMP/warp-8342-venv"
uv pip install --python "$env:TEMP/warp-8342-venv/Scripts/python.exe" warp-lang==1.17.0 mujoco==3.12.0 mujoco-warp==3.12.0
uv run --no-project --python "$env:TEMP/warp-8342-venv/Scripts/python.exe" python tools/repro_isaaclab_8342/minimal_alloc_capture.py
```

`run_probe.py` and `run_matrix.py` launch children with the supplied virtual
environment interpreter and save output. On Windows, `run_probe.py` uses
`taskkill /T` on its own child when it times out, including the actual Python
interpreter underneath a virtual-environment launcher. This requires Windows
process-query/termination access; it stops with an error if cleanup fails.
The process-tree timeout was checked directly with a nested sleeping child.
`run_matrix.py` retains its original single-child termination behavior.
For example:

```powershell
uv run python tools/repro_isaaclab_8342/run_matrix.py --python "$env:TEMP/warp-8342-venv/Scripts/python.exe" --label warp117 --trials 3
uv run python tools/repro_isaaclab_8342/run_probe.py --python "$env:TEMP/warp-8342-venv/Scripts/python.exe" --label mujoco117 --timeout 300 probe_mujoco_capture.py --warmup
```

The older OVStage-based renderer probes additionally require `ovrtx==0.5.0.377615` and
`ovstage==0.2.0.377349`. Allow several minutes for cold shader compilation:

```powershell
uv run python tools/repro_isaaclab_8342/run_probe.py --python "$env:TEMP/warp-8342-venv/Scripts/python.exe" --label renderer-mujoco --timeout 900 --trials 3 probe_renderer_capture.py --mujoco
```

`run_isaaclab.py` wraps the stock benchmark and logs graph-capture entry/return.
Its default behavior preserves the original capture target and settings. Its
diagnostic switches are `--allocation-only`, `--mujoco-only`,
`--kinematics-only`, `--kernel-only`, `--raw-allocation`, `--nonblocking`, and `--new-blocking`. The first five alter
captured physics and must not be interpreted as fixes or validated simulation.
`--warmup-capture early|solver` adds one captured allocation before the
environment exists or when Newton schedules capture; see the mitigation section.
Run it with the isolated Isaac Lab interpreter and the original benchmark CLI
arguments, including both renderer environment variables from the issue.

The confirmed capture-body reduction can be run with the already-installed
isolated environment on this machine:

```powershell
$env:ISAAC_LAB_OVRTX_USE_OVSTAGE = '1'
$env:ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS = '1'
$env:OMNICLIENT_HUB_MODE = 'disabled'
$env:HEADLESS = '1'
uv run python tools/repro_isaaclab_8342/run_probe.py --python "$env:TEMP/warp-8342/IsaacLab/.venv/Scripts/python.exe" --label reduced-kernel --timeout 600 run_isaaclab.py --kernel-only --task Isaac-Lift-KukaAllegro-Camera --num_envs 8192 --num_steps 3 --warmup_steps 3 --seed 42 --device cuda:0 presets=newton_mjwarp,ovrtx,single_camera,simple_shading_full_mdl64
```

Omit `--kernel-only` to reproduce the original capture target. This requires the
Isaac Lab source checkout and its renderer extra; the temporary environment
above was created at commit `1d5a8fce0eb3fdf2ba4eb0f5a441c61794c3fed1`.

`probe_renderer_capture.py --cameras 8192 --kernel-first --gpu-map --rounds 3`
tests a standalone renderer fixture with the same camera count. It passed here
and therefore is a candidate reduction, not a reproducer.

The `evidence/` directory contains final per-process summaries, representative
logs, and GPU information. Complete scratch logs and the downloaded attachment
remain in `C:/Users/horde/AppData/Local/Temp/warp-8342`.

Rows marked `failed` in the saved summaries for diagnosed stalled runs
reflect manual termination after saving Python/native diagnostics. They are
hang observations, not Python exceptions or spontaneous application crashes.
The fresh-blocking-stream control was interrupted before completion and is
excluded from the result table.

## Remaining work

The CUDA Python reproduction attributes the stall to the Windows CUDA driver's
UVM-Lite VA placement. It does not depend on Warp, OVRTX, or the renderer's
bundled libraries. Remaining items:

- Report the driver behavior with `minimal_uvm_va_capture.py`.
- Propose an early graph-window warm-up to Isaac Lab. It was tested on this A40
  only, not on the reporters' RTX 6000 Ada and L40 machines.
- Determine whether the driver's alternative window placement is random, or
  depends on process state.

No core Warp fix is needed for the root cause. Warp could reserve the graph
window early as a mitigation, but that has not been evaluated.
