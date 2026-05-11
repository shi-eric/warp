# Hardware-Coherent Cross-Device Memory Access

**Status**: Proposed

## Motivation

Warp currently enforces a strict rule: every array argument passed to `wp.launch()` must reside on the same device as the kernel launch target. If a user creates an array on the CPU and attempts to launch a CUDA kernel that reads it, Warp raises a `RuntimeError`. This enforcement exists in `warp/_src/context.py` inside the `pack_arg()` function (around line 6808):

```python
# check device
if value.device != device:
    raise RuntimeError(
        f"Error launching kernel '{kernel.key}', trying to launch on "
        f"device='{device}', but input array for argument '{arg_name}' "
        f"is on device={value.device}."
    )
```

This restriction is correct on discrete-GPU systems (e.g., a workstation with a PCIe-connected NVIDIA GPU) where the GPU genuinely cannot dereference a pointer to unpinned CPU memory. However, a growing class of NVIDIA hardware uses **unified memory architectures** where the GPU _can_ directly access CPU memory, and on some systems the CPU can also directly access GPU allocations:

- **Grace C2C systems (GH200, GB200, DGX Spark)** -- Grace ARM CPU + Hopper or Blackwell GPU connected via NVLink Chip-to-Chip (C2C). These systems provide bidirectional ATS: the GPU can access all system memory, and the CPU can directly access CUDA device allocations.
- **Jetson Orin and other limited Tegra systems** -- Integrated GPUs sharing the same DRAM as the CPU, but with a limited unified memory model where ordinary system allocations are not necessarily GPU-accessible.
- **Jetson Thor** -- Tegra Blackwell SoC with CUDA-reported ATS. On a Thor development kit tested with CUDA 13.0, the GPU can directly access ordinary system allocations (`malloc`, anonymous `mmap`, and file-backed `mmap`) and host-native atomics work, but CPU direct access to `cudaMalloc` memory is still not supported.
- **HMM-capable discrete systems** -- Linux kernel 6.1.24+ with Heterogeneous Memory Management (HMM) enabled allows software-coherent access to all system memory from PCIe GPUs, without requiring explicit CUDA allocation APIs.

On all systems where the CUDA device reports `pageable_memory_access`, the strict `value.device != device` check is overly conservative and forces users into unnecessary `wp.copy()` or `.to(device)` calls that are both a performance penalty and an ergonomic burden. On HMM and ATS systems in particular, a plain `malloc`'d pointer is directly accessible from the GPU -- there is no need to copy data at all.

### User impact

A user on DGX Spark writing:

```python
data = wp.array([1.0, 2.0, 3.0], device="cpu")
wp.launch(my_kernel, dim=3, inputs=[data], device="cuda:0")
```

gets a `RuntimeError` even though the hardware can handle this directly. The user must write:

```python
data = wp.array([1.0, 2.0, 3.0], device="cpu")
data_gpu = data.to("cuda:0")  # unnecessary copy on ATS systems
wp.launch(my_kernel, dim=3, inputs=[data_gpu], device="cuda:0")
```

This is not just an inconvenience -- it defeats one of the primary benefits of unified-memory hardware, which is eliminating explicit data movement.

## Background: CUDA Unified Memory Paradigms

CUDA exposes unified memory capabilities through device attributes. The sections below group the relevant attribute combinations into four capability buckets so the implementation can reason about access rules mechanically, not by assuming behavior from a product family name. Three buckets have concrete platform examples in this document; the managed-only bucket is retained as a conservative fallback for the attribute combination where managed memory is fully shared but ordinary pageable system memory is not.

### Paradigm 1: Limited Unified Memory (Limited Tegra, Windows, WSL)

**Detection:** `cudaDevAttrConcurrentManagedAccess == 0`

Applies to Windows systems including WSL and to Tegra/Jetson devices whose CUDA attributes report limited managed access. Do not infer this from the Jetson family name alone: Jetson Thor tested with CUDA 13.0 reports `concurrentManagedAccess == 1`, `pageableMemoryAccess == 1`, and `pageableMemoryAccessUsesHostPageTables == 1`, so it does not fall into this paradigm.

Characteristics:
- Only memory explicitly allocated via `cudaMallocManaged` (or `cudaMallocFromPoolAsync` with `cudaMemAllocationTypeManaged`, or `__managed__` globals) behaves as unified memory.
- Managed memory starts in CPU physical memory, is bulk-migrated to the GPU when a kernel begins executing, and is bulk-migrated back on synchronization.
- The CPU must not access managed memory while the GPU is active.
- Oversubscription of GPU memory is not allowed.
- System allocations (`malloc`, `mmap`) are NOT GPU-accessible.

On limited/non-I/O-coherent Tegra specifically:
- `cudaHostRegister()` is not supported on non-I/O-coherent Tegra devices.
- `cudaMallocHost` produces uncached memory from the GPU's perspective on non-I/O-coherent Tegra.
- All memory physically resides in the same shared DRAM, but visibility is controlled by the CUDA driver.

### Paradigm 2: Full Unified Memory for CUDA-Managed Allocations Only

**Detection:** `cudaDevAttrConcurrentManagedAccess == 1` AND `cudaDevAttrPageableMemoryAccess == 0`

Characteristics:
- Memory allocated via `cudaMallocManaged` has full unified memory support (page-granularity migration, concurrent CPU/GPU access, oversubscription).
- System allocations (`malloc`, `mmap`) are still NOT GPU-accessible.
- This is an attribute-defined bucket. This document does not currently identify a specific tested platform for it, but the implementation should handle it separately because its access rules differ from both limited unified memory and HMM/ATS.

### Paradigm 3: Full Unified Memory with Software Coherency (HMM)

**Detection:** `cudaDevAttrPageableMemoryAccess == 1` AND `cudaDevAttrPageableMemoryAccessUsesHostPageTables == 0` AND `cudaDevAttrConcurrentManagedAccess == 1`

Available on Linux with kernel 6.1.24+ / 6.2.11+ / 6.3+ with HMM enabled. Can be verified via `nvidia-smi -q | grep Addressing` showing `HMM`.

Characteristics:
- ALL system-allocated memory (`malloc`, `mmap`, file-backed mappings) automatically behaves as unified memory. No CUDA allocation APIs are required.
- Migration happens via page faults at page granularity (software coherence).
- Oversubscription is allowed.
- `cudaMallocManaged` still works but is unnecessary for basic access -- `malloc` suffices.
- GPU `cudaMalloc` allocations are NOT CPU-accessible (unlike bidirectional ATS).

### Paradigm 4: Full System-Memory Access with Host Page Tables (ATS)

**Detection:** `cudaDevAttrPageableMemoryAccessUsesHostPageTables == 1` AND `cudaDevAttrPageableMemoryAccess == 1` AND `cudaDevAttrConcurrentManagedAccess == 1`

Available on Grace Hopper, Grace Blackwell (including DGX Spark), Jetson Thor, and future systems where CUDA reports pageable memory access through host page tables. `nvidia-smi -q` reports these systems as `Addressing Mode: ATS`.

Characteristics:
- ALL system-allocated memory is GPU-accessible (same as HMM).
- GPU `cudaMalloc` allocations are CPU-accessible only when `cudaDevAttrDirectManagedMemAccessFromHost == 1`. This is true on Grace Hopper / Grace Blackwell systems, but false on Jetson Thor as tested with CUDA 13.0.
- Native CPU-GPU atomics work when `cudaDevAttrHostNativeAtomicSupported == 1`. This is a separate capability bit and does not imply CPU access to `cudaMalloc` allocations.
- Host page tables are used for system-memory access. On systems with distinct CPU and GPU memory pools (Grace Hopper / Grace Blackwell), physical placement still matters for performance. On integrated SoCs such as Jetson Thor, the CPU and GPU share a single DRAM pool.
- ATS subsumes the system-memory access capabilities of HMM. When ATS is available, HMM is automatically disabled.

#### Observed Jetson Thor Behavior

The previous version of this document speculated that Jetson Thor would follow the limited Tegra model. Testing on a Jetson Thor development kit on 2026-05-11 showed otherwise:

- Platform: Linux `6.8.12-tegra`, CUDA Toolkit 13.0, Driver 13.0, GPU `NVIDIA Thor`, `sm_110`.
- `nvidia-smi -q` reports `Addressing Mode: ATS`.
- CUDA attributes: `integrated == 1`, `unifiedAddressing == 1`, `managedMemory == 1`, `concurrentManagedAccess == 1`, `pageableMemoryAccess == 1`, `pageableMemoryAccessUsesHostPageTables == 1`, `directManagedMemAccessFromHost == 0`, `hostNativeAtomicSupported == 1`, `canUseHostPointerForRegisteredMem == 1`.
- CUDA kernels successfully read and wrote ordinary `malloc`, anonymous `mmap`, file-backed `mmap`, `cudaMallocHost`, `cudaHostRegister`, and `cudaMallocManaged` allocations.
- `cudaMemPrefetchAsync` succeeded for both managed memory and ordinary `malloc` memory.
- Direct CPU load/store of a `cudaMalloc` pointer faulted, matching `directManagedMemAccessFromHost == 0`.
- A stress test with overlapping CPU atomic increments and GPU `atomicAdd()` produced the exact expected result for ordinary `malloc`, pinned host memory, and managed memory.

The implementation must therefore treat "GPU can access system memory", "CPU can access GPU allocations", and "native CPU-GPU atomics work" as three independent capabilities.

### Summary of Access Rules by Paradigm

| Allocation type | Limited (Tegra/Win) | Full Managed Only | HMM (Software) | ATS system-memory only (Thor) | ATS bidirectional (Grace/GB) |
|---|---|---|---|---|---|
| `malloc` / system | CPU only | CPU only | CPU + GPU | CPU + GPU | CPU + GPU |
| `mmap` / file-backed | CPU only | CPU only | CPU + GPU | CPU + GPU | CPU + GPU |
| `cudaMallocManaged` | Limited shared | Full shared | Full shared | Full shared | Full shared |
| `cudaMallocHost` (pinned) | CPU + GPU (zero-copy) | CPU + GPU | CPU + GPU | CPU + GPU | CPU + GPU |
| `cudaHostRegister` | Device-dependent | CPU + GPU | CPU + GPU | CPU + GPU | CPU + GPU |
| `cudaMalloc` | GPU only | GPU only | GPU only | GPU only | **CPU + GPU** |

### Performance Considerations on ATS Systems

Even when all system memory is GPU-accessible on ATS systems, physical placement can still matter for performance. On systems with distinct CPU and GPU memory pools, a GPU kernel repeatedly reading data physically resident in CPU LPDDR5X over NVLink C2C pays the C2C latency on every cache miss. CUDA provides mechanisms to control placement:

1. **Explicit prefetch** (`cudaMemPrefetchAsync`): Stream-ordered migration of a memory region to a specified device. Works on any allocation including system `malloc`. This is the primary tool for optimizing data placement.

2. **Access counter migration**: On ATS systems, the GPU hardware tracks access frequency to remote pages. When enabled via `cudaMemAdviseSetAccessedBy`, pages that the GPU accesses frequently are automatically migrated to GPU-local memory. Available for system-allocated memory starting with CUDA 12.4. Does not apply to file-backed `mmap` allocations.

3. **Placement hints** (`cudaMemAdvise`):
   - `cudaMemAdviseSetPreferredLocation(device)` -- encourages data to stay on the specified device.
   - `cudaMemAdviseSetReadMostly` -- allows read replication across devices.
   - `cudaMemAdviseSetAccessedBy(device)` -- enables access counter migration on ATS systems; establishes direct mappings on other systems.

On integrated ATS systems such as Jetson Thor, CPU and GPU memory share one DRAM pool, so prefetch may still succeed but may not provide a useful "closer" placement. Automatic prefetch should therefore remain disabled on integrated GPUs.

**Important performance caveat**: On bidirectional ATS systems, the CUDA documentation warns against frequent CPU writes to GPU-resident memory. ARM (Grace) caches require all memory operations to pass through the cache hierarchy, so writing to GPU-resident memory causes cache misses that pull data across C2C before writing. The recommended pattern is: write to CPU-resident memory, let the GPU read it remotely or prefetch it.

### Comparison: DGX Spark vs. Jetson Thor

Both DGX Spark and Jetson Thor use Blackwell GPUs, but their memory architectures differ fundamentally:

| Aspect | DGX Spark (Grace Blackwell) | Jetson Thor (Tegra Blackwell) |
|---|---|---|
| CPU-GPU interconnect | NVLink C2C (high bandwidth, coherent) | On-chip SoC fabric |
| ATS available | Yes | Yes (`nvidia-smi` reports ATS) |
| Coherency model | Host-page-table ATS with distinct CPU/GPU memory pools | Host-page-table ATS for system memory on an integrated SoC |
| `malloc` GPU-accessible | Yes | Yes |
| `cudaMalloc` CPU-accessible | Yes | No |
| Native CPU-GPU atomics | Yes | Yes for host-visible memory |
| Memory topology | Grace LPDDR5X + Blackwell HBM (NUMA) | Single shared DRAM pool |
| Unified memory paradigm | ATS bidirectional (Paradigm 4) | ATS system-memory only (Paradigm 4) |
| Best default allocator | System allocator (`malloc`) for shared CPU/GPU data | System allocator (`malloc`) for CPU-produced GPU-readable data; `cudaMalloc` for GPU-private data |

This means the implementation must query capabilities independently instead of assuming a single "ATS" behavior. Jetson Thor can launch CUDA kernels directly over CPU arrays, but CPU kernels still cannot dereference `cudaMalloc` arrays.

## Requirements

| ID  | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| R1 | `wp.launch()` must remove the per-argument device check so cross-device array arguments are always passed through to the hardware | Must | Core behavior change, zero launch overhead |
| R2 | Provide an opt-in verification mode (`warp.config.verify_launch`) that restores device-access checking with clear diagnostics | Must | Debuggability for users who hit CUDA illegal memory access errors |
| R3 | Provide `wp.prefetch()` API for explicit data migration hints | Should | Performance optimization for HMM / host-page-table ATS |
| R4 | Optional automatic prefetch in `wp.launch()` for cross-device arrays on coherent systems | Could | Convenience, but needs careful defaults |
| R5 | `wp.copy()` should skip staging buffers when direct access is available between devices | Could | Performance optimization, marked as TODO in current code |

**Non-goals:**
- Changing the default allocator strategy (e.g., using `cudaMallocManaged` by default on limited Tegra systems). Allocator selection is a separate concern.
- Supporting cross-device access for CUDA graph capture. Graph capture has additional constraints and should be addressed separately.
- Automatically determining the optimal physical placement for every array. This is a performance tuning concern best left to the user via hints.
- Proactively detecting and warning about cross-device launches at `wp.launch()` time. The hardware enforces access rules; the verification mode is available for diagnosis when needed.

## Design

### CUDA Version Compatibility

Warp currently supports building with CUDA 12.0 through 13.2. The default toolkit distributed on PyPI is CUDA 12.9. Full support for this feature set on CUDA 12.0 through 12.7 is not a goal, but the feature must degrade cleanly (compile without errors, disable gracefully at runtime) on older toolkit versions.

**Device attribute queries (Phases 1, 2, 3, 5):** All device attributes used in this plan are `CUdevice_attribute` enum values present in `cuda.h` since CUDA 8.0 or 9.2 at the latest:

| Attribute | Enum value | Present since | Phase |
|---|---|---|---|
| `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS` | 88 | CUDA 8.0 | 1 |
| `CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST` | 101 | CUDA 9.2 | 1 |
| `CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED` | 86 | CUDA 8.0 | 1 |
| `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES` | 100 | CUDA 9.2 | 2 |
| `CU_DEVICE_ATTRIBUTE_INTEGRATED` | 18 | CUDA 2.0 | 3 |
| `CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS` | 89 | CUDA 8.0 | 5 |

All predate Warp's minimum of CUDA 12.0, so no `#if CUDA_VERSION` compile-time guards are needed for attribute queries. The attributes are queried via `cuDeviceGetAttribute`, which Warp already loads dynamically via `cuGetProcAddress` at version 2000 (`cuda_util.cpp:228`). The driver returns 0 for any attribute the hardware does not support, which is the correct "feature not available" default.

**`cuMemPrefetchAsync` (Phase 2):** This driver API has two versions:

| API version | Signature | Toolkit requirement | Driver requirement |
|---|---|---|---|
| v1 (version 8000) | `(CUdeviceptr, size_t, CUdevice, CUstream)` | CUDA 8.0+ | CUDA 8.0+ driver |
| v2 (version 12080) | `(CUdeviceptr, size_t, CUmemLocation, unsigned int, CUstream)` | CUDA 12.8+ | CUDA 12.8+ driver |

In CUDA 13.0 headers, `cuMemPrefetchAsync` is `#define`'d to `cuMemPrefetchAsync_v2`. Warp must handle both via `cuGetProcAddress` dynamic dispatch, following the existing pattern used for `cuMemcpyBatchAsync` (`cuda_util.cpp:234`). The v1 API is sufficient for all planned use cases. The v2 API adds NUMA node targeting but is not required. When compiled with CUDA 12.0--12.7, only v1 is available; this is fine. See Phase 2 for the full dispatch implementation.

**Summary by toolkit version:**

| Feature | CUDA 12.0 -- 12.7 | CUDA 12.8 -- 12.9 (PyPI default) | CUDA 13.0+ |
|---|---|---|---|
| Phase 1 (cross-device launch) | Full support | Full support | Full support |
| Phase 2 (prefetch) | v1 API only | v2 API available | v2 API available |
| Phase 3 (auto-prefetch) | Full support (uses Phase 2 API) | Full support | Full support |
| Phase 4 (`wp.copy()` optimization) | Full support | Full support | Full support |
| Phase 5 (allocator awareness) | Full support | Full support | Full support |

No phase requires a minimum toolkit version beyond CUDA 12.0. Degradation on older toolkits only affects which `cuMemPrefetchAsync` signature is available, which is handled transparently by the dynamic dispatch.

### Overview: What Each Phase Introduces

Each phase introduces only the device attributes, native functions, and Python APIs that it directly consumes. No phase adds speculative API surface for a future phase to use.

| Phase | What it delivers | Attributes introduced | Native functions introduced |
|---|---|---|---|
| 1 | Remove device check from `wp.launch()`, add verification mode, redesign `can_access()` | `pageable_memory_access`, `direct_managed_mem_access_from_host`, `host_native_atomic_supported` | Three `wp_cuda_device_get_*` accessors |
| 2 | `wp.prefetch()` for explicit data placement | `pageable_memory_access_uses_host_page_tables` (to distinguish HMM from host-page-table ATS for warning/no-op behavior) | `wp_cuda_mem_prefetch_async` |
| 3 | Auto-prefetch in `wp.launch()` | `is_integrated` (to avoid pointless prefetches on shared-DRAM SoCs) | None |
| 4 | `wp.copy()` staging-buffer optimization | None (reuses `can_access()` from Phase 1) | None |
| 5 | Allocator-aware fine-grained access checks | `concurrent_managed_access` (to distinguish limited vs. full managed memory) | None |

### Phase 1: Cross-Device Launch Support

**Goal:** Remove the per-argument device check from `wp.launch()` so that cross-device array arguments are passed straight through to the hardware. On systems with unified system-memory access (HMM or host-page-table ATS), this means CUDA kernels can directly consume CPU arrays with zero launch overhead and zero friction. On systems where the access is illegal, the CUDA runtime produces an error. A verification mode (`warp.config.verify_launch`) is available to diagnose such errors with clear, argument-level diagnostics before the kernel runs.

This phase delivers four things: (a) query three new device attributes, (b) redesign `Device.can_access()`, (c) remove the `pack_arg()` device check (with opt-in verification), (d) add a config flag and verification logic.

#### 1a. Query Device Attributes

Three CUDA device attributes are needed:

- **`CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS`** -- answers "can this GPU access ordinary `malloc`'d host memory?" This is the attribute that determines whether a Warp `wp.array(device="cpu")` (backed by `malloc` via `CpuDefaultAllocator`) can be dereferenced by a CUDA kernel. Without it, we cannot distinguish a system where the GPU can read CPU pointers (HMM, host-page-table ATS, Jetson Thor) from one where it cannot (discrete GPU without HMM, limited Tegra, Windows).

- **`CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST`** -- answers "can the CPU read/write GPU device memory?" This is the attribute that determines whether a Warp `wp.array(device="cuda:0")` (backed by `cuMemAlloc` via `CudaDefaultAllocator`) can be passed to a CPU kernel. Without it, we cannot distinguish bidirectional ATS systems (Grace Hopper / Grace Blackwell) from HMM or Jetson Thor systems where GPU-to-CPU-array access works but CPU-to-`cudaMalloc` access does not.

- **`CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED`** -- answers "do CPU-GPU atomics work natively across the interconnect?" On systems where this is true (Grace Hopper, Grace Blackwell, and Jetson Thor as tested), a GPU `atomicAdd` targeting a CPU-resident address produces correct results via hardware coherency. On HMM systems, the same operation can silently produce wrong results -- the GPU atomic hits a page backed by CPU physical memory without hardware coherency for atomic operations. Exposing this as a device property lets users and downstream tools (e.g., documentation, `wp.prefetch()` heuristics) reason about atomic safety. This attribute must be treated independently from `direct_managed_mem_access_from_host`.

The first two attributes are the minimum needed because `can_access()` has exactly two cross-device branches that need gating: GPU-accessing-CPU and CPU-accessing-GPU. Each branch needs exactly one attribute. The third is exposed as a queryable device property for users who need to know whether cross-device atomics are safe on their system.

**Native layer changes (`warp/native/warp.cu`, `warp/native/warp.h`)**

Add three fields to `DeviceInfo` (currently at `warp.cu:136`):

```cpp
struct DeviceInfo {
    // ... existing fields ...
    int pageable_memory_access = 0;
    int direct_managed_mem_access_from_host = 0;
    int host_native_atomic_supported = 0;
};
```

Query them during device enumeration (currently at `warp.cu:270`), alongside the existing `cuDeviceGetAttribute` calls:

```cpp
check_cu(cuDeviceGetAttribute_f(
    &g_devices[i].pageable_memory_access,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, device));
check_cu(cuDeviceGetAttribute_f(
    &g_devices[i].direct_managed_mem_access_from_host,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST, device));
check_cu(cuDeviceGetAttribute_f(
    &g_devices[i].host_native_atomic_supported,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, device));
```

**CUDA version requirements:** All three attributes are enum values in `CUdevice_attribute` that have been present since well before CUDA 12.0 (Warp's minimum supported CUDA toolkit version): `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS` (= 88, CUDA 8.0+), `CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST` (= 101, CUDA 9.2+), `CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED` (= 86, CUDA 8.0+). No `#if CUDA_VERSION` guard is needed. They are queried via `cuDeviceGetAttribute`, which Warp already loads dynamically via `cuGetProcAddress` (see `cuda_util.cpp:228`, version 2000). The driver will return 0 for any attribute the hardware does not support, which is the correct default (feature not available).

Add accessor functions (following the existing pattern of `wp_cuda_device_is_uva()` at `warp.cu:1980`):

```cpp
// warp.h
WP_API int wp_cuda_device_get_pageable_memory_access(int ordinal);
WP_API int wp_cuda_device_get_direct_managed_mem_access_from_host(int ordinal);
WP_API int wp_cuda_device_get_host_native_atomic_supported(int ordinal);

// warp.cu
int wp_cuda_device_get_pageable_memory_access(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].pageable_memory_access;
    return 0;
}

int wp_cuda_device_get_direct_managed_mem_access_from_host(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].direct_managed_mem_access_from_host;
    return 0;
}

int wp_cuda_device_get_host_native_atomic_supported(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].host_native_atomic_supported;
    return 0;
}
```

**Python layer changes (`warp/_src/context.py`)**

Register the new ctypes bindings in `Runtime.__init__` (following the pattern at `context.py:4666`):

```python
self.core.wp_cuda_device_get_pageable_memory_access.argtypes = [ctypes.c_int]
self.core.wp_cuda_device_get_pageable_memory_access.restype = ctypes.c_int
self.core.wp_cuda_device_get_direct_managed_mem_access_from_host.argtypes = [ctypes.c_int]
self.core.wp_cuda_device_get_direct_managed_mem_access_from_host.restype = ctypes.c_int
self.core.wp_cuda_device_get_host_native_atomic_supported.argtypes = [ctypes.c_int]
self.core.wp_cuda_device_get_host_native_atomic_supported.restype = ctypes.c_int
```

Add properties to `Device.__init__` for CUDA devices (following the pattern at `context.py:3541`):

```python
# Unified memory capability attributes
self.pageable_memory_access = (
    runtime.core.wp_cuda_device_get_pageable_memory_access(ordinal) > 0
)
self.direct_managed_mem_access_from_host = (
    runtime.core.wp_cuda_device_get_direct_managed_mem_access_from_host(ordinal) > 0
)
self.host_native_atomic_supported = (
    runtime.core.wp_cuda_device_get_host_native_atomic_supported(ordinal) > 0
)
```

For the CPU device (at `context.py:3520`), set all three to `False`.

Add derived convenience properties:

```python
@property
def can_access_host_memory(self):
    """Whether this GPU can directly access all host (CPU) memory.

    True on HMM and host-page-table ATS systems, including Jetson Thor.
    False on discrete GPU systems without HMM and on limited Tegra /
    Windows systems.
    """
    return self.pageable_memory_access

@property
def is_host_accessible(self):
    """Whether ``cuMemAlloc`` (device) allocations on this device can be
    read and written by the CPU.

    True only when ``cudaDevAttrDirectManagedMemAccessFromHost`` is true
    (for example Grace Hopper and Grace Blackwell / DGX Spark). This is
    false on Jetson Thor even though Thor reports ATS for system memory.
    """
    return self.direct_managed_mem_access_from_host
```

#### 1b. Redesign `Device.can_access()`

The current implementation (at `context.py:3788`) has a TODO acknowledging it needs redesign:

```python
def can_access(self, other):
    # TODO: this function should be redesigned in terms of (device, resource).
    # - a device can access any resource on the same device
    # - a CUDA device can access pinned memory on the host
    # - a CUDA device can access regular allocations on a peer device if peer access is enabled
    # - a CUDA device can access mempool allocations on a peer device if mempool access is enabled
    other = self.runtime.get_device(other)
    if self.context == other.context:
        return True
    else:
        return False
```

Replace with:

```python
def can_access(self, other):
    """Check if a kernel running on this device can access memory on ``other``.

    This considers same-device access, unified memory capabilities (HMM
    and host-page-table ATS), and CUDA peer access between GPUs. The
    check is conservative:
    it returns ``True`` only when all standard allocations on ``other``
    are accessible from kernels on ``self``.

    For allocation types it cannot distinguish (e.g., pinned vs. unpinned
    CPU memory), it assumes the less-accessible type. Callers that know
    the allocation is pinned can bypass this check.

    Args:
        other: The device (or device-like identifier) where the memory
            resides.

    Returns:
        ``True`` if all standard allocations on ``other`` are accessible
        from kernels on ``self``.
    """
    other = self.runtime.get_device(other)

    # Same device -- always accessible.
    if self == other:
        return True

    # GPU accessing CPU memory.
    if self.is_cuda and other.is_cpu:
        # HMM or host-page-table ATS: all system memory is GPU-accessible.
        if self.can_access_host_memory:
            return True
        # Without HMM/host-page-table ATS, only pinned allocations are
        # accessible, but we cannot distinguish pinned from unpinned
        # here, so return False.
        return False

    # CPU accessing GPU memory.
    if self.is_cpu and other.is_cuda:
        # Direct managed memory access: cuMemAlloc memory is CPU-accessible.
        if other.is_host_accessible:
            return True
        return False

    # GPU-to-GPU.
    if self.is_cuda and other.is_cuda:
        # Same CUDA context (can happen with context sharing).
        if self.context == other.context:
            return True
        # Peer access.
        return is_peer_access_enabled(other, self)

    # CPU-to-CPU is always True (single address space).
    if self.is_cpu and other.is_cpu:
        return True

    return False
```

**Notes on `can_access()` implementation details:**

- The `is_peer_access_enabled()` call in the GPU-to-GPU branch is the module-level function at `warp/_src/context.py:6002`. It lives in the same module as `Device`, so no additional import is needed.
- `Device.__eq__` (at `context.py:3771`) compares by CUDA context pointer when both operands are `Device` instances, so `self == other` is equivalent to `self.context == other.context` for two `Device` objects. The `self == other` check at the top and the `self.context == other.context` fallback in the GPU-to-GPU branch are not redundant — the top check catches same-alias devices efficiently; the GPU-to-GPU fallback catches the edge case of two different `Device` objects sharing a CUDA context (context sharing).
- The TODO in the existing code mentions that access depends on the _resource_ (allocation type), not just the device pair. A fully precise check would need to know whether a specific allocation is pinned, managed, or default. This design intentionally makes a conservative device-level check: it returns `True` only when _all standard allocations_ on the other device are accessible. This avoids needing to thread allocation metadata through every call site. Phase 5 (allocator awareness) addresses the resource-level refinement.
- `can_access()` is NOT called in the default launch path. It is only invoked when `warp.config.verify_launch` is `True` (diagnostic mode) or by other APIs (`wp.copy()`, `wp.prefetch()`, user code). In the verification path, the HMM / host-page-table ATS branches are cheap property lookups. The GPU-to-GPU peer branch calls `is_peer_access_enabled()` which goes through ctypes into the native layer. If profiling shows this matters in verification mode, the result could be cached per `(self, other)` device pair and invalidated when `set_peer_access_enabled()` is called.

#### 1c. Remove the Launch Device Check

Change `pack_arg()` at `context.py:6808`.

**Scope:** This change only removes the device check for `array` arguments. Other device-bound types (textures, volumes, hash grids) have their own device checks later in `pack_arg()` (e.g., `texture1d_t` at `context.py:6886`) and remain strict (`value.device != device`). Relaxing those is out of scope for Phase 1: textures and volumes have GPU-side handles (CUDA texture objects, device pointers to internal structures) that may not be accessible cross-device even on systems with ATS system-memory access.

The `pack_arg()` function is called for both forward and adjoint arguments (see `pack_args()` at `context.py:7307`, which processes forward args and then adjoint args with `adjoint=True`). The removed check applies to both paths, so cross-device arrays work in backward passes on capable hardware.

Replace:

```python
# check device
if value.device != device:
    raise RuntimeError(
        f"Error launching kernel '{kernel.key}', trying to launch on "
        f"device='{device}', but input array for argument '{arg_name}' "
        f"is on device={value.device}."
    )
```

With:

```python
# Verify device accessibility (opt-in diagnostic mode).
# By default, no check is performed and the pointer is passed
# straight through to the hardware.  On systems with unified
# system-memory access (HMM or host-page-table ATS) this is correct;
# on discrete GPUs without
# HMM the kernel will fault with CUDA_ERROR_ILLEGAL_ADDRESS if the
# access is invalid.  Enable warp.config.verify_launch to get a
# clear Python error *before* the kernel runs.
if warp.config.verify_launch and value.device != device:
    if not device.can_access(value.device):
        raise RuntimeError(
            f"Error launching kernel '{kernel.key}', trying to "
            f"launch on device='{device}', but input array for "
            f"argument '{arg_name}' is on device={value.device} "
            f"which is not accessible from '{device}'. Disable "
            f"warp.config.verify_launch to skip this check."
        )
```

**Design rationale:** The previous design called `can_access()` on every array argument of every launch. Even though `can_access()` is cheap (property lookups), it adds up in hot launch paths with many array arguments. Removing the check entirely by default means `pack_arg()` does strictly less work than it does today (the old `value.device != device` comparison is gone). The verification mode gates the check behind a single boolean test, which the branch predictor will eliminate after the first few calls.

#### 1d. Verification Mode Config Flag

Add to `warp/config.py`:

```python
verify_launch = False
"""When True, wp.launch() checks that every array argument is accessible
from the launch device before running the kernel.  When False (the
default), array pointers are passed through to the hardware without
validation.  Enable this to diagnose CUDA_ERROR_ILLEGAL_ADDRESS errors
with clear, per-argument diagnostics."""
```

**When to use:** If a user on a discrete GPU (without HMM) accidentally passes a CPU array to a CUDA kernel, the kernel will fault with `CUDA_ERROR_ILLEGAL_ADDRESS`. This error is asynchronous and can corrupt the CUDA context, requiring a process restart. The recommended workflow is:

1. Observe the CUDA error.
2. Set `warp.config.verify_launch = True`.
3. Re-run. The clear Python `RuntimeError` identifies which kernel and which argument caused the mismatch, before the kernel ever launches.
4. Fix the code, disable verification.

#### Behavior matrix after Phase 1

Default mode (`verify_launch = False`): no Python-level checking. The hardware decides.

| Launch device | Array device | Discrete GPU (no HMM) | HMM system | Jetson Thor | Bidirectional ATS (DGX Spark) |
|---|---|---|---|---|---|
| `cuda:0` | `cuda:0` | OK (same device) | OK | OK | OK |
| `cuda:0` | `cpu` | **CUDA fault** | **OK** (HMM) | **OK** (ATS system memory) | **OK** (ATS) |
| `cpu` | `cuda:0` | **Segfault** | **Segfault** | **Segfault** | **OK** (direct managed access) |
| `cuda:0` | `cuda:1` | CUDA fault / OK (peer) | CUDA fault / OK (peer) | N/A on single-GPU Thor | CUDA fault / OK (peer) |

Verification mode (`verify_launch = True`): `can_access()` is checked per argument.

| Launch device | Array device | Discrete GPU (no HMM) | HMM system | Jetson Thor | Bidirectional ATS (DGX Spark) |
|---|---|---|---|---|---|
| `cuda:0` | `cuda:0` | OK (same device) | OK | OK | OK |
| `cuda:0` | `cpu` | **RuntimeError** | **OK** (HMM) | **OK** (ATS system memory) | **OK** (ATS) |
| `cpu` | `cuda:0` | **RuntimeError** | **RuntimeError** | **RuntimeError** | **OK** (direct managed access) |
| `cuda:0` | `cuda:1` | RuntimeError / OK (peer) | RuntimeError / OK (peer) | N/A on single-GPU Thor | RuntimeError / OK (peer) |

On a standard discrete-GPU workstation without HMM, users who pass a CPU array to a CUDA kernel will get a CUDA fault instead of the current Python `RuntimeError`. This is a deliberate tradeoff: zero overhead in the launch path for all users, at the cost of a less friendly error for an incorrect program. The verification mode restores the friendly error for diagnosis.

#### Stream selection for cross-device launches

When an array on device A is passed to a kernel on device B, Warp must ensure proper synchronization. The current `wp.launch()` already selects a stream based on the launch device (at `context.py:7282`). The kernel launch happens on that stream, and since the pointer is passed through without checking, the hardware coherency or HMM page fault mechanism handles visibility on capable systems.

However, if the array was _produced_ by a kernel on a different stream, the caller is responsible for synchronizing (e.g., via `wp.synchronize()` or stream events). This is the same requirement as for same-device multi-stream usage and does not need special handling here.

### Phase 2: Explicit Prefetch API (`wp.prefetch()`)

**Goal:** Provide a public API for users to request migration of array data to a specific device, without copying. This is a performance optimization for HMM and ATS systems where data is accessible across processors but performance can depend on physical placement.

This phase introduces one additional device attribute and one new native function.

#### New device attribute: `pageable_memory_access_uses_host_page_tables`

**CUDA attribute:** `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES`

This attribute distinguishes HMM (software coherency) from host-page-table ATS. Both have `pageable_memory_access == 1`, but prefetch behavior differs:
- On host-page-table ATS with distinct CPU/GPU memory pools, prefetch can migrate physical pages via hardware DMA with cache-line coherency. On integrated systems such as Jetson Thor, prefetch may succeed but may not improve placement because CPU and GPU share the same DRAM.
- On HMM, prefetch triggers software page migration with TLB shootdowns. It works but has higher overhead and different failure modes.

The `wp.prefetch()` implementation needs this to provide accurate diagnostics (e.g., warning when prefetching on a system where it may cause page-fault storms) and to choose the right native API call path.

**Native layer:** Same pattern as Phase 1 -- add to `DeviceInfo`, query during enumeration, add accessor function. `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES` (= 100) has been present since CUDA 9.2, well before Warp's minimum of CUDA 12.0. No compile-time guard needed.

```cpp
// DeviceInfo addition
int pageable_memory_access_uses_host_page_tables = 0;

// Query
check_cu(cuDeviceGetAttribute_f(
    &g_devices[i].pageable_memory_access_uses_host_page_tables,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, device));

// Accessor
WP_API int wp_cuda_device_get_pageable_memory_access_uses_host_page_tables(int ordinal);
```

**Python layer:**

```python
self.pageable_memory_access_uses_host_page_tables = (
    runtime.core.wp_cuda_device_get_pageable_memory_access_uses_host_page_tables(ordinal) > 0
)
```

#### New native function: `wp_cuda_mem_prefetch_async`

```cpp
// warp.h
WP_API int wp_cuda_mem_prefetch_async(void* ptr, size_t size_in_bytes,
                                       int device_ordinal, void* stream);
```

This wraps `cuMemPrefetchAsync` (driver API). The `device_ordinal` can be `-1` to indicate the CPU as the target (maps to `CU_DEVICE_CPU` in the driver API).

**CUDA API versioning:** The `cuMemPrefetchAsync` driver API has two versions:

- **v1** (CUDA 8.0+, version 8000): `cuMemPrefetchAsync(CUdeviceptr, size_t, CUdevice dstDevice, CUstream)` -- takes a simple `CUdevice` ordinal for the destination.
- **v2** (CUDA 12.8+, version 12080): `cuMemPrefetchAsync(CUdeviceptr, size_t, CUmemLocation location, unsigned int flags, CUstream)` -- takes a `CUmemLocation` struct (supports NUMA node targeting) and flags.

In CUDA 13.0 headers, `cuMemPrefetchAsync` is `#define`'d to `cuMemPrefetchAsync_v2`. Warp dynamically loads driver entry points via `cuGetProcAddress` (see `cuda_util.cpp:150`), so the implementation must handle both versions:

```cpp
// In init_cuda_driver(), load the prefetch entry point:
#if CUDA_VERSION >= 12080
if (driver_version >= 12080)
    get_driver_entry_point("cuMemPrefetchAsync", 12080, &(void*&)pfn_cuMemPrefetchAsync_v2);
else
#endif
    get_driver_entry_point("cuMemPrefetchAsync", 8000, &(void*&)pfn_cuMemPrefetchAsync_v1);
```

The `wp_cuda_mem_prefetch_async` wrapper dispatches to whichever version was loaded:

```cpp
int wp_cuda_mem_prefetch_async(void* ptr, size_t size_in_bytes,
                                int device_ordinal, void* stream)
{
    CUdeviceptr devPtr = (CUdeviceptr)ptr;
    CUstream hStream = (CUstream)stream;

#if CUDA_VERSION >= 12080
    if (pfn_cuMemPrefetchAsync_v2) {
        CUmemLocation location;
        if (device_ordinal >= 0) {
            location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            location.id = device_ordinal;
        } else {
            location.type = CU_MEM_LOCATION_TYPE_HOST;
            location.id = 0;
        }
        return check_cu(pfn_cuMemPrefetchAsync_v2(devPtr, size_in_bytes,
                                                    location, 0, hStream)) ? 0 : -1;
    }
#endif
    if (pfn_cuMemPrefetchAsync_v1) {
        CUdevice dstDevice = (device_ordinal >= 0)
            ? g_devices[device_ordinal].device
            : CU_DEVICE_CPU;
        return check_cu(pfn_cuMemPrefetchAsync_v1(devPtr, size_in_bytes,
                                                    dstDevice, hStream)) ? 0 : -1;
    }
    return -1;  // prefetch not available
}
```

This pattern follows the existing convention in `cuda_util.cpp` (e.g., `cuMemcpyBatchAsync` at line 234 uses the same `#if CUDA_VERSION >= 12080` / `driver_version >= 12080` pattern).

**Compile-time / runtime compatibility matrix for Phase 2:**

| Toolkit used to build Warp | Runtime driver | Prefetch available? | API version used |
|---|---|---|---|
| CUDA 12.0 -- 12.7 | Any 12.0+ | Yes | v1 (CUdevice) |
| CUDA 12.8+ | Driver < 12.8 | Yes | v1 (CUdevice) |
| CUDA 12.8+ | Driver >= 12.8 | Yes | v2 (CUmemLocation) |

The v1 API is fully sufficient for the `wp.prefetch()` use case (migrate to a device or to the CPU). The v2 API adds NUMA node targeting which is not needed initially but is available when both toolkit and driver support it.

**Disabling prefetch on older CUDA:** If Warp is compiled with CUDA 12.0 -- 12.7, only the v1 entry point is loaded. The v1 API works for `cudaMallocManaged` allocations on all systems, and also for system-allocated (`malloc`) memory on HMM / host-page-table ATS systems. The Python `wp.prefetch()` wrapper should catch errors from the driver (e.g., if the pointer is not in a prefetchable region) and emit a warning rather than raising, since prefetch is a performance hint.

Implementation notes:
- `cuMemPrefetchAsync` works on any pointer that falls within a unified memory region -- including plain `malloc` on HMM / host-page-table ATS systems, `cuMemAllocManaged` allocations, and `cuMemAlloc` allocations on systems where device allocations are host-accessible.
- On systems where the pointer is not in a prefetchable region, the call returns an error. The Python wrapper should catch this and either warn or silently ignore, since prefetch is a hint.
- The prefetch is stream-ordered: it begins after all prior operations on the stream complete and finishes before any subsequent operations on the stream begin.

#### Python API

```python
def prefetch(
    array: warp.array,
    device: DeviceLike = None,
    stream: Stream | None = None,
):
    """Request asynchronous migration of ``array`` data toward ``device``.

    On systems with host-page-table ATS or software coherency (HMM),
    this issues a ``cuMemPrefetchAsync`` to migrate the
    array's physical pages closer to the specified device. The array
    remains valid and accessible from any device during and after the
    prefetch.

    On systems without unified memory support for the array's allocation
    type, this function is a no-op and emits a warning.

    This is a performance hint, not a correctness requirement. Kernels
    will produce correct results regardless of whether prefetch is
    called.

    Args:
        array: The array whose data should be migrated.
        device: The target device. If ``None``, uses the default device.
        stream: The stream on which to order the prefetch. If ``None``,
            uses the current stream on the target device.
    """
```

#### Usage example

```python
# On DGX Spark (bidirectional ATS system):
data = wp.array(np.random.randn(1000000), dtype=wp.float32, device="cpu")

# Prefetch to GPU before a compute-heavy kernel
wp.prefetch(data, device="cuda:0")
wp.launch(heavy_compute_kernel, dim=data.size, inputs=[data], device="cuda:0")

# Prefetch back to CPU before CPU-side post-processing
wp.prefetch(data, device="cpu")
result = data.numpy()
```

### Phase 3: Optional Automatic Prefetch in `wp.launch()` (Future)

**Goal:** When a cross-device array argument is detected in `pack_arg()` on a coherent system, optionally issue a prefetch automatically before the kernel launch. This is a convenience optimization that should be off by default.

This phase introduces one additional device attribute.

#### New device attribute: `is_integrated`

**CUDA attribute:** `CU_DEVICE_ATTRIBUTE_INTEGRATED`

This attribute indicates whether the GPU is physically integrated into the same chip/package as the CPU (Tegra/Jetson SoCs). It is already queried in the native layer (`warp.cu:295`) but stored in a local variable `device_attribute_integrated` used only for the IPC check. This phase promotes it to a stored `DeviceInfo` field exposed to Python.

The auto-prefetch heuristic needs this because prefetch on an integrated GPU is usually pointless -- the CPU and GPU share the same physical DRAM, so there is no "closer" location to migrate data to. Jetson Thor testing showed that `cuMemPrefetchAsync` can succeed for ordinary `malloc` memory, but that does not make automatic prefetch useful. Without this attribute, the auto-prefetch code would waste time issuing low-value prefetch calls on every integrated-GPU kernel launch with cross-device arrays.

#### Config flag

Add to `warp/config.py`:

```python
auto_prefetch = False
"""When True and launching a kernel on a device that can access memory on
another device (e.g., GPU accessing CPU memory on an HMM or ATS system),
automatically prefetch cross-device array arguments to the launch device
before the kernel begins. Default is False because automatic prefetch is
not always beneficial -- for example, streaming read-once access patterns
are better served by remote access over NVLink C2C than by migrating
the data."""
```

#### Implementation in `pack_arg()`

After the device accessibility check passes (Phase 1), and before returning the packed argument:

```python
if value.device != device and warp.config.auto_prefetch:
    # Skip prefetch on integrated GPUs -- CPU and GPU share the same
    # DRAM, so migration is meaningless.
    if not device.is_integrated:
        try:
            stream_handle = device.stream.cuda_stream if device.is_cuda else 0
            device_ordinal = device.ordinal if device.is_cuda else -1
            runtime.core.wp_cuda_mem_prefetch_async(
                value.ptr, value.capacity, device_ordinal, stream_handle
            )
        except Exception:
            pass  # Prefetch is best-effort
```

#### Why off by default

Automatic prefetch has several cases where it hurts more than it helps:

1. **Read-once data**: If a kernel reads an array once and never again, prefetching (which may involve a DMA transfer or page-table work) can be slower than direct access.
2. **CPU-produced, GPU-consumed streaming data**: If the CPU is continuously writing to a buffer that the GPU reads, prefetching would fight with the CPU's writes. The CUDA documentation explicitly recommends keeping such data CPU-resident and letting the GPU read remotely.
3. **Small arrays**: The overhead of issuing a prefetch (driver call, DMA setup) exceeds the benefit for small transfers.
4. **Multiple kernels**: If multiple kernels on different devices access the same array, prefetching to one device may pessimize access from another.

Users who want automatic prefetch can enable it globally via `warp.config.auto_prefetch = True` or per-launch by calling `wp.prefetch()` explicitly before the launch.

### Phase 4: Improve `wp.copy()` for Coherent Systems (Future)

**Goal:** When source and destination arrays are on different devices that can directly access each other's memory, skip the staging buffer logic in `wp.copy()`.

No new attributes or native functions -- this reuses `Device.can_access()` from Phase 1.

The current `wp.copy()` implementation (at `context.py:8702`) has a TODO for this:

```python
# Copying between different devices requires contiguous arrays.  If the arrays
# are not contiguous, we must use temporary staging buffers for the transfer.
# TODO: We can skip the staging if device access is enabled.
```

On systems with `pageable_memory_access`, the GPU can directly read non-contiguous CPU memory, so staging is unnecessary for destination-device copy kernels that read CPU source arrays. The reverse direction is only available when `direct_managed_mem_access_from_host` is true. The fix is straightforward:

```python
if src.device != dest.device:
    # If direct access is available, we can copy non-contiguous arrays
    # without staging, using a kernel on the destination device.
    if dest.device.can_access(src.device):
        # Launch a copy kernel on the destination device that reads
        # directly from the source array's memory.
        pass  # Implementation details TBD
    else:
        # Existing staging buffer logic for non-contiguous arrays...
        ...
```

This is a performance optimization and not required for correctness -- the existing staging approach works correctly on all systems.

### Phase 5: Allocator-Aware Fine-Grained Access Checks (Future)

**Goal:** Track the "accessibility class" of each allocation so that `can_access()` can make fine-grained decisions even on systems without full HMM / host-page-table ATS.

This phase introduces one additional device attribute.

#### New device attribute: `concurrent_managed_access`

**CUDA attribute:** `CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS`

This attribute distinguishes the "limited" unified memory paradigm (limited Tegra, Windows -- `concurrent_managed_access == 0`) from the "full" paradigms (`concurrent_managed_access == 1`). On limited systems, `cudaMallocManaged` allocations bulk-migrate and cannot be concurrently accessed by CPU and GPU. On full systems, managed allocations support page-granularity migration with concurrent access.

The allocator-aware `can_access()` needs this because it must answer: "if this specific array was allocated with `cudaMallocManaged` (a Warp managed allocator), can the GPU access it concurrently with the CPU?" The answer depends on this attribute.

#### Allocator tracking

Currently, Warp has four allocator classes:
- `CpuDefaultAllocator` -- uses `wp_alloc_host` (wraps `malloc`/`calloc`)
- `CpuPinnedAllocator` -- uses `wp_alloc_pinned` (wraps `cudaMallocHost`)
- `CudaDefaultAllocator` -- uses `wp_alloc_device_default` (wraps `cuMemAlloc`)
- `CudaMempoolAllocator` -- uses `wp_alloc_device_async` (wraps `cuMemAllocAsync`)

On a discrete GPU without HMM:
- Pinned CPU allocations (`CpuPinnedAllocator`) ARE GPU-accessible, but the Phase 1 `can_access()` returns `False` for CPU-to-GPU because it cannot distinguish pinned from unpinned.
- Default CPU allocations (`CpuDefaultAllocator`) are NOT GPU-accessible.
- Both CUDA allocators produce GPU-only memory.

If arrays tracked which allocator produced them (they already store `self.allocator` implicitly via `device.get_allocator()`), `can_access()` could accept an optional `allocation_kind` parameter:

```python
def can_access(self, other, allocation_kind=None):
    """..."""
    # ... existing logic ...

    # GPU accessing CPU memory without HMM/host-page-table ATS.
    if self.is_cuda and other.is_cpu:
        if self.can_access_host_memory:
            return True
        # Pinned allocations are always GPU-accessible via UVA.
        if allocation_kind == "pinned" and self.is_uva:
            return True
        return False
```

This would allow `wp.launch()` to accept pinned CPU arrays even on discrete GPUs, which is useful for zero-copy access patterns. However, this requires plumbing the allocation kind through the array type and into `pack_arg()`, which is a larger refactor. It should be considered after the core unified memory support is stable.

## Testing Strategy

### Phase 1 tests

Add a test module `warp/tests/test_unified_memory.py` (registered in `warp/tests/unittest_suites.py`).

**Attribute query tests (run on all hardware):**
- Verify `pageable_memory_access`, `direct_managed_mem_access_from_host`, and `host_native_atomic_supported` are `bool` for CUDA devices and `False` for CPU devices.
- Verify `can_access_host_memory` and `is_host_accessible` return values consistent with the raw attributes.
- If `direct_managed_mem_access_from_host` is `True`, then `pageable_memory_access` should also be `True` (host-visible device memory implies unified host access). This is an invariant check.
- Do not assert that `host_native_atomic_supported` implies `direct_managed_mem_access_from_host`; Jetson Thor reports native host atomics while still rejecting direct CPU access to `cudaMalloc` memory.

**`can_access()` tests (run on all hardware):**
- `device.can_access(device)` is always `True` for every device.
- CPU-to-CPU: always `True`.
- GPU-to-CPU and CPU-to-GPU: assert the result is consistent with the queried attributes:
  - If `pageable_memory_access` is `True`, GPU-to-CPU should be `True`.
  - If `direct_managed_mem_access_from_host` is `True`, CPU-to-GPU should be `True`.
  - If either attribute is `False`, only the corresponding direction should be `False`.
- GPU-to-GPU peer: enable peer access, verify `can_access()` returns `True`. Disable, verify `False`.

**Cross-device launch tests (hardware-dependent, skip on incapable systems):**
- On systems where `cuda_device.can_access_host_memory` is `True`: allocate a CPU array, launch a CUDA kernel that reads and writes it, verify results match expected values.
- On systems where `is_host_accessible` is `True`: allocate a GPU array, verify CPU can access it after launch. This should skip on Jetson Thor.
- Test with output arrays (not just inputs).
- Test with multi-dimensional arrays with non-trivial strides.

**Verification mode tests (run on all hardware):**
- With `warp.config.verify_launch = True` on a discrete GPU without HMM: verify that launching with a CPU array raises `RuntimeError` (not a CUDA fault).
- With `warp.config.verify_launch = True` on an HMM / host-page-table ATS system: verify that CUDA launches with CPU arrays still succeed (no false positive).
- With `warp.config.verify_launch = False` (default): verify that no Python-level device check occurs (cross-device arrays are passed through without error from `pack_arg()`).

### Phase 2 tests (prefetch)

- On HMM / host-page-table ATS systems: prefetch a CPU array to GPU, launch a kernel, verify correctness.
- On systems without HMM / host-page-table ATS: calling `wp.prefetch()` should not raise (no-op or warning).
- Test stream ordering: prefetch then kernel on same stream, verify results.
- Test prefetch back to CPU: prefetch to GPU, then prefetch to CPU, verify CPU access.

### Phase 3 tests (auto-prefetch)

- Enable `warp.config.auto_prefetch`, launch cross-device kernel, verify correctness.
- Verify auto-prefetch is not issued on integrated GPUs (may require mocking or checking driver call counts).

### CI considerations

- The existing CI may not have Grace Hopper / Grace Blackwell / DGX Spark hardware. Tests that require specific paradigms should use `unittest.skipUnless` based on the device attributes queried in Phase 1.
- Tests that only query attributes (Phase 1 attribute and `can_access()` invariant tests) should run on all hardware.
- Consider adding a CI label or tag for "unified memory" tests so they can be selectively run on appropriate hardware.

### Device compatibility matrix for test expectations

| Test scenario | Discrete (no HMM) | Discrete (HMM) | Grace Hopper/Blackwell (bidirectional ATS) | Jetson Orin / limited Tegra | Jetson Thor |
|---|---|---|---|---|---|
| GPU can access CPU arrays | No | Yes | Yes | No | Yes |
| CPU can access GPU arrays | No | No | Yes | No | No |
| Native CPU-GPU atomics on host-visible memory | No | No | Yes | Device-dependent | Yes |
| Cross-device launch GPU->CPU array (default) | CUDA fault | OK | OK | CUDA fault | OK |
| Cross-device launch CPU->GPU array (default) | Segfault | Segfault | OK | Segfault | Segfault |
| Cross-device launch GPU->CPU array (verify mode) | RuntimeError | OK | OK | RuntimeError | OK |
| Cross-device launch CPU->GPU array (verify mode) | RuntimeError | RuntimeError | OK | RuntimeError | RuntimeError |
| `wp.prefetch()` for CPU arrays | No-op / warning | Yes (SW) | Yes (HW) | No-op / warning | Accepted; low expected benefit on integrated DRAM |
