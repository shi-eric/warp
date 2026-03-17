# Pluggable Allocator Interface and RMM Integration

**Status**: Implemented

**Issue**: [GH-781](https://github.com/NVIDIA/warp/issues/781)

## Motivation

Users running mixed GPU workloads (PyTorch + cuML + Warp) benefit from a shared memory
pool across all frameworks. RAPIDS Memory Manager (RMM) is the standard tool for this:
it provides pool, arena, and other allocator strategies that reduce fragmentation and
allocation overhead.

Today, Warp hardcodes four allocator classes (`CpuDefaultAllocator`, `CpuPinnedAllocator`,
`CudaDefaultAllocator`, `CudaMempoolAllocator`) with no extension mechanism. Users cannot
redirect Warp's GPU allocations through an external allocator like RMM.

This feature adds a pluggable allocator interface that enables RMM integration and any
other custom allocator backend.

## Requirements

| ID  | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| R1  | Formal `Allocator` protocol with `allocate`/`deallocate` methods | Must | Formalizes the existing pattern |
| R2  | Per-device custom allocator support for CUDA devices | Must | `set_allocator()` for all, `set_device_allocator()` for one |
| R3  | Built-in RMM adapter (`RmmAllocator`) | Must | Lazy `rmm` import, no hard dependency |
| R4  | `ScopedAllocator` context manager | Should | Follows `ScopedMempool` pattern |
| R5  | `get_device_allocator()` query function | Should | |
| R6  | Custom allocator example (non-RMM) | Should | Teaches the protocol |
| R7  | Documentation in `allocators.rst` | Must | |

**Non-goals:**

- Internal native C++/CUDA allocations (mesh BVH, sparse matrix temporaries, sort scratch
  space). These are small and short-lived. Can be addressed in a follow-up.
- CPU and pinned-memory allocators. RMM manages device memory only.
- Adding RMM as a project dependency. It is Linux-only and CUDA-version-specific.

## Design

### Approach

**Allocator property on Device (Approach 3)** — add `device._custom_allocator` as a
settable attribute on `Device`. When set, `get_allocator()` returns it instead of the
built-in allocator. This is explicit, easy to understand, and keeps pinned memory
unaffected.

### Alternatives Considered

**Protocol + device monkey-patch (Approach 1):** Replace `device.current_allocator`
directly via `set_allocator()`. Works but is informal — the attribute being overwritten
is also used by `set_mempool_enabled()`, creating a confusing interaction where enabling
mempools would silently override the custom allocator.

**Allocator registry (Approach 2):** A central registry mapping `(device, allocator_type)`
to instances. Over-engineered — we only need to replace the CUDA device allocator, not
support named allocator types.

### Key Implementation Details

#### Allocator Protocol

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Allocator(Protocol):
    def allocate(self, size_in_bytes: int) -> int: ...
    def deallocate(self, ptr: int, size_in_bytes: int) -> None: ...
```

The four existing allocator classes are updated to use `allocate`/`deallocate` method names
instead of `alloc`/`free`. The `deleter` attribute is removed; callers use
`allocator.deallocate` directly.

These classes are internal (`warp._src.context`), not public API. No backward-compatible
aliases are provided. Any internal code still referencing `alloc`/`free`/`deleter` is
updated in the same change.

The protocol intentionally omits a `stream` parameter. Warp's existing allocators do not
expose stream selection at the Python level — stream-ordered allocation is handled
internally by `CudaMempoolAllocator` using the device's current stream. Custom allocators
that need stream awareness (e.g., RMM with `CudaAsyncMemoryResource`) should obtain the
stream from the current CUDA context at call time. A `stream` parameter may be added in
a future revision if needed.

#### Validation

A shared `_validate_allocator()` helper in `context.py` checks that an allocator conforms
to the `Allocator` protocol via `isinstance()` (enabled by `@runtime_checkable`). This
helper is called from `set_allocator()`, `set_device_allocator()`, and
`ScopedAllocator.__init__()`, avoiding duplication of the validation logic and error
message.

#### Device Integration

```python
# In Device.__init__, CUDA branch, after existing allocator setup:
self._custom_allocator = None

# Modified get_allocator():
def get_allocator(self, pinned: bool = False):
    if self.is_cuda:
        if self._custom_allocator is not None:
            return self._custom_allocator
        return self.current_allocator
    else:
        return self.pinned_allocator if pinned else self.default_allocator
```

The custom allocator takes priority over the built-in mempool/default allocator.
Setting `_custom_allocator` to `None` restores the built-in allocator.

**Interaction with `set_mempool_enabled()`:** When a custom allocator is active,
`set_mempool_enabled()` still updates `device.current_allocator` (the built-in allocator
selection) but has no visible effect because `get_allocator()` returns the custom allocator
first. When the custom allocator is later removed, the built-in allocator reflects whatever
mempool setting was last configured. This is the expected behavior — the two mechanisms are
independent layers. No warning is needed.

**Thread safety:** Setting `device._custom_allocator` is a single Python attribute
assignment (atomic under the GIL). This matches the thread-safety model of the existing
`device.current_allocator`. Warp's allocator infrastructure is not designed for concurrent
modification from multiple threads; users should set allocators before starting concurrent
work.

#### Public API

```python
def set_allocator(allocator: Allocator | None) -> None:
    """Set the memory allocator for all CUDA devices."""
    _validate_allocator(allocator)
    for device in get_cuda_devices():
        device._custom_allocator = allocator

def set_device_allocator(device: DeviceLike, allocator: Allocator | None) -> None:
    """Set the memory allocator for a specific CUDA device."""
    device = get_device(device)
    if not device.is_cuda:
        raise RuntimeError("Custom allocators are only supported on CUDA devices")
    _validate_allocator(allocator)
    device._custom_allocator = allocator

def get_device_allocator(device: DeviceLike) -> Allocator:
    """Get the current effective memory allocator for a device."""
    device = get_device(device)
    return device.get_allocator()
```

`ScopedAllocator` in `warp/_src/utils.py` follows the `ScopedMempool` pattern — saves
`device._custom_allocator` on enter, restores on exit. Its `__enter__` returns `self`
to support `with ... as ctx:` usage.

#### Array Lifecycle

In `array._init_new`:

```python
allocator = device.get_allocator(pinned=pinned)
if capacity > 0:
    with device.context_guard:
        ptr = allocator.allocate(capacity)
else:
    ptr = None
# ...
self.deleter = allocator.deallocate
self._allocator = allocator
```

**CUDA context correctness:** The existing built-in allocators pass the device context
explicitly to native C functions (`wp_alloc_device_default(self.device.context, ...)`), so
they work regardless of the current CUDA context. Custom allocators (including RMM) call
Python-level APIs that operate on the current CUDA context. Wrapping the `allocate()` call
in `device.context_guard` ensures the correct CUDA device is active. This guard is also
used in `fabricarray.__init__` for bucket allocation and in `array.__del__` for
deallocation.

`array.__del__` calls `self.deleter(self.ptr, self.capacity)` within `device.context_guard`.
A guard at the top of `__del__` skips deallocation for partially-initialized arrays (where
`self.device` may not exist) and zero-size arrays (where `self.ptr` is `None`).
`TypeError`/`AttributeError` exceptions are suppressed during interpreter shutdown when
callables become `None`. Other exceptions from custom allocator `deallocate` calls are
caught and reported via `warn()` to avoid noisy `__del__` tracebacks. The array holds a
reference to `self._allocator`, keeping the allocator (and its state, e.g.,
`RmmAllocator._buffers`) alive until all arrays allocated through it are garbage-collected.

#### Built-in RMM Adapter

New module `warp/_src/rmm_allocator.py`:

```python
class RmmAllocator:
    """Allocator that routes Warp device memory through RMM."""

    def __init__(self, stream=None):
        try:
            import rmm
        except ImportError as e:
            raise ImportError(
                "Failed to import 'rmm'. Ensure it is installed and compatible with your CUDA version. "
                "See https://docs.rapids.ai/install/ for installation instructions."
            ) from e
        self._buffers: dict[int, object] = {}
        self._stream = stream

    def allocate(self, size_in_bytes: int) -> int:
        if size_in_bytes == 0:
            return 0
        import rmm
        buf = (
            rmm.DeviceBuffer(size=size_in_bytes)
            if self._stream is None
            else rmm.DeviceBuffer(size=size_in_bytes, stream=self._stream)
        )
        ptr = buf.ptr
        self._buffers[ptr] = buf
        return ptr

    def deallocate(self, ptr: int, size_in_bytes: int) -> None:
        if ptr == 0:
            return  # Zero-size allocation; nothing was allocated.
        try:
            del self._buffers[ptr]
        except KeyError:
            raise RuntimeError(
                f"RmmAllocator.deallocate called with unrecognized pointer {ptr:#x} ..."
            ) from None
```

Key design decisions:

- **GC prevention** uses a pointer-keyed dict (same pattern as Numba's RMM integration).
  Each `DeviceBuffer` is stored by its pointer address; deleting the entry releases the
  buffer back to RMM.
- **Zero-size handling:** `allocate(0)` returns `0` without touching RMM. `deallocate`
  with `ptr == 0` returns immediately. In practice, `array.__del__` guards against
  `ptr is None` (set for zero-capacity arrays in `_init_new`) before calling `deallocate`,
  so the `ptr == 0` check is a defensive guard for direct callers of `deallocate`.
- **Double-free detection:** `deallocate` raises `RuntimeError` if the pointer is not in
  `_buffers`, catching double-free bugs and mismatched allocator usage.
- **Resource resolution timing:** Each allocation delegates to `rmm.DeviceBuffer`, which
  uses whichever `DeviceMemoryResource` is active at the time of allocation. Changing the
  RMM resource between allocations affects subsequent allocations.

`RmmAllocator` is exported from `warp/__init__.py` via lazy import.

A single `RmmAllocator` instance can safely be shared across multiple CUDA devices via
`set_allocator()` because `allocate()` is always called within `device.context_guard`
(see Array Lifecycle above), ensuring `rmm.DeviceBuffer` allocates on the correct device.
The `_buffers` dict may contain pointers from different devices; this is safe because
CUDA device pointers are unique across devices and deallocation also happens under the
correct context guard.

#### Graph Capture Caveat

Custom allocators that do not use stream-ordered allocation may produce silently corrupted
graphs during CUDA graph capture — the capture succeeds but replay uses stale pointers.
Warp does not emit a warning because it cannot reliably determine whether a custom allocator
supports graph capture. Users are responsible for ensuring their allocator is compatible
(e.g., using `CudaAsyncMemoryResource` with RMM). The `allocators.rst` documentation
includes a warning about this limitation.

#### FEM and Fabric Integration

`warp/_src/fem/cache.py` (`TemporaryStore.Pool`): The pool captures the allocator's
`deallocate` callable per-allocation in its `_allocs` dict, stored as
`(capacity, deallocate)` tuples. This ensures each buffer is freed using the allocator
that created it, even if the device's allocator changes between allocations.

`warp/_src/fabric.py` (`fabricarray`): Bucket allocation uses `device.get_allocator()`
and wraps the `allocate()` call in `device.context_guard`, consistent with `array.__init__`.

## Testing Strategy

New test module `warp/tests/test_allocator.py`, added to `default_suite` in
`warp/tests/unittest_suites.py`:

| Test | Description | Device |
| --- | --- | --- |
| Protocol conformance | Built-in allocators satisfy `Allocator` protocol | CPU + CUDA |
| Custom allocator | Counting allocator tracks allocations via `set_allocator()` (tests `wp.zeros`, `wp.empty`, `wp.full`) | CUDA |
| Per-device allocator | Different allocators on different CUDA devices | Multi-GPU |
| ScopedAllocator | Restore-on-exit, including on exception | CUDA |
| Reset to default | `set_allocator(None)` restores built-in behavior | CUDA |
| RMM allocator | Create arrays, round-trip, verify `_buffers` cleanup | CUDA (Linux) |
| Interop | RMM + Warp array exported to PyTorch | CUDA (Linux) |
| Allocator swap with live arrays | Set custom allocator, create arrays, reset to default, verify arrays deallocate correctly via old allocator | CUDA |
| Allocate failure | Custom allocator that raises on allocate; verify Warp propagates the error cleanly without leaking state | CUDA |
| Double-free detection | Verify `RmmAllocator` raises on duplicate deallocation | CUDA (Linux) |
| Zero-size allocation | Verify zero-size arrays skip allocation/deallocation | CUDA |
| Multi-device broadcast | `set_allocator()` applies to all CUDA devices | Multi-GPU |
| Per-device isolation | `set_device_allocator()` affects only the target device | Multi-GPU |

The example (`warp/examples/core/example_custom_allocator.py`) is registered in
`warp/tests/test_examples.py` with `cuda_test_devices`.

RMM tests are guarded with `@unittest.skipUnless(rmm_available, "rmm not installed")`.
CI runs them on Linux + CUDA 12 with:

```bash
uv run --with rmm-cu12 --extra dev -m warp.tests -s autodetect -k TestAllocator
```

## Files Changed

| File | Change |
| --- | --- |
| `warp/_src/context.py` | `Allocator` protocol, `_validate_allocator()` helper, rename `alloc`/`free` to `allocate`/`deallocate`, `Device._custom_allocator`, `set_allocator()`, `set_device_allocator()`, `get_device_allocator()` |
| `warp/_src/types.py` | `array._init_new` uses `allocate`/`deallocate`, `__del__` guard for partial init |
| `warp/_src/utils.py` | `ScopedAllocator` context manager |
| `warp/_src/rmm_allocator.py` | New: `RmmAllocator` class |
| `warp/_src/fabric.py` | `fabricarray` uses `allocate`/`deallocate`, `context_guard` around allocation |
| `warp/_src/fem/cache.py` | `TemporaryStore.Pool` captures deallocator per-allocation |
| `warp/__init__.py` | Export new public API |
| `warp/tests/test_allocator.py` | New: allocator tests |
| `warp/tests/test_examples.py` | Register `example_custom_allocator` |
| `warp/tests/unittest_suites.py` | Add `test_allocator` to `default_suite` |
| `docs/deep_dive/allocators.rst` | Custom allocators and RMM documentation |
| `warp/examples/core/example_custom_allocator.py` | New: custom allocator example |
| `CHANGELOG.md` | Unreleased entry |
