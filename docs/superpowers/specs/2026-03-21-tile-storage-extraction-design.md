# Extract tile_shared_storage_t to tile_storage.h

**Date:** 2026-03-21
**Status:** Proposed

## Problem

The `WP_NO_TILE` compile guard cannot fully exclude `tile.h` (~5300 lines) because the CUDA and CPU kernel templates unconditionally reference `tile_shared_storage_t`, which is defined in `tile.h`. On the current branch, this is worked around by wrapping the `tile_shared_storage_t` boilerplate in the codegen templates with `#ifndef WP_NO_TILE`. This means kernels that don't use tiles skip the storage setup — a behavioral change from `main`, where every kernel always gets the storage.

The goal is to allow `WP_NO_TILE` to exclude the heavy tile operation code (tile registers, reductions, scans, sorts, matmul, etc.) while keeping `tile_shared_storage_t` always available, restoring the `main` behavior where every kernel unconditionally sets up tile shared storage.

## Design

### 1. New header: `warp/native/tile_storage.h`

Extract from `tile.h`:
- `aligned_alloc()` helper function (~5 lines)
- `shared_tile_storage` pointer/register declaration (~15 lines, under `WP_ENABLE_TILES_IN_STACK_MEMORY`)
- `tile_shared_storage_t` class definition (~90 lines)

Guard with standard `#pragma once`.

### 2. Update `tile.h`

Replace the extracted code with `#include "tile_storage.h"` at the same location. Everything else in `tile.h` stays — tile registers, tile operations, reductions, sorts, etc.

### 3. Update `builtin.h`

Add `#include "tile_storage.h"` **outside** the `#ifndef WP_NO_TILE` block so the type is always available regardless of whether tile operations are included.

### 4. Remove `#ifndef WP_NO_TILE` guards from codegen templates

In `codegen.py`, remove the `#ifndef WP_NO_TILE` / `#endif` wrapping around `tile_shared_storage_t tile_mem` and `tile_shared_storage_t::init()` in all four kernel templates:
- `cuda_kernel_template_forward`
- `cuda_kernel_template_backward`
- `cpu_module_template_forward`
- `cpu_module_template_backward`

This restores the `main` behavior where every kernel unconditionally creates the storage instance and calls `init()`. The type is always defined via `tile_storage.h`, so no `#ifndef` guard is needed.

### 5. No other changes

- No changes to `_COMPILE_GUARD_DEPS`, `VALID_COMPILE_GUARDS`, or the guard collection logic
- No changes to `_SOURCE_GUARD_PATTERNS`
- No changes to any other native headers

## What This Achieves

When `WP_NO_TILE` is defined (kernel doesn't use tile operations):
- **Before this change**: `tile.h` is excluded, but the codegen template skips the storage boilerplate entirely (behavioral change from `main`)
- **After this change**: `tile.h` is excluded (saving ~5200 lines of parsing), but `tile_storage.h` (~110 lines) is always included and the storage boilerplate runs unconditionally (matching `main` behavior)

The `tile_shared_storage_t` constructor/destructor is cheap — on CUDA it just zeroes an offset counter in `__shared__` memory; on CPU it sets up a 256KB stack allocation. This cost is negligible compared to the compile-time savings from excluding the rest of `tile.h`.

## Files Changed

| File | Change |
| --- | --- |
| `warp/native/tile_storage.h` | New file — extracted `tile_shared_storage_t` class |
| `warp/native/tile.h` | Replace extracted code with `#include "tile_storage.h"` |
| `warp/native/builtin.h` | Add `#include "tile_storage.h"` outside `WP_NO_TILE` guard |
| `warp/_src/codegen.py` | Remove `#ifndef WP_NO_TILE` wrapping from 4 kernel templates |

## Testing

- All existing tests pass (the `tile_shared_storage_t` behavior is unchanged)
- Run `bench_compile_time.py` to verify compile-time improvement is preserved
- Run tile tests specifically: `uv run --extra dev -m warp.tests -s autodetect -k TestTile`
