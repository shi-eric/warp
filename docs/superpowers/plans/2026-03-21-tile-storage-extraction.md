# tile_storage.h Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `tile_shared_storage_t` from `tile.h` into a standalone `tile_storage.h` so that `WP_NO_TILE` can fully exclude the heavy tile operations (~5200 lines) while keeping tile shared storage always available.

**Architecture:** Move `tile_shared_storage_t`, its helper function `tile_align()`, and the `WP_TILE_THREAD_IDX` macro into a new `tile_storage.h`. Include it unconditionally from `builtin.h` (outside the `WP_NO_TILE` guard). Remove the `#ifndef WP_NO_TILE` wrapping from codegen kernel templates, restoring the `main` behavior where every kernel sets up tile storage unconditionally.

**Tech Stack:** C++ (native headers), Python (codegen.py)

**Spec:** `docs/superpowers/specs/2026-03-21-tile-storage-extraction-design.md`

---

### Task 1: Create `warp/native/tile_storage.h`

**Files:**
- Create: `warp/native/tile_storage.h`
- Modify: `warp/native/tile.h:68-999`

- [ ] **Step 1: Create `tile_storage.h`**

Create `warp/native/tile_storage.h` with the following content extracted from `tile.h`:

1. License header (copy from `tile.h` lines 1-2)
2. `#pragma once`
3. `WP_TILE_THREAD_IDX` macro (tile.h lines 68-72)
4. `tile_align()` function (tile.h lines 872-881)
5. `shared_tile_storage` pointer/register declaration (tile.h lines 883-900)
6. `tile_shared_storage_t` class definition (tile.h lines 902-999)

The file must NOT include `builtin.h` or `tile.h` — it will be included FROM `builtin.h`, so `CUDA_CALLABLE` and other macros are already defined by that point. It also does NOT need `rand.h`.

```cpp
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Thread index for tile shared memory — maps to threadIdx.x on CUDA, 0 on CPU.
#if defined(__CUDACC_RTC__)
#define WP_TILE_THREAD_IDX threadIdx.x
#else
#define WP_TILE_THREAD_IDX 0
#endif

// [paste tile_align() from tile.h lines 872-881]
// [paste shared_tile_storage pointer from tile.h lines 883-900]
// [paste tile_shared_storage_t class from tile.h lines 902-999]
```

- [ ] **Step 2: Replace extracted code in `tile.h` with `#include "tile_storage.h"`**

In `tile.h`, remove lines 68-72 (`WP_TILE_THREAD_IDX`), lines 872-999 (`tile_align` through end of `tile_shared_storage_t` class), and add `#include "tile_storage.h"` in their place. The include should go after the `rand.h` include and clang diagnostic pragmas but before any code that uses `tile_shared_storage_t`.

Specifically, replace tile.h lines 68-72 with a comment:
```cpp
// WP_TILE_THREAD_IDX is defined in tile_storage.h
```

And replace tile.h lines 872-999 with:
```cpp
#include "tile_storage.h"
```

- [ ] **Step 3: Verify `tile.h` still compiles via a trivial import test**

```bash
uv run python -c "import warp as wp; wp.init(); print('OK')"
```

Expected: `OK` (no compilation errors from header restructuring).

- [ ] **Step 4: Commit**

```bash
git add warp/native/tile_storage.h warp/native/tile.h
git commit -s -m "Extract tile_shared_storage_t to tile_storage.h"
```

---

### Task 2: Include `tile_storage.h` unconditionally from `builtin.h`

**Files:**
- Modify: `warp/native/builtin.h:2205-2210`

- [ ] **Step 1: Add `#include "tile_storage.h"` before the `WP_NO_TILE` guard in `builtin.h`**

In `builtin.h`, the current tile guard block (around line 2205) looks like:

```cpp
#ifndef WP_NO_TILE
#include "tile.h"
#include "tile_reduce.h"
#include "tile_scan.h"
#include "tile_radix_sort.h"
#endif
```

Add `#include "tile_storage.h"` BEFORE the `#ifndef WP_NO_TILE` line:

```cpp
#include "tile_storage.h"
#ifndef WP_NO_TILE
#include "tile.h"
#include "tile_reduce.h"
#include "tile_scan.h"
#include "tile_radix_sort.h"
#endif
```

This ensures `tile_shared_storage_t` is always available regardless of whether `WP_NO_TILE` is defined.

- [ ] **Step 2: Verify Warp still imports cleanly**

```bash
uv run python -c "import warp as wp; wp.init(); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add warp/native/builtin.h
git commit -s -m "Include tile_storage.h unconditionally from builtin.h"
```

---

### Task 3: Remove `#ifndef WP_NO_TILE` guards from codegen kernel templates

**Files:**
- Modify: `warp/_src/codegen.py:4338-4382` (CUDA templates)
- Modify: `warp/_src/codegen.py:4405-4446` (CPU templates)

- [ ] **Step 1: Remove guards from `cuda_kernel_template_forward` (line ~4338)**

Change from:
```
{{
#ifndef WP_NO_TILE
{line_directive}    wp::tile_shared_storage_t tile_mem;
#endif
...
    {{
#ifndef WP_NO_TILE
            // reset shared memory allocator
{line_directive}        wp::tile_shared_storage_t::init();
#endif
```

To (restoring `main` behavior):
```
{{
{line_directive}    wp::tile_shared_storage_t tile_mem;
...
    {{
{line_directive}        wp::tile_shared_storage_t::init();
```

- [ ] **Step 2: Remove guards from `cuda_kernel_template_backward` (line ~4361)**

Same pattern as Step 1.

- [ ] **Step 3: Remove guards from `cpu_module_template_forward` (line ~4405)**

Change from:
```
{{
#ifndef WP_NO_TILE
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif
#endif
```

To (restoring `main` behavior):
```
{{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif
```

- [ ] **Step 4: Remove guards from `cpu_module_template_backward` (line ~4431)**

Same pattern as Step 3.

- [ ] **Step 5: Commit**

```bash
git add warp/_src/codegen.py
git commit -s -m "Remove WP_NO_TILE guards from kernel templates

tile_shared_storage_t is now always available via tile_storage.h,
so the #ifndef WP_NO_TILE wrapping is no longer needed. This
restores the main behavior where every kernel unconditionally
sets up tile shared storage."
```

---

### Task 4: Run tests and benchmark

**Files:**
- No code changes — verification only

- [ ] **Step 1: Run the compile guard unit tests**

```bash
uv run python warp/tests/test_compile_guards.py -v
```

Expected: All 26 tests pass.

- [ ] **Step 2: Run tile-specific tests**

```bash
uv run --extra dev -m warp.tests -s autodetect --failfast -k TestTile -k TestTileLoad -k TestTileMatmul -k TestTileCholesky -k TestTileAtomicBitwise
```

Expected: All tile tests pass on both CPU and CUDA.

- [ ] **Step 3: Run the compile-time benchmark**

```bash
uv run bench_compile_time.py --device cpu --samples 5
uv run bench_compile_time.py --device cuda:0 --samples 5
```

Expected: CPU ~0.33s, CUDA ~0.15s (same as before — the optimization is preserved and now tile.h is fully excluded).

- [ ] **Step 4: Run a broader test set to verify no regressions**

```bash
uv run --extra dev -m warp.tests -s autodetect --failfast -k TestCodeGen -k TestFunc -k TestModuleLite -k TestMatLite -k TestFem
```

Expected: All tests pass.

- [ ] **Step 5: Run the full test suite**

```bash
uv run --extra dev -m warp.tests -s autodetect 2>&1 | tee /tmp/tile_storage_tests.log | tail -5
```

Expected: 6626 tests, 0 errors, 16 skipped.
