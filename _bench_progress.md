---
name: benchmark_progress
description: Tracking progress of compile-time benchmark recollection (2026-03-25)
---

# Benchmark Recollection Progress — 2026-03-25

## Context
- Recollecting CPU + CUDA data for FEM and Newton examples
- Main branch now includes `ershi/fix-cu12-pch` (commit 66766b51) which significantly reduces CUDA baseline times
- PCH fix impact confirmed: e.g., fem.navier_stokes CUDA baseline dropped from 23.6s → 13.6s

## Runs Completed

### 1. Baseline FEM (DONE)
- Command: `python3 _bench_comprehensive.py --label baseline --suite fem --samples 3`
- Output: `_benchmark_baseline.json`
- Status: Complete, all examples succeeded
- Key observation: CUDA times ~40-50% lower than previous historical data due to PCH fix on main

### 2. Baseline Newton (IN PROGRESS)
- Command: `python3 _bench_comprehensive.py --label baseline --suite newton --samples 3 --output _benchmark_baseline_newton_new.json`
- Output: `_benchmark_baseline_newton_new.json`
- Status: Running in background (task ID: b18ebzqr4)
- Output file: /tmp/claude-1000/-home-horde-code-projects-warp-worktree-3/7853b631-921d-4c2c-b54a-9d743eefd9bd/tasks/b18ebzqr4.output

## Runs Remaining

### 3. Branch FEM
- Command: `python3 _bench_comprehensive.py --label branch --suite fem --samples 3`
- Output: `_benchmark_branch.json` (default) or specify `--output`

### 4. Branch Newton
- Command: `python3 _bench_comprehensive.py --label branch --suite newton --samples 3 --output _benchmark_branch_newton_new.json`

## After All Runs

### 5. Merge JSONs
Need to merge baseline FEM + Newton into `_benchmark_baseline_all.json` and branch FEM + Newton into `_benchmark_branch_all.json` (or use the individual files directly).

### 6. Compare
```bash
python3 _bench_compare.py _benchmark_baseline_all.json _benchmark_branch_all.json -o _benchmark_comparison.md
```

### 7. Update _mr_benchmark_data.md
Add new "Comparison with previous measurements" section noting the PCH fix impact on CUDA baselines.

### 8. Version control
Commit updated JSON + markdown in ../warp-worktree-2.

## Historical Reference
- Previous baseline main commit: 737e4e58 (2026-03-22)
- Current baseline main commit: c687bc58 (includes PCH fix)
- Branch: ershi/robust-compile-guards (warp-worktree-3)
