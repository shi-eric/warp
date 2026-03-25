#!/bin/bash
# Runs all benchmark phases sequentially, survives session drops.
# Progress is logged to _bench_full_run.log

set -e
cd /home/horde/code-projects/warp-worktree-2

LOG=_bench_full_run.log
echo "=== Benchmark run started $(date) ===" >> "$LOG"

# Phase 1: Baseline Newton (always run — previous in-session run may have been killed)
echo "=== Phase 1: Baseline Newton — $(date) ===" >> "$LOG"
python3 _bench_comprehensive.py --label baseline --suite newton --samples 3 --output _benchmark_baseline_newton_new.json >> "$LOG" 2>&1

# Phase 2: Branch FEM
echo "=== Phase 2: Branch FEM — $(date) ===" >> "$LOG"
python3 _bench_comprehensive.py --label branch --suite fem --samples 3 --output _benchmark_branch_fem_new.json >> "$LOG" 2>&1

# Phase 3: Branch Newton
echo "=== Phase 3: Branch Newton — $(date) ===" >> "$LOG"
python3 _bench_comprehensive.py --label branch --suite newton --samples 3 --output _benchmark_branch_newton_new.json >> "$LOG" 2>&1

# Phase 4: Merge JSONs into _all files for comparison
echo "=== Phase 4: Merging JSONs — $(date) ===" >> "$LOG"
python3 -c "
import json

# Merge baseline
baseline = {}
for f in ['_benchmark_baseline.json', '_benchmark_baseline_newton_new.json']:
    with open(f) as fh:
        baseline.update(json.load(fh))
with open('_benchmark_baseline_all.json', 'w') as fh:
    json.dump(baseline, fh, indent=2)

# Merge branch
branch = {}
for f in ['_benchmark_branch_fem_new.json', '_benchmark_branch_newton_new.json']:
    with open(f) as fh:
        branch.update(json.load(fh))
with open('_benchmark_branch_all.json', 'w') as fh:
    json.dump(branch, fh, indent=2)

print('Merged into _benchmark_baseline_all.json and _benchmark_branch_all.json')
" >> "$LOG" 2>&1

# Phase 5: Generate comparison
echo "=== Phase 5: Generating comparison — $(date) ===" >> "$LOG"
python3 _bench_compare.py _benchmark_baseline_all.json _benchmark_branch_all.json --output _benchmark_comparison.md >> "$LOG" 2>&1

echo "=== All benchmarks complete — $(date) ===" >> "$LOG"
echo "DONE" > _bench_run_status.txt
