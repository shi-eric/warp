#!/usr/bin/env python3
"""Compare two benchmark JSON files and produce markdown tables.

Usage:
    python _bench_compare.py _benchmark_baseline.json _benchmark_branch.json
    python _bench_compare.py _benchmark_baseline.json _benchmark_branch.json --output comparison.md
"""

import argparse
import json
import math
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def compare_section(baseline, branch, section_key, title):
    """Compare one section (e.g. fem_cpu). Returns markdown string."""
    b_data = baseline.get(section_key, {})
    br_data = branch.get(section_key, {})

    if not b_data and not br_data:
        return ""

    all_keys = list(dict.fromkeys(list(b_data.keys()) + list(br_data.keys())))

    lines = []
    lines.append(f"\n### {title}\n")
    lines.append(f"| Example | Main | Branch | Speedup | Delta | Status |")
    lines.append(f"| --- | ---: | ---: | ---: | ---: | --- |")

    regressions = []

    for key in all_keys:
        b = b_data.get(key, {})
        br_ = br_data.get(key, {})

        b_med = b.get("median", -1)
        br_med = br_.get("median", -1)
        b_status = b.get("status", "missing")
        br_status = br_.get("status", "missing")

        if b_med <= 0 and br_med <= 0:
            lines.append(f"| {key} | N/A | N/A | — | — | both failed |")
            continue
        if b_med <= 0:
            lines.append(f"| {key} | N/A | {br_med:.1f}s | — | — | baseline failed |")
            continue
        if br_med <= 0:
            lines.append(f"| {key} | {b_med:.1f}s | N/A | — | — | branch failed |")
            continue

        speedup = b_med / br_med
        delta_pct = (br_med - b_med) / b_med * 100

        # Statistical significance: flag if delta exceeds combined noise
        b_std = b.get("stdev", 0)
        br_std = br_.get("stdev", 0)
        combined_noise = math.sqrt(b_std**2 + br_std**2) if (b_std > 0 or br_std > 0) else 0
        # Use ~2 sigma as significance threshold
        sig_threshold = 2 * combined_noise / b_med * 100 if b_med > 0 else 10

        if delta_pct > max(5, sig_threshold):
            status = "**REGRESSION**"
            regressions.append((key, b_med, br_med, delta_pct))
        elif delta_pct < -max(5, sig_threshold):
            status = "improved"
        elif abs(delta_pct) < sig_threshold:
            status = "noise"
        else:
            status = "ok"

        lines.append(
            f"| {key} | {b_med:.1f}s | {br_med:.1f}s "
            f"| **{speedup:.2f}x** | {delta_pct:+.1f}% | {status} |"
        )

    return "\n".join(lines), regressions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", help="Baseline JSON file")
    parser.add_argument("branch", help="Branch JSON file")
    parser.add_argument("--output", "-o", default=None, help="Output markdown file")
    args = parser.parse_args()

    baseline = load(args.baseline)
    branch = load(args.branch)

    all_sections = sorted(set(list(baseline.keys()) + list(branch.keys())))

    md_parts = ["# Compile-Time Benchmark Comparison\n"]
    md_parts.append(f"- Baseline: `{args.baseline}`")
    md_parts.append(f"- Branch: `{args.branch}`\n")

    all_regressions = []

    for section in all_sections:
        # Parse section name for a nice title
        parts = section.split("_", 1)
        suite = parts[0].upper()
        device = parts[1] if len(parts) > 1 else "unknown"
        title = f"{suite} — {device}"

        result, regs = compare_section(baseline, branch, section, title)
        if result:
            md_parts.append(result)
            all_regressions.extend(regs)

    # Summary
    md_parts.append("\n## Summary\n")
    if all_regressions:
        md_parts.append(f"**{len(all_regressions)} regression(s) detected:**\n")
        for name, b, br, delta in all_regressions:
            md_parts.append(f"- {name}: {b:.1f}s → {br:.1f}s ({delta:+.1f}%)")
    else:
        md_parts.append("No statistically significant regressions detected.")

    output = "\n".join(md_parts) + "\n"

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Comparison written to {args.output}", flush=True)
    else:
        print(output)


if __name__ == "__main__":
    main()
