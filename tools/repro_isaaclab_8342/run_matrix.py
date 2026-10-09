"""Run each graph-allocation case in its own process with a bounded timeout."""

import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    folder = Path(__file__).parent
    results_dir = folder / "results" / args.label
    results_dir.mkdir(parents=True, exist_ok=True)
    cases = list(
        itertools.product(
            ("thread_local", "relaxed", "global"), ("default", "nonblocking"), (False, True), (False, True)
        )
    )
    results = []
    for trial in range(args.trials):
        for mode, stream, legacy_work, raw in cases:
            name = f"{mode}-{stream}-legacy{int(legacy_work)}-raw{int(raw)}-trial{trial + 1}"
            command = [
                args.python,
                "-u",
                str(folder / "probe_alloc_capture.py"),
                "--mode",
                mode,
                "--stream",
                stream,
                "--rounds",
                str(args.rounds),
            ]
            if legacy_work:
                command.append("--legacy-work")
            if raw:
                command.append("--raw")
            started = time.monotonic()
            with (results_dir / f"{name}.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=folder)
                try:
                    code = process.wait(timeout=args.timeout)
                    outcome = "passed" if code == 0 else "failed"
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    code = None
                    outcome = "timeout"
            row = {
                "case": name,
                "outcome": outcome,
                "exit_code": code,
                "seconds": round(time.monotonic() - started, 3),
                "command": command,
            }
            results.append(row)
            (results_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)
    if any(row["outcome"] != "passed" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
