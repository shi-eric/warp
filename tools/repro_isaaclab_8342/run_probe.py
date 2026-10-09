"""Save one diagnostic's output and terminate that process on timeout."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--python", required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--timeout", type=int, default=180)
parser.add_argument("--trials", type=int, default=1)
parser.add_argument("script")
parser.add_argument("arguments", nargs=argparse.REMAINDER)
args = parser.parse_args()
folder = Path(__file__).parent
results_dir = folder / "results" / args.label
results_dir.mkdir(parents=True, exist_ok=True)
results = []
for trial in range(args.trials):
    command = [args.python, "-u", str(folder / args.script), *args.arguments]
    started = time.monotonic()
    with (results_dir / f"trial{trial + 1}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=args.timeout)
            outcome = "passed" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            if sys.platform == "win32":
                # A virtual-environment launcher can have a separate Python child.
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    check=True,
                    timeout=10,
                )
            else:
                process.kill()
            process.wait()
            code = None
            outcome = "timeout"
    row = {"outcome": outcome, "exit_code": code, "seconds": round(time.monotonic() - started, 3), "command": command}
    print(json.dumps(row), flush=True)
    results.append(row)
    (results_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
if any(row["outcome"] != "passed" for row in results):
    raise SystemExit(1)
