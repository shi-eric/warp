"""Run a diagnostic under a native CUPTI parameter logger on Windows."""

import argparse
import ctypes
import os
import runpy
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dll", required=True)
parser.add_argument("--cupti", required=True)
parser.add_argument("--log", required=True)
parser.add_argument("script")
parser.add_argument("arguments", nargs=argparse.REMAINDER)
args = parser.parse_args()
search_path = os.add_dll_directory(str(Path(args.cupti).resolve()))
native = ctypes.CDLL(str(Path(args.dll).resolve()))
native.start_trace.argtypes = [ctypes.c_char_p]
native.start_trace.restype = ctypes.c_int
native.stop_trace.argtypes = []
native.stop_trace.restype = ctypes.c_int
status = native.start_trace(os.fsencode(str(Path(args.log).resolve())))
if status:
    raise RuntimeError(f"CUPTI logger initialization failed: {status}")
print("CUPTI parameter log:", args.log, flush=True)
script = Path(args.script).resolve()
sys.path.insert(0, str(script.parent))
sys.argv = [str(script), *args.arguments]
try:
    runpy.run_path(str(script), run_name="__main__")
finally:
    status = native.stop_trace()
    if status:
        raise RuntimeError(f"CUPTI logger finalization failed: {status}")
