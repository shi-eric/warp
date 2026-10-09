"""Run a script while logging the CUDA driver's UVM-Lite VA reservation IOCTLs on Windows.

The CUDA driver (``nvcuda64.dll``) sends these to ``\\Device\\UVMLiteController*`` once
managed memory has initialized UVM-Lite. Their 24-byte parameters are
``{base, length, status}``. Each failed window placement appears as a reserve
followed by a slow release, then a reserve 2 MiB higher. This patches
``nvcuda64.dll``'s import of ``DeviceIoControl`` and needs only ctypes, so it works in
the OVRTX environment too::

    python trace_uvm_ioctls.py --log uvm.txt minimal_uvm_va_capture.py
"""

import argparse
import ctypes
import runpy
import struct
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

NAMES = {0x0022E004: "reserve", 0x0022E008: "release", 0x0022E00C: "commit"}
PROTOTYPE = ctypes.WINFUNCTYPE(
    wintypes.BOOL,
    wintypes.HANDLE,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
    ctypes.c_void_p,
)

parser = argparse.ArgumentParser()
parser.add_argument("--log", type=Path, required=True)
parser.add_argument("script", type=Path)
parser.add_argument("arguments", nargs=argparse.REMAINDER)
args = parser.parse_args()

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
kernel32.GetModuleHandleW.restype = ctypes.c_void_p
kernel32.VirtualProtect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
if ctypes.WinDLL("nvcuda.dll").cuInit(0):
    raise RuntimeError("cuInit failed")


def import_slot(module, library, function):
    """Return the address of ``module``'s import address table entry for ``library!function``."""
    base = kernel32.GetModuleHandleW(module)
    u32 = lambda address: ctypes.c_uint32.from_address(address).value  # noqa: E731
    descriptor = base + u32(base + u32(base + 0x3C) + 24 + 112 + 8)
    while u32(descriptor + 12):
        if ctypes.string_at(base + u32(descriptor + 12)).decode().lower() == library:
            names, slots = base + (u32(descriptor) or u32(descriptor + 16)), base + u32(descriptor + 16)
            for index in range(1 << 16):
                thunk = ctypes.c_uint64.from_address(names + 8 * index).value
                if not thunk:
                    break
                if not thunk >> 63 and ctypes.string_at(base + (thunk & 0x7FFFFFFF) + 2).decode() == function:
                    return slots + 8 * index
        descriptor += 20
    raise RuntimeError(f"{module} does not import {library}!{function}")


log = args.log.open("w", encoding="utf-8", buffering=1)
lock = threading.Lock()
started = time.perf_counter()
log.write(f"Timestamps are seconds after perf_counter={started:.3f}\n")
slot = import_slot("nvcuda64.dll", "kernel32.dll", "DeviceIoControl")
original = PROTOTYPE(ctypes.c_void_p.from_address(slot).value)


@PROTOTYPE
def device_io_control(handle, code, in_buffer, in_size, out_buffer, out_size, returned, overlapped):
    if code not in NAMES or in_size < 16:
        return original(handle, code, in_buffer, in_size, out_buffer, out_size, returned, overlapped)
    base, length = struct.unpack("<QQ", ctypes.string_at(in_buffer, 16))
    start = time.perf_counter()
    result = original(handle, code, in_buffer, in_size, out_buffer, out_size, returned, overlapped)
    elapsed = time.perf_counter() - start
    with lock:
        log.write(
            f"[{start - started:9.3f}s] tid={threading.get_native_id()} {NAMES[code]:7} base={base:#x} "
            f"length={length:#x} ok={result} {elapsed * 1000:9.3f} ms\n"
        )
    return result


old = wintypes.DWORD()
kernel32.VirtualProtect(slot, 8, 0x04, ctypes.byref(old))
ctypes.c_void_p.from_address(slot).value = ctypes.cast(device_io_control, ctypes.c_void_p).value
kernel32.VirtualProtect(slot, 8, old.value, ctypes.byref(old))
script = args.script.resolve()
sys.argv = [str(script), *args.arguments]
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name="__main__")
