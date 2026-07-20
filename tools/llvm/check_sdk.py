#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sanity-check a deployed LLVM SDK tree (tree shape, machine type, ABI markers).

Platform binary tools used: readelf + nm (Linux), lipo + otool (macOS),
dumpbin (Windows; requires a VS developer environment).
"""

import argparse
import os
import subprocess
import sys

PLATFORMS = ("linux-x86_64", "linux-aarch64", "macos-arm64", "windows-x86_64", "windows-arm64")

_ELF_MACHINES = {"linux-x86_64": "X86-64", "linux-aarch64": "AArch64"}
_PE_MACHINES = {"windows-x86_64": "8664", "windows-arm64": "AA64"}
_MIN_ARCHIVES = 50
_SPOT_CHECK_LIB = {"linux": "libLLVMSupport.a", "macos": "libLLVMSupport.a", "windows": "LLVMSupport.lib"}


class ToolError(Exception):
    """A required platform binary tool is missing or its invocation failed."""


def run(cmd):
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True
        ).stdout
    except FileNotFoundError as exc:
        raise ToolError(f"required tool not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise ToolError(f"{' '.join(cmd)} failed with exit {exc.returncode}: {stderr[:200]}") from exc


def check_tree(sdk, platform, errors):
    suffix = ".lib" if platform.startswith("windows") else ".a"
    for required in ("include/llvm/IR", "include/clang", "lib", "licenses/llvm", "licenses/clang"):
        if not os.path.isdir(os.path.join(sdk, required)):
            errors.append(f"missing directory: {required}")
    for lic in ("licenses/llvm/LICENSE.TXT", "licenses/clang/LICENSE.TXT"):
        if not os.path.isfile(os.path.join(sdk, lic)):
            errors.append(f"missing license: {lic}")
    lib_dir = os.path.join(sdk, "lib")
    if not os.path.isdir(lib_dir):
        return []
    libs = sorted(os.listdir(lib_dir))
    non_static = [name for name in libs if not name.endswith(suffix)]
    if non_static:
        errors.append(f"non-static-library files in lib/: {non_static}")
    if len(libs) < _MIN_ARCHIVES:
        errors.append(f"only {len(libs)} archives in lib/ (expected >= {_MIN_ARCHIVES})")
    pruned = [name for name in libs if "clangStaticAnalyzer" in name]
    if pruned:
        errors.append(f"static analyzer archives should have been pruned: {pruned}")
    return libs


def check_linux(sdk, platform, errors):
    lib = os.path.join(sdk, "lib", _SPOT_CHECK_LIB["linux"])
    header = run(["readelf", "-h", lib])
    if _ELF_MACHINES[platform] not in header:
        errors.append(f"{lib}: expected ELF machine {_ELF_MACHINES[platform]}")
    # The pre-C++11 ABI leaves no std::__cxx11 mangled names anywhere.
    symbols = run(["nm", "--format=posix", lib])
    if "_ZNSt7__cxx11" in symbols:
        errors.append(f"{lib}: found std::__cxx11 symbols; _GLIBCXX_USE_CXX11_ABI=0 was not applied")


def check_macos(sdk, errors):
    lib = os.path.join(sdk, "lib", _SPOT_CHECK_LIB["macos"])
    archs = run(["lipo", "-archs", lib]).strip()
    if archs != "arm64":
        errors.append(f"{lib}: lipo -archs reported {archs!r}, expected arm64")
    load_cmds = run(["otool", "-l", lib])
    if "minos 11.0" not in load_cmds:
        errors.append(f"{lib}: LC_BUILD_VERSION minos 11.0 not found (deployment target drifted)")


def check_windows(sdk, platform, errors):
    lib = os.path.join(sdk, "lib", _SPOT_CHECK_LIB["windows"])
    headers = run(["dumpbin", "/headers", lib])
    if _PE_MACHINES[platform] not in headers:
        errors.append(f"{lib}: expected PE machine {_PE_MACHINES[platform]}")
    directives = run(["dumpbin", "/directives", lib]).lower()
    if "libcmt" not in directives:
        errors.append(f"{lib}: LIBCMT directive missing (static CRT not used)")
    if "msvcrt" in directives:
        errors.append(f"{lib}: MSVCRT directive present (dynamic CRT leaked in)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-dir", required=True)
    parser.add_argument("--platform", required=True, choices=PLATFORMS)
    args = parser.parse_args(argv)

    errors = []
    libs = check_tree(args.sdk_dir, args.platform, errors)
    if libs:
        try:
            if args.platform.startswith("linux"):
                check_linux(args.sdk_dir, args.platform, errors)
            elif args.platform == "macos-arm64":
                check_macos(args.sdk_dir, errors)
            else:
                check_windows(args.sdk_dir, args.platform, errors)
        except ToolError as exc:
            errors.append(str(exc))

    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
