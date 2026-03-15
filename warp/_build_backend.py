# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PEP 517 build backend that auto-builds native libraries when needed.

This module wraps ``setuptools.build_meta`` and ensures Warp's native
libraries exist before wheel or editable-wheel creation.  When
``warp/bin/`` already contains the runtime library (e.g. from a CI
pre-build step), the build is skipped.

Usage in ``pyproject.toml``::

    [build-system]
    requires = ["setuptools>=75.3.2", "build", "wheel", "numpy"]
    build-backend = "warp._build_backend"
    backend-path = ["."]

Config settings (passed via ``pip install -C key=value``)::

    warp-no-cuda=true        -> --no-cuda
    warp-quick=true          -> --quick
    warp-jobs=N              -> -j N
    warp-no-standalone=true  -> --no-standalone
    warp-mode=debug          -> --mode debug
    warp-verbose=false       -> --no-verbose
    warp-verify-fp=true      -> --verify-fp
    warp-fast-math=true      -> --fast-math
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys

from setuptools.build_meta import *  # noqa: F401, F403
from setuptools.build_meta import build_editable as _build_editable
from setuptools.build_meta import build_wheel as _build_wheel


def _runtime_lib_name() -> str:
    """Return the platform-specific name of the core Warp runtime library."""
    if platform.system() == "Windows":
        return "warp.dll"
    elif platform.system() == "Darwin":
        return "libwarp.dylib"
    return "warp.so"


def _has_runtime_library() -> bool:
    """Check whether warp/bin/ already contains the runtime library."""
    warp_bin = os.path.join(os.path.dirname(__file__), "bin")
    return os.path.isfile(os.path.join(warp_bin, _runtime_lib_name()))


def _build_lib_args_from_config(config_settings: dict | None) -> list[str]:
    """Translate ``-C`` config settings into ``build_lib.py`` CLI arguments."""
    if not config_settings:
        return []

    # PEP 517: config_settings values may be str or list[str].
    # Normalize list values to their last element.
    for key, value in config_settings.items():
        if isinstance(value, list):
            config_settings[key] = value[-1] if value else ""

    args: list[str] = []

    # Boolean flags: config key -> (CLI flag when true, CLI flag when false)
    bool_flags = {
        "warp-no-cuda": ("--no-cuda", None),
        "warp-quick": ("--quick", None),
        "warp-no-standalone": ("--no-standalone", None),
        "warp-verbose": ("--verbose", "--no-verbose"),
        "warp-verify-fp": ("--verify-fp", None),
        "warp-fast-math": ("--fast-math", None),
        "warp-build-llvm": ("--build-llvm", None),
        "warp-debug-llvm": ("--debug-llvm", None),
        "warp-clang-build-toolchain": ("--clang-build-toolchain", None),
        "warp-compile-time-trace": ("--compile-time-trace", None),
        "warp-no-use-libmathdx": ("--no-use-libmathdx", None),
    }

    for key, (flag_true, flag_false) in bool_flags.items():
        value = config_settings.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.lower() in ("true", "1", "yes")
        if value and flag_true:
            args.append(flag_true)
        elif not value and flag_false:
            args.append(flag_false)

    # Value flags: config key -> CLI flag
    value_flags = {
        "warp-mode": "--mode",
        "warp-jobs": "-j",
        "warp-msvc-path": "--msvc-path",
        "warp-sdk-path": "--sdk-path",
        "warp-cuda-path": "--cuda-path",
        "warp-libmathdx-path": "--libmathdx-path",
        "warp-llvm-path": "--llvm-path",
        "warp-llvm-source-path": "--llvm-source-path",
    }

    for key, flag in value_flags.items():
        value = config_settings.get(key)
        if value is not None:
            args.extend([flag, str(value)])

    return args


def _ensure_native_libs(config_settings: dict | None = None) -> None:
    """Build native libraries if they are not already present."""
    if _has_runtime_library():
        return

    print("Warp native libraries not found in warp/bin/ — running build_lib.py ...")

    # build_lib.py lives at the repo root (one level above warp/)
    repo_root = os.path.dirname(os.path.dirname(__file__))
    build_script = os.path.join(repo_root, "build_lib.py")

    if not os.path.isfile(build_script):
        raise FileNotFoundError(
            f"build_lib.py not found at {build_script}. "
            "Ensure you are building from a complete source tree or sdist."
        )

    cmd = [sys.executable, build_script] + _build_lib_args_from_config(config_settings)
    result = subprocess.run(cmd, cwd=repo_root)
    if result.returncode != 0:
        raise RuntimeError(
            f"build_lib.py failed with exit code {result.returncode}. "
            "Check the build output above for details."
        )

    if not _has_runtime_library():
        raise RuntimeError(
            f"build_lib.py completed but {_runtime_lib_name()} was not found in warp/bin/. "
            "The build may have failed silently."
        )


def build_wheel(
    wheel_directory: str,
    config_settings: dict | None = None,
    metadata_directory: str | None = None,
) -> str:
    _ensure_native_libs(config_settings)
    return _build_wheel(wheel_directory, config_settings=config_settings, metadata_directory=metadata_directory)


def build_editable(
    wheel_directory: str,
    config_settings: dict | None = None,
    metadata_directory: str | None = None,
) -> str:
    _ensure_native_libs(config_settings)
    return _build_editable(wheel_directory, config_settings=config_settings, metadata_directory=metadata_directory)
