#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Consumer smoke test for Linux SDKs, run inside the matching manylinux
# container: link warp-clang against the deployed SDK at the manylinux
# glibc floor, verify the linked library's glibc symbol-version ceiling,
# and JIT-execute a CPU kernel. Required env: SDK_DIR, GLIBC_CEILING.
set -euo pipefail

PYTHON=/opt/python/cp312-cp312/bin/python

"$PYTHON" -m pip install --quiet numpy

"$PYTHON" build_lib.py --llvm-path "$SDK_DIR"

"$PYTHON" - "$GLIBC_CEILING" <<'EOF'
import re
import subprocess
import sys

# gcc-toolset-14's static-libgcc unwinder references _dl_find_object, which
# AlmaLinux 9's ld.so backports with its upstream GLIBC_2.35 version tag.
# It is injected by the image's toolchain, not by the SDK (the SDK archives
# carry no versioned glibc refs), so it does not count against the SDK's
# manylinux ceiling.
TOOLCHAIN_INJECTED = {"_dl_find_object@GLIBC_2.35"}

ceiling = tuple(int(p) for p in sys.argv[1].split("."))
out = subprocess.check_output(
    ["readelf", "--dyn-syms", "--wide", "warp/bin/warp-clang.so"], text=True
)
offenders = {}
for name, major, minor in re.findall(r"(\S+)@+(?:GLIBC_)(\d+)\.(\d+)", out):
    symbol = f"{name}@GLIBC_{major}.{minor}"
    if symbol in TOOLCHAIN_INJECTED:
        continue
    version = (int(major), int(minor))
    if version > ceiling:
        offenders[symbol] = version
assert not offenders, f"glibc symbols above {ceiling}: {sorted(offenders)}"
print(f"glibc ceiling OK (max allowed {ceiling})")
EOF

cat > _smoke_test.py <<'PYEOF'
import numpy as np

import warp as wp


@wp.kernel
def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = a[tid] + b[tid]


with wp.ScopedDevice("cpu"):
    n = 1024
    a = wp.array(np.arange(n, dtype=np.float32))
    b = wp.array(np.ones(n, dtype=np.float32))
    out = wp.zeros(n, dtype=float)
    wp.launch(add_kernel, dim=n, inputs=[a, b], outputs=[out])
    np.testing.assert_allclose(out.numpy(), np.arange(n, dtype=np.float32) + 1.0)
print("CPU JIT smoke test passed")
PYEOF
"$PYTHON" _smoke_test.py
