#!/bin/bash
# Build libnvshmem_device.ltoir from NVSHMEM source.
#
# This produces the LTOIR file that Warp links into NVSHMEM-enabled kernels
# via nvJitLink at runtime. The file should be passed to build_lib.py via
# --nvshmem-ltoir or placed in warp/bin/ before building.
#
# Usage (docker, no local dependencies):
#   ./tools/build_nvshmem_device_ltoir.sh
#
# Usage (local, requires NVSHMEM source + CUDA Toolkit):
#   NVSHMEM_SRC=/path/to/nvshmem ./tools/build_nvshmem_device_ltoir.sh
#
# Output: warp/bin/libnvshmem_device.ltoir

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT="$REPO_DIR/warp/bin/libnvshmem_device.ltoir"
NVSHMEM_VERSION="v3.6.5-0"

if [ -n "${NVSHMEM_SRC:-}" ] && [ -d "$NVSHMEM_SRC/src" ]; then
    # Local build from existing source checkout
    echo "Building from local NVSHMEM source: $NVSHMEM_SRC"
    BUILD_DIR=$(mktemp -d)
    trap "rm -rf $BUILD_DIR" EXIT

    # Ensure nvcc is on PATH
    if ! command -v nvcc &> /dev/null; then
        if [ -d "${CUDA_HOME:-/usr/local/cuda}/bin" ]; then
            export PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
        fi
    fi

    cmake -S "$NVSHMEM_SRC" -B "$BUILD_DIR" \
        -DNVSHMEM_BUILD_LTOIR_LIBRARY=ON \
        -DCMAKE_CUDA_ARCHITECTURES=80 \
        -DNVSHMEM_MPI_SUPPORT=OFF \
        -DNVSHMEM_SHMEM_SUPPORT=OFF \
        -DNVSHMEM_PMIX_SUPPORT=OFF \
        -DNVSHMEM_BUILD_TESTS=OFF \
        -DNVSHMEM_BUILD_EXAMPLES=OFF \
        -DNVSHMEM_IBGDA_SUPPORT=OFF \
        -DNVSHMEM_USE_GDRCOPY=OFF \
        -DNVSHMEM_USE_NCCL=OFF

    make -C "$BUILD_DIR" -j"$(nproc)" libnvshmem_device_ltoir
    cp "$BUILD_DIR/src/lib/libnvshmem_device.ltoir" "$OUTPUT"
else
    # Docker build (no local NVSHMEM source needed)
    echo "Building via docker (nvcr.io/nvidia/nvhpc:26.3-devel-cuda13.1-ubuntu24.04)..."
    docker run --rm -v "$REPO_DIR/warp/bin:/output" \
        nvcr.io/nvidia/nvhpc:26.3-devel-cuda13.1-ubuntu24.04 bash -c "
        apt-get update -qq && apt-get install -y -qq python3 > /dev/null 2>&1
        git clone --depth 1 --branch $NVSHMEM_VERSION https://github.com/NVIDIA/nvshmem.git /tmp/nvshmem
        NVCC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/cuda/13.1/bin/nvcc
        cmake -S /tmp/nvshmem -B /tmp/build \
            -DCMAKE_CUDA_COMPILER=\$NVCC \
            -DNVSHMEM_BUILD_LTOIR_LIBRARY=ON \
            -DCMAKE_CUDA_ARCHITECTURES=80 \
            -DNVSHMEM_MPI_SUPPORT=OFF \
            -DNVSHMEM_SHMEM_SUPPORT=OFF \
            -DNVSHMEM_PMIX_SUPPORT=OFF \
            -DNVSHMEM_BUILD_TESTS=OFF \
            -DNVSHMEM_BUILD_EXAMPLES=OFF \
            -DNVSHMEM_IBGDA_SUPPORT=OFF \
            -DNVSHMEM_USE_GDRCOPY=OFF \
            -DNVSHMEM_USE_NCCL=OFF
        make -C /tmp/build -j\$(nproc) libnvshmem_device_ltoir
        cp /tmp/build/src/lib/libnvshmem_device.ltoir /output/
    "
fi

echo "Built: $OUTPUT ($(stat -c%s "$OUTPUT") bytes)"
