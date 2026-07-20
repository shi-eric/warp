#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runs inside a manylinux container: build, deploy, package, and check one
# Linux LLVM SDK. Required env: LLVM_VERSION, BUNDLE_REVISION, PROFILE
# (e.g. linux-x86_64), IMAGE_DIGEST, OUTPUT_DIR (host-mounted).
set -euo pipefail

CONAN_VERSION=2.30.0

# pipx puts console scripts on ~/.local/bin regardless of which Python the
# manylinux image exposes; plain pip's script location varies by image.
pipx install "conan==${CONAN_VERSION}"
pipx install cmake==3.31.6
pipx install ninja==1.11.1.4
export PATH="$HOME/.local/bin:$PATH"

conan profile detect --force
conan create tools/llvm --version "${LLVM_VERSION}" \
  -pr:h "tools/llvm/profiles/${PROFILE}" -pr:b default

conan install --requires "clang-warp/${LLVM_VERSION}" \
  -pr:h "tools/llvm/profiles/${PROFILE}" -pr:b default \
  --deployer tools/llvm/deployers/llvm_sdk.py --deployer-folder _sdk_deploy

python3 tools/llvm/check_sdk.py --sdk-dir _sdk_deploy/llvm-sdk --platform "${PROFILE}"

python3 tools/llvm/package_sdk.py \
  --sdk-dir _sdk_deploy/llvm-sdk \
  --profile "tools/llvm/profiles/${PROFILE}" \
  --llvm-version "${LLVM_VERSION}" \
  --bundle-revision "${BUNDLE_REVISION}" \
  --output-dir "${OUTPUT_DIR}" \
  --recipe-sha "${GITHUB_SHA:-unknown}" \
  --image-digest "${IMAGE_DIGEST}" \
  --conan-version "${CONAN_VERSION}" \
  --toolchain-info "$(gcc --version | head -1)"
