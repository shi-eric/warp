// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "builtin.h"

extern "C" {

WP_API int wp_is_nvshmem_enabled();
WP_API uint64_t wp_nvshmem_malloc(size_t nbytes);
WP_API void wp_nvshmem_free(uint64_t ptr);
WP_API int wp_nvshmem_cumodule_init(uint64_t module);
WP_API int wp_nvshmem_cumodule_finalize(uint64_t module);

}  // extern "C"
