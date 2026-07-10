// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "builtin.h"

#define WP_NVSHMEM_DEVICE_LIBRARY_NONE   0
#define WP_NVSHMEM_DEVICE_LIBRARY_LTOIR  1
#define WP_NVSHMEM_DEVICE_LIBRARY_FATBIN 2

extern "C" {

WP_API int wp_is_nvshmem_enabled();
WP_API uint32_t wp_nvshmem_get_build_version();
WP_API uint32_t wp_nvshmem_get_loaded_version();
WP_API const void* wp_nvshmem_get_device_library(size_t* size, int* kind);
WP_API uint64_t wp_nvshmem_malloc(size_t nbytes);
WP_API void wp_nvshmem_free(uint64_t ptr);
WP_API int wp_nvshmem_cumodule_init(uint64_t module);
WP_API int wp_nvshmem_cumodule_finalize(uint64_t module);

}  // extern "C"
