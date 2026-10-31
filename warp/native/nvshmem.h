// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "builtin.h"

extern "C" {

// Whether this build can link NVSHMEM device code into kernels (Linux CUDA builds).
WP_API int wp_is_nvshmem_enabled();

// Version of the NVSHMEM host library loaded in the process, packed as (major << 16) | (minor << 8) | patch,
// or 0 when none is loaded.
WP_API uint32_t wp_nvshmem_get_loaded_version();

// Copies the path of the loaded NVSHMEM host library into `buffer`. Returns the path length, 0 when no host
// library is loaded, or -1 when `size` cannot hold the path and its terminator.
WP_API int wp_nvshmem_get_host_library_path(char* buffer, size_t size);

// Register or unregister a loaded CUDA module with the NVSHMEM runtime. Returns -1 when no host library is
// loaded or NVSHMEM is not initialized; otherwise NVSHMEM's own return value.
WP_API int wp_nvshmem_cumodule_init(uint64_t module);
WP_API int wp_nvshmem_cumodule_finalize(uint64_t module);

// Whether `ptr` is an address in the NVSHMEM symmetric heap of the calling PE. Returns 1 when it is, 0 when it is
// not, or -1 when this cannot be determined: no host library is loaded, NVSHMEM is not initialized, or the host
// library lacks nvshmem_ptr() or nvshmem_my_pe().
WP_API int wp_nvshmem_is_symmetric(uint64_t ptr);

}  // extern "C"
