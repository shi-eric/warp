// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nvshmem.h"

#if !WP_ENABLE_CUDA || !WP_ENABLE_NVSHMEM

extern "C" {

int wp_is_nvshmem_enabled() { return 0; }

uint32_t wp_nvshmem_get_loaded_version() { return 0; }

int wp_nvshmem_get_host_library_path(char* buffer, size_t size)
{
    (void)buffer;
    (void)size;
    return 0;
}

int wp_nvshmem_cumodule_init(uint64_t module)
{
    printf("Warp was not compiled with NVSHMEM support.\n");
    return -1;
}

int wp_nvshmem_cumodule_finalize(uint64_t module)
{
    printf("Warp was not compiled with NVSHMEM support.\n");
    return -1;
}

}  // extern "C"

#else  // WP_ENABLE_CUDA && WP_ENABLE_NVSHMEM

#include <dlfcn.h>
#include <string.h>

// Function pointer types for the NVSHMEM host API functions we need.
// The NVSHMEM headers are not included: the host library is found at runtime, and the device library is
// linked from the installed NVSHMEM, so Warp builds without any NVSHMEM files.
typedef int (*nvshmemx_cumodule_init_fn)(void*);
typedef int (*nvshmemx_cumodule_finalize_fn)(void*);
typedef int (*nvshmemx_init_status_fn)();
typedef void (*nvshmemx_vendor_get_version_info_fn)(int*, int*, int*);

static void* s_nvshmem_host_lib = nullptr;
static nvshmemx_cumodule_init_fn s_nvshmemx_cumodule_init = nullptr;
static nvshmemx_cumodule_finalize_fn s_nvshmemx_cumodule_finalize = nullptr;
static nvshmemx_init_status_fn s_nvshmemx_init_status = nullptr;
static nvshmemx_vendor_get_version_info_fn s_nvshmemx_vendor_get_version_info = nullptr;

static uint32_t encode_nvshmem_version(int major, int minor, int patch)
{
    return (uint32_t(major) << 16) | (uint32_t(minor) << 8) | uint32_t(patch);
}

static bool ensure_nvshmem_host_loaded()
{
    if (s_nvshmem_host_lib)
        return true;

    // Use RTLD_NOLOAD to find the libnvshmem_host.so that the application (typically nvshmem4py) already
    // loaded. This avoids loading a different copy from the system path.
    s_nvshmem_host_lib = dlopen("libnvshmem_host.so", RTLD_NOW | RTLD_NOLOAD);
    if (!s_nvshmem_host_lib) {
        s_nvshmem_host_lib = dlopen("libnvshmem_host.so.3", RTLD_NOW | RTLD_NOLOAD);
    }
    if (!s_nvshmem_host_lib) {
        // Not loaded yet. The application must initialize NVSHMEM first.
        return false;
    }

    s_nvshmemx_cumodule_init = (nvshmemx_cumodule_init_fn)dlsym(s_nvshmem_host_lib, "nvshmemx_cumodule_init");
    s_nvshmemx_cumodule_finalize
        = (nvshmemx_cumodule_finalize_fn)dlsym(s_nvshmem_host_lib, "nvshmemx_cumodule_finalize");
    s_nvshmemx_init_status = (nvshmemx_init_status_fn)dlsym(s_nvshmem_host_lib, "nvshmemx_init_status");
    s_nvshmemx_vendor_get_version_info
        = (nvshmemx_vendor_get_version_info_fn)dlsym(s_nvshmem_host_lib, "nvshmemx_vendor_get_version_info");

    if (!s_nvshmemx_cumodule_init || !s_nvshmemx_cumodule_finalize || !s_nvshmemx_init_status
        || !s_nvshmemx_vendor_get_version_info) {
        printf("Failed to resolve NVSHMEM host symbols: %s\n", dlerror());
        s_nvshmem_host_lib = nullptr;
        return false;
    }

    return true;
}

extern "C" {

int wp_is_nvshmem_enabled() { return 1; }

uint32_t wp_nvshmem_get_loaded_version()
{
    if (!ensure_nvshmem_host_loaded())
        return 0;

    int major = 0;
    int minor = 0;
    int patch = 0;
    s_nvshmemx_vendor_get_version_info(&major, &minor, &patch);
    return encode_nvshmem_version(major, minor, patch);
}

int wp_nvshmem_get_host_library_path(char* buffer, size_t size)
{
    if (!ensure_nvshmem_host_loaded())
        return 0;

    Dl_info info;
    if (dladdr((void*)s_nvshmemx_cumodule_init, &info) == 0 || info.dli_fname == nullptr)
        return 0;

    const size_t length = strlen(info.dli_fname);
    if (buffer == nullptr || size <= length)
        return -1;

    memcpy(buffer, info.dli_fname, length + 1);
    return int(length);
}

int wp_nvshmem_cumodule_init(uint64_t module)
{
    if (!ensure_nvshmem_host_loaded())
        return -1;
    // nvshmemx_init_status() returns >= 2 (INITIALIZED) when ready.
    if (s_nvshmemx_init_status() < 2)
        return -1;
    return s_nvshmemx_cumodule_init((void*)module);
}

int wp_nvshmem_cumodule_finalize(uint64_t module)
{
    if (!ensure_nvshmem_host_loaded())
        return -1;
    if (s_nvshmemx_init_status() < 2)
        return -1;
    return s_nvshmemx_cumodule_finalize((void*)module);
}

}  // extern "C"

#endif  // !WP_ENABLE_CUDA || !WP_ENABLE_NVSHMEM
