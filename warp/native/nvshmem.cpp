// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nvshmem.h"

#if !WP_ENABLE_CUDA || !WP_ENABLE_NVSHMEM

extern "C" {

int wp_is_nvshmem_enabled() { return 0; }

uint32_t wp_nvshmem_get_build_version() { return 0; }

uint32_t wp_nvshmem_get_loaded_version() { return 0; }

const void* wp_nvshmem_get_device_library(size_t* size, int* kind)
{
    *size = 0;
    *kind = WP_NVSHMEM_DEVICE_LIBRARY_NONE;
    return nullptr;
}

uint64_t wp_nvshmem_malloc(size_t nbytes)
{
    printf("Warp was not compiled with NVSHMEM support.\n");
    return 0;
}

void wp_nvshmem_free(uint64_t ptr) { printf("Warp was not compiled with NVSHMEM support.\n"); }

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
#include <non_abi/nvshmem_version.h>

#ifndef WP_NVSHMEM_DEVICE_LIBRARY_KIND
#define WP_NVSHMEM_DEVICE_LIBRARY_KIND WP_NVSHMEM_DEVICE_LIBRARY_NONE
#endif

#if WP_NVSHMEM_DEVICE_LIBRARY_KIND != WP_NVSHMEM_DEVICE_LIBRARY_NONE
extern "C" {
extern const unsigned char wp_nvshmem_device_library_start[];
extern const unsigned char wp_nvshmem_device_library_end[];
}
#endif

// Function pointer types for the NVSHMEM host API functions we need.
// We cannot #include <nvshmem.h> because the headers pull in CCCL (cuda/std/tuple, etc.)
// which is unavailable to the host C++ compiler (g++).
typedef void* (*nvshmem_malloc_fn)(size_t);
typedef void (*nvshmem_free_fn)(void*);
typedef int (*nvshmemx_cumodule_init_fn)(void*);
typedef int (*nvshmemx_cumodule_finalize_fn)(void*);
typedef int (*nvshmemx_init_status_fn)();
typedef void (*nvshmemx_vendor_get_version_info_fn)(int*, int*, int*);

static void* s_nvshmem_host_lib = nullptr;
static nvshmem_malloc_fn s_nvshmem_malloc = nullptr;
static nvshmem_free_fn s_nvshmem_free = nullptr;
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

    // Use RTLD_NOLOAD to find the libnvshmem_host.so that nvshmem4py already loaded
    // during nvshmem.init(). This avoids loading a different copy from the system path.
    s_nvshmem_host_lib = dlopen("libnvshmem_host.so", RTLD_NOW | RTLD_NOLOAD);
    if (!s_nvshmem_host_lib) {
        s_nvshmem_host_lib = dlopen("libnvshmem_host.so.3", RTLD_NOW | RTLD_NOLOAD);
    }
    if (!s_nvshmem_host_lib) {
        // Not yet loaded by nvshmem4py. The user must call nvshmem.init() first.
        return false;
    }

    s_nvshmem_malloc = (nvshmem_malloc_fn)dlsym(s_nvshmem_host_lib, "nvshmem_malloc");
    s_nvshmem_free = (nvshmem_free_fn)dlsym(s_nvshmem_host_lib, "nvshmem_free");
    s_nvshmemx_cumodule_init = (nvshmemx_cumodule_init_fn)dlsym(s_nvshmem_host_lib, "nvshmemx_cumodule_init");
    s_nvshmemx_cumodule_finalize
        = (nvshmemx_cumodule_finalize_fn)dlsym(s_nvshmem_host_lib, "nvshmemx_cumodule_finalize");
    s_nvshmemx_init_status = (nvshmemx_init_status_fn)dlsym(s_nvshmem_host_lib, "nvshmemx_init_status");
    s_nvshmemx_vendor_get_version_info
        = (nvshmemx_vendor_get_version_info_fn)dlsym(s_nvshmem_host_lib, "nvshmemx_vendor_get_version_info");

    if (!s_nvshmem_malloc || !s_nvshmem_free || !s_nvshmemx_cumodule_init || !s_nvshmemx_cumodule_finalize
        || !s_nvshmemx_init_status || !s_nvshmemx_vendor_get_version_info) {
        printf("Failed to resolve NVSHMEM host symbols: %s\n", dlerror());
        s_nvshmem_host_lib = nullptr;
        return false;
    }

    return true;
}

extern "C" {

int wp_is_nvshmem_enabled() { return 1; }

uint32_t wp_nvshmem_get_build_version()
{
    return encode_nvshmem_version(
        NVSHMEM_VENDOR_MAJOR_VERSION, NVSHMEM_VENDOR_MINOR_VERSION, NVSHMEM_VENDOR_PATCH_VERSION
    );
}

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

const void* wp_nvshmem_get_device_library(size_t* size, int* kind)
{
#if WP_NVSHMEM_DEVICE_LIBRARY_KIND != WP_NVSHMEM_DEVICE_LIBRARY_NONE
    *size = size_t(wp_nvshmem_device_library_end - wp_nvshmem_device_library_start);
    *kind = WP_NVSHMEM_DEVICE_LIBRARY_KIND;
    return wp_nvshmem_device_library_start;
#else
    *size = 0;
    *kind = WP_NVSHMEM_DEVICE_LIBRARY_NONE;
    return nullptr;
#endif
}

uint64_t wp_nvshmem_malloc(size_t nbytes)
{
    if (!ensure_nvshmem_host_loaded()) {
        printf("NVSHMEM host library not loaded. Call nvshmem.init() before allocating symmetric memory.\n");
        return 0;
    }
    void* ptr = s_nvshmem_malloc(nbytes);
    return (uint64_t)ptr;
}

void wp_nvshmem_free(uint64_t ptr)
{
    if (!ensure_nvshmem_host_loaded())
        return;
    s_nvshmem_free((void*)ptr);
}

int wp_nvshmem_cumodule_init(uint64_t module)
{
    if (!ensure_nvshmem_host_loaded())
        return -1;
    if (wp_nvshmem_get_loaded_version() != wp_nvshmem_get_build_version())
        return -2;
    // nvshmemx_init_status() returns >= 2 (INITIALIZED) when ready.
    if (s_nvshmemx_init_status() < 2)
        return -1;
    return s_nvshmemx_cumodule_init((void*)module);
}

int wp_nvshmem_cumodule_finalize(uint64_t module)
{
    if (!ensure_nvshmem_host_loaded())
        return -1;
    if (wp_nvshmem_get_loaded_version() != wp_nvshmem_get_build_version())
        return -2;
    if (s_nvshmemx_init_status() < 2)
        return -1;
    return s_nvshmemx_cumodule_finalize((void*)module);
}

}  // extern "C"

#endif  // !WP_ENABLE_CUDA || !WP_ENABLE_NVSHMEM
