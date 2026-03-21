// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Lightweight header for tile shared memory storage.
// Extracted from tile.h so that kernel templates can reference
// tile_shared_storage_t without pulling in the full tile machinery.
// This file is included unconditionally from builtin.h (outside the
// WP_NO_TILE guard), so it must NOT include tile.h, builtin.h, or rand.h.

#pragma once

#if defined(__CUDACC_RTC__)
#define WP_TILE_THREAD_IDX threadIdx.x
#else
#define WP_TILE_THREAD_IDX 0
#endif  //

namespace wp {

inline CUDA_CALLABLE int tile_align(int num_bytes)
{
    // note this much match value in Python types.py
    const int alignment = 16;

    const int num_bytes_abs = num_bytes < 0 ? -num_bytes : num_bytes;
    const int sign = num_bytes < 0 ? -1 : 1;

    return sign * ((num_bytes_abs + alignment - 1) / alignment) * alignment;
}

#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
// On the CPU we use a fixed size block of stack memory for shared tile allocations.
// We store a pointer to the current allocation storage either in a reserved register
// (AArch64) or a static variable (x86-64).
#if !defined(__CUDA_ARCH__)
class tile_shared_storage_t;
#if defined(__aarch64__)
// x28 is is the last callee-saved register on AArch64. This allows us to call externally
// compiled functions without worrying about clobbering the pointer.
// We pass -target-feature +reserve-x28 to Clang to exclude it from register allocation.
register tile_shared_storage_t* shared_tile_storage asm("x28");
#else
// Ideally this would be thread_local, but LLVM's JIT doesn't support TLS yet
// There is also no support for something like -ffixed-r15 either
static tile_shared_storage_t* shared_tile_storage;
#endif
#endif
#endif

// This class manages a block of "shared" memory for use by tiles.
// On the GPU this maps to dynamic shared memory, while on the CPU we allocate
// a fixed size block of memory on the stack and manage allocations from it.
// An instance of this class gets created at the start of a kernel.
class tile_shared_storage_t {
private:
#if !defined(__CUDA_ARCH__)
#define WP_MAX_CPU_SHARED 256*1024
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    tile_shared_storage_t* old_value;
    unsigned int smem_base[WP_TILE_BLOCK_DIM];
    char dynamic_smem_base[WP_MAX_CPU_SHARED];  // on CPU allocate a fixed 256k block to use for shared allocs
#endif
#endif

    // we maintain a per-thread offset into dynamic
    // shared memory that allows us to keep track of
    // current use across dynamic function calls
    static inline CUDA_CALLABLE unsigned int* get_smem_base()
    {
#if defined(__CUDA_ARCH__)
        __shared__ unsigned int smem_base[WP_TILE_BLOCK_DIM];
        return smem_base;
#elif defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
        return shared_tile_storage->smem_base;
#else
        static unsigned int smem_base[WP_TILE_BLOCK_DIM];
        return smem_base;
#endif
    }

    static inline CUDA_CALLABLE char* get_dynamic_smem_base()
    {
#if defined(__CUDA_ARCH__)
        extern __shared__ char dynamic_smem_base[];
        return dynamic_smem_base;
#elif defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
        return shared_tile_storage->dynamic_smem_base;
#else
        static char dynamic_smem_base[WP_MAX_CPU_SHARED];
        return dynamic_smem_base;
#endif
    }

public:
    // cppcheck-suppress uninitMemberVar
    inline CUDA_CALLABLE tile_shared_storage_t()
    {
#if !defined(__CUDA_ARCH__) && defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
        // On the CPU save a pointer to this instance in a reserved register
        // or static variable so it can be accessed from anywhere within a kernel.
        old_value = shared_tile_storage;
        shared_tile_storage = this;
#endif

        init();
    }

    inline CUDA_CALLABLE ~tile_shared_storage_t()
    {
        check();

#if !defined(__CUDA_ARCH__) && defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
        shared_tile_storage = old_value;
#endif
    }

    static inline CUDA_CALLABLE void init()
    {
        unsigned int* smem_base = get_smem_base();

        smem_base[WP_TILE_THREAD_IDX] = 0;
    }

    static inline CUDA_CALLABLE void check()
    {
        unsigned int* smem_base = get_smem_base();

        assert(smem_base[WP_TILE_THREAD_IDX] == 0);
    }

    static inline CUDA_CALLABLE void* alloc(int num_bytes)
    {
        unsigned int* smem_base = get_smem_base();
        char* dynamic_smem_base = get_dynamic_smem_base();

        const unsigned int offset = smem_base[WP_TILE_THREAD_IDX];

        // one entry per-thread so no need for synchronization
        smem_base[WP_TILE_THREAD_IDX] += tile_align(num_bytes);

#if !defined(__CUDA_ARCH__)
        assert(smem_base[WP_TILE_THREAD_IDX] <= WP_MAX_CPU_SHARED);
#endif

        return &(dynamic_smem_base[offset]);
    }
};

}  // namespace wp
