// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Device-side NVSHMEM built-ins.
//
// Modules that call these functions are linked against the NVSHMEM device library through nvJitLink,
// so only the NVSHMEM entry points are declared here. The NVSHMEM headers cannot be included because
// they depend on CUDA C++ headers that NVRTC does not provide.
//
// Elements move through the unsigned-integer NVSHMEM variants sized to the element's alignment, so any
// Warp dtype transfers bit for bit, and elements wider than 8 bytes move as several aligned parts.

#if defined(__CUDACC__)

// The declarations below use the NVSHMEM headers' own parameter types. NVRTC and nvcc provide ptrdiff_t;
// Clang-compiled kernels see no C headers, so take it from the compiler.
#if defined(__clang__) && defined(__CUDA__) && !defined(__CUDACC_RTC__)
typedef __PTRDIFF_TYPE__ ptrdiff_t;
#endif

extern "C" {

__device__ int nvshmem_my_pe();
__device__ int nvshmem_n_pes();
__device__ void nvshmem_quiet();
__device__ void nvshmem_fence();
__device__ void nvshmem_barrier_all();
__device__ void nvshmem_putmem(void* dest, const void* source, size_t bytes, int pe);
__device__ void nvshmem_getmem(void* dest, const void* source, size_t bytes, int pe);
__device__ void nvshmem_putmem_signal(
    void* dest, const void* source, size_t bytes, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe
);
__device__ void nvshmemx_signal_op(uint64_t* sig_addr, uint64_t signal, int sig_op, int pe);
__device__ uint64_t nvshmem_signal_wait_until(uint64_t* sig_addr, int cmp, uint64_t cmp_value);

#define WP_NVSHMEM_DECLARE_SIZED_RMA(BITS)                                                                             \
    __device__ void nvshmem_uint##BITS##_p(uint##BITS##_t* dest, uint##BITS##_t value, int pe);                        \
    __device__ uint##BITS##_t nvshmem_uint##BITS##_g(const uint##BITS##_t* source, int pe);                            \
    __device__ void nvshmem_uint##BITS##_iput(                                                                         \
        uint##BITS##_t* dest, const uint##BITS##_t* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe        \
    );                                                                                                                 \
    __device__ void nvshmem_uint##BITS##_iget(                                                                         \
        uint##BITS##_t* dest, const uint##BITS##_t* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe        \
    );

WP_NVSHMEM_DECLARE_SIZED_RMA(8)
WP_NVSHMEM_DECLARE_SIZED_RMA(16)
WP_NVSHMEM_DECLARE_SIZED_RMA(32)
WP_NVSHMEM_DECLARE_SIZED_RMA(64)

#undef WP_NVSHMEM_DECLARE_SIZED_RMA

}  // extern "C"

namespace wp {

namespace nvshmem_detail {

template <int Bytes> struct rma;

#define WP_NVSHMEM_SIZED_RMA(BYTES, BITS)                                                                              \
    template <> struct rma<BYTES> {                                                                                    \
        using type = uint##BITS##_t;                                                                                   \
                                                                                                                       \
        static inline CUDA_CALLABLE_DEVICE void p(type* dest, type value, int pe)                                      \
        {                                                                                                              \
            ::nvshmem_uint##BITS##_p(dest, value, pe);                                                                 \
        }                                                                                                              \
        static inline CUDA_CALLABLE_DEVICE type g(const type* source, int pe)                                          \
        {                                                                                                              \
            return ::nvshmem_uint##BITS##_g(source, pe);                                                               \
        }                                                                                                              \
        static inline CUDA_CALLABLE_DEVICE void                                                                        \
        iput(type* dest, const type* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe)                      \
        {                                                                                                              \
            ::nvshmem_uint##BITS##_iput(dest, source, dst, sst, nelems, pe);                                           \
        }                                                                                                              \
        static inline CUDA_CALLABLE_DEVICE void                                                                        \
        iget(type* dest, const type* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe)                      \
        {                                                                                                              \
            ::nvshmem_uint##BITS##_iget(dest, source, dst, sst, nelems, pe);                                           \
        }                                                                                                              \
    };

WP_NVSHMEM_SIZED_RMA(1, 8)
WP_NVSHMEM_SIZED_RMA(2, 16)
WP_NVSHMEM_SIZED_RMA(4, 32)
WP_NVSHMEM_SIZED_RMA(8, 64)

#undef WP_NVSHMEM_SIZED_RMA

// An element moves as `count` parts of `bytes` each: its alignment, capped at the widest NVSHMEM scalar.
template <typename T> struct parts {
    static constexpr int bytes = alignof(T) < 8 ? int(alignof(T)) : 8;
    static constexpr int count = int(sizeof(T)) / bytes;
    using ops = rma<bytes>;
    using type = typename ops::type;
};

template <typename T> inline CUDA_CALLABLE_DEVICE void put_element(T* dest, const T& value, int pe)
{
    using P = parts<T>;
    typename P::type values[P::count];
    memcpy(values, &value, sizeof(T));
    typename P::type* dest_parts = reinterpret_cast<typename P::type*>(dest);
    for (int c = 0; c < P::count; ++c) {
        P::ops::p(dest_parts + c, values[c], pe);
    }
}

template <typename T> inline CUDA_CALLABLE_DEVICE T get_element(const T* source, int pe)
{
    using P = parts<T>;
    typename P::type values[P::count];
    const typename P::type* source_parts = reinterpret_cast<const typename P::type*>(source);
    for (int c = 0; c < P::count; ++c) {
        values[c] = P::ops::g(source_parts + c, pe);
    }
    T result;
    memcpy(&result, values, sizeof(T));
    return result;
}

template <typename T> inline CUDA_CALLABLE_DEVICE size_t element_count(const array_t<T>& a)
{
    size_t count = 1;
    for (int d = 0; d < a.ndim; ++d) {
        count *= size_t(a.shape[d]);
    }
    return count;
}

// Whether the elements are packed in row-major order, so the array can move as one block of bytes.
template <typename T> inline CUDA_CALLABLE_DEVICE bool is_packed(const array_t<T>& a)
{
    long long expected = sizeof(T);
    for (int d = a.ndim - 1; d >= 0; --d) {
        if (a.shape[d] > 1 && a.strides[d] != expected) {
            return false;
        }
        expected *= a.shape[d];
    }
    return true;
}

// Address of the n-th element in row-major order, honoring strides.
template <typename T> inline CUDA_CALLABLE_DEVICE T* element_at(const array_t<T>& a, size_t n)
{
    long long offset = 0;
    for (int d = a.ndim - 1; d >= 0; --d) {
        const size_t extent = size_t(a.shape[d]);
        offset += (long long)(n % extent) * a.strides[d];
        n /= extent;
    }
    return reinterpret_cast<T*>(reinterpret_cast<char*>(a.data) + offset);
}

template <typename T> inline CUDA_CALLABLE_DEVICE void assert_same_shape(const array_t<T>& a, const array_t<T>& b)
{
    assert(a.ndim == b.ndim);
    for (int d = 0; d < a.ndim; ++d) {
        assert(a.shape[d] == b.shape[d]);
    }
}

}  // namespace nvshmem_detail

inline CUDA_CALLABLE_DEVICE int32 nvshmem_my_pe() { return ::nvshmem_my_pe(); }

inline CUDA_CALLABLE_DEVICE int32 nvshmem_n_pes() { return ::nvshmem_n_pes(); }

inline CUDA_CALLABLE_DEVICE void nvshmem_quiet() { ::nvshmem_quiet(); }

inline CUDA_CALLABLE_DEVICE void nvshmem_fence() { ::nvshmem_fence(); }

inline CUDA_CALLABLE_DEVICE void nvshmem_barrier_all() { ::nvshmem_barrier_all(); }

template <typename T> inline CUDA_CALLABLE_DEVICE void nvshmem_p(const array_t<T>& dest, int i, T value, int pe)
{
    nvshmem_detail::put_element(&index(dest, i), value, pe);
}

template <typename T> inline CUDA_CALLABLE_DEVICE void nvshmem_p(const array_t<T>& dest, int i, int j, T value, int pe)
{
    nvshmem_detail::put_element(&index(dest, i, j), value, pe);
}

template <typename T>
inline CUDA_CALLABLE_DEVICE void nvshmem_p(const array_t<T>& dest, int i, int j, int k, T value, int pe)
{
    nvshmem_detail::put_element(&index(dest, i, j, k), value, pe);
}

template <typename T>
inline CUDA_CALLABLE_DEVICE void nvshmem_p(const array_t<T>& dest, int i, int j, int k, int l, T value, int pe)
{
    nvshmem_detail::put_element(&index(dest, i, j, k, l), value, pe);
}

template <typename T> inline CUDA_CALLABLE_DEVICE T nvshmem_g(const array_t<T>& source, int i, int pe)
{
    return nvshmem_detail::get_element(&index(source, i), pe);
}

template <typename T> inline CUDA_CALLABLE_DEVICE T nvshmem_g(const array_t<T>& source, int i, int j, int pe)
{
    return nvshmem_detail::get_element(&index(source, i, j), pe);
}

template <typename T> inline CUDA_CALLABLE_DEVICE T nvshmem_g(const array_t<T>& source, int i, int j, int k, int pe)
{
    return nvshmem_detail::get_element(&index(source, i, j, k), pe);
}

template <typename T>
inline CUDA_CALLABLE_DEVICE T nvshmem_g(const array_t<T>& source, int i, int j, int k, int l, int pe)
{
    return nvshmem_detail::get_element(&index(source, i, j, k, l), pe);
}

template <typename T>
inline CUDA_CALLABLE_DEVICE void nvshmem_put(const array_t<T>& dest, const array_t<T>& source, int pe)
{
    nvshmem_detail::assert_same_shape(dest, source);

    if (nvshmem_detail::is_packed(dest) && nvshmem_detail::is_packed(source)) {
        ::nvshmem_putmem(dest.data, source.data, nvshmem_detail::element_count(source) * sizeof(T), pe);
    } else if (source.ndim == 1) {
        using P = nvshmem_detail::parts<T>;
        using U = typename P::type;
        for (int c = 0; c < P::count; ++c) {
            P::ops::iput(
                reinterpret_cast<U*>(dest.data) + c, reinterpret_cast<const U*>(source.data) + c,
                dest.strides[0] / P::bytes, source.strides[0] / P::bytes, size_t(source.shape[0]), pe
            );
        }
    } else {
        const size_t count = nvshmem_detail::element_count(source);
        for (size_t n = 0; n < count; ++n) {
            ::nvshmem_putmem(nvshmem_detail::element_at(dest, n), nvshmem_detail::element_at(source, n), sizeof(T), pe);
        }
    }
}

template <typename T>
inline CUDA_CALLABLE_DEVICE void nvshmem_get(const array_t<T>& dest, const array_t<T>& source, int pe)
{
    nvshmem_detail::assert_same_shape(dest, source);

    if (nvshmem_detail::is_packed(dest) && nvshmem_detail::is_packed(source)) {
        ::nvshmem_getmem(dest.data, source.data, nvshmem_detail::element_count(source) * sizeof(T), pe);
    } else if (source.ndim == 1) {
        using P = nvshmem_detail::parts<T>;
        using U = typename P::type;
        for (int c = 0; c < P::count; ++c) {
            P::ops::iget(
                reinterpret_cast<U*>(dest.data) + c, reinterpret_cast<const U*>(source.data) + c,
                dest.strides[0] / P::bytes, source.strides[0] / P::bytes, size_t(source.shape[0]), pe
            );
        }
    } else {
        const size_t count = nvshmem_detail::element_count(source);
        for (size_t n = 0; n < count; ++n) {
            ::nvshmem_getmem(nvshmem_detail::element_at(dest, n), nvshmem_detail::element_at(source, n), sizeof(T), pe);
        }
    }
}

template <typename T, typename S>
inline CUDA_CALLABLE_DEVICE void nvshmem_put_signal(
    const array_t<T>& dest,
    const array_t<T>& source,
    const array_t<uint64>& signals,
    int signal_index,
    S signal,
    int32 signal_op,
    int pe
)
{
    nvshmem_detail::assert_same_shape(dest, source);
    uint64* sig_addr = &index(signals, signal_index);

    if (nvshmem_detail::is_packed(dest) && nvshmem_detail::is_packed(source)) {
        ::nvshmem_putmem_signal(
            dest.data, source.data, nvshmem_detail::element_count(source) * sizeof(T), sig_addr, uint64(signal),
            signal_op, pe
        );
    } else {
        // NVSHMEM has no strided put-with-signal; complete the data transfer before signaling.
        nvshmem_put(dest, source, pe);
        ::nvshmem_quiet();
        ::nvshmemx_signal_op(sig_addr, uint64(signal), signal_op, pe);
    }
}

template <typename S>
inline CUDA_CALLABLE_DEVICE void
nvshmem_signal_op(const array_t<uint64>& signals, int i, S signal, int32 signal_op, int pe)
{
    ::nvshmemx_signal_op(&index(signals, i), uint64(signal), signal_op, pe);
}

template <typename S>
inline CUDA_CALLABLE_DEVICE uint64 nvshmem_signal_wait_until(const array_t<uint64>& signals, int i, int32 cmp, S value)
{
    return ::nvshmem_signal_wait_until(&index(signals, i), cmp, uint64(value));
}

}  // namespace wp

#endif  // defined(__CUDACC__)
