#pragma once

#include <cub/iterator/arg_index_input_iterator.cuh>

namespace radfoam {

/// @brief An output iterator that removes the key from a key-value pair
template <typename T>
struct UnenumerateIterator {
    using value_type = void;
    using reference = void;
    using pointer = void;
    using difference_type = ptrdiff_t;
    using iterator_category = std::output_iterator_tag;

    T *ptr;

    __host__ __device__ __forceinline__ UnenumerateIterator() : ptr(nullptr) {}

    __host__ __device__ __forceinline__ UnenumerateIterator(T *ptr)
        : ptr(ptr) {}

    __host__ __device__ __forceinline__ UnenumerateIterator operator++() {
        return UnenumerateIterator(ptr + 1);
    }

    __host__ __device__ __forceinline__ UnenumerateIterator operator++(int) {
        UnenumerateIterator retval = *this;
        ptr++;
        return retval;
    }

    __host__ __device__ __forceinline__ UnenumerateIterator &operator*() {
        return *this;
    }

    template <typename Distance>
    __host__ __device__ __forceinline__ UnenumerateIterator
    operator+(Distance n) const {
        return UnenumerateIterator(ptr + n);
    }

    template <typename Distance>
    __host__ __device__ __forceinline__ UnenumerateIterator &
    operator+=(Distance n) {
        ptr += n;
        return *this;
    }

    template <typename Distance>
    __host__ __device__ __forceinline__ UnenumerateIterator
    operator-(Distance n) const {
        return UnenumerateIterator(ptr - n);
    }

    template <typename Distance>
    __host__ __device__ __forceinline__ UnenumerateIterator &
    operator-=(Distance n) {
        ptr -= n;
        return *this;
    }

    __host__ __device__ __forceinline__ ptrdiff_t
    operator-(UnenumerateIterator other) const {
        return ptr - other.ptr;
    }

    template <typename Distance>
    __host__ __device__ __forceinline__ UnenumerateIterator
    operator[](Distance n) {
        return UnenumerateIterator(ptr + n);
    }

    __host__ __device__ __forceinline__ void
    operator=(const cub::KeyValuePair<ptrdiff_t, T> &x) {
        *ptr = x.value;
    }
};

} // namespace radfoam
