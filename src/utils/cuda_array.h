#pragma once

#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "cuda_helpers.h"
#include "typing.h"

#define CUB_CALL(call)                                                         \
    {                                                                          \
        size_t temp_bytes = 0;                                                 \
        void *temp_data;                                                       \
        CUDAArray<uint8_t> cub_temp_buffer;                                    \
        for (temp_data = nullptr;;) {                                          \
            cuda_check(call);                                                  \
            if (temp_data)                                                     \
                break;                                                         \
            else {                                                             \
                cub_temp_buffer.resize(temp_bytes);                            \
                temp_data = cub_temp_buffer.begin();                           \
            }                                                                  \
        }                                                                      \
    }

namespace radfoam {

struct OpaqueBuffer {
    virtual ~OpaqueBuffer() = default;

    virtual void *data() = 0;
};

std::unique_ptr<OpaqueBuffer> allocate_buffer(size_t bytes);

/// @brief RAII wrapper for CUDA array
template <typename T>
class CUDAArray {
    template <typename U>
    friend class CUDAArray;

  private:
    CUdeviceptr begin_ptr;
    CUdeviceptr end_ptr;
    std::unique_ptr<OpaqueBuffer> buffer;

  public:
    /// @brief Construct an empty CUDA array
    CUDAArray() : begin_ptr(0), end_ptr(0), buffer(nullptr) {}

    /// @brief Construct a CUDA array of a given size
    CUDAArray(size_t size) {
        if (size == 0) {
            begin_ptr = 0;
            end_ptr = 0;
            buffer = nullptr;
            return;
        }
        buffer = allocate_buffer(size * sizeof(T));
        begin_ptr = reinterpret_cast<CUdeviceptr>(buffer->data());
        end_ptr = begin_ptr + (size * sizeof(T));
    }

    CUDAArray(const CUDAArray &other) = delete;

    CUDAArray(CUDAArray &&other) {
        begin_ptr = other.begin_ptr;
        end_ptr = other.end_ptr;
        other.begin_ptr = 0;
        other.end_ptr = 0;
        buffer = std::move(other.buffer);
    }

    ~CUDAArray() = default;

    CUDAArray &operator=(const CUDAArray &other) = delete;

    CUDAArray &operator=(CUDAArray &&other) {
        begin_ptr = other.begin_ptr;
        end_ptr = other.end_ptr;
        other.begin_ptr = 0;
        other.end_ptr = 0;
        buffer = std::move(other.buffer);
        return *this;
    }

    template <typename U>
    CUDAArray(CUDAArray<U> &&other) {
        begin_ptr = other.begin_ptr;
        end_ptr = other.end_ptr;
        other.begin_ptr = 0;
        other.end_ptr = 0;
        buffer = std::move(other.buffer);
    }

    /// @brief Get a pointer to the beginning of the array
    T *begin() { return reinterpret_cast<T *>(begin_ptr); }

    /// @brief Get a pointer to the end of the array
    T *end() {
        size_t size_bytes = end_ptr - begin_ptr;
        size_t elements = size_bytes / sizeof(T);
        return reinterpret_cast<T *>(begin_ptr) + elements;
    }

    /// @brief Get a pointer to the beginning of the array
    const T *begin() const { return reinterpret_cast<const T *>(begin_ptr); }

    /// @brief Get a pointer to the end of the array
    const T *end() const {
        size_t size_bytes = end_ptr - begin_ptr;
        size_t elements = size_bytes / sizeof(T);
        return reinterpret_cast<const T *>(begin_ptr) + elements;
    }

    /// @brief Get a pointer to the beginning of the array as a different type
    template <typename U>
    U *begin_as() {
        return reinterpret_cast<U *>(begin_ptr);
    }

    /// @brief Get a pointer to the end of the array as a different type
    template <typename U>
    U *end_as() {
        size_t size_bytes = end_ptr - begin_ptr;
        size_t elements = size_bytes / sizeof(T);
        return reinterpret_cast<U *>(reinterpret_cast<const T *>(begin_ptr) +
                                     elements);
    }

    /// @brief Get a pointer to the beginning of the array as a different type
    template <typename U>
    const U *begin_as() const {
        return reinterpret_cast<const U *>(begin_ptr);
    }

    /// @brief Get a pointer to the end of the array as a different type
    template <typename U>
    const U *end_as() const {
        size_t size_bytes = end_ptr - begin_ptr;
        size_t elements = size_bytes / sizeof(T);
        return reinterpret_cast<const U *>(
            reinterpret_cast<const T *>(begin_ptr) + elements);
    }

    /// @brief Get the number of elements in the array
    size_t size() const { return (end_ptr - begin_ptr) / sizeof(T); }

    /// @brief Set the number of elements in the array
    /// @param size The new size of the array
    /// @param preserve_content If true, the elements currently in the array
    /// will be copied to the beginning of the new array, up to the minimum of
    /// the old and new sizes
    void resize(size_t size, bool preserve_content = false) {
        if (size == this->size()) {
            return;
        }
        if (begin_ptr) {
            auto new_buffer = allocate_buffer(size * sizeof(T));
            CUdeviceptr new_begin_ptr =
                reinterpret_cast<CUdeviceptr>(new_buffer->data());
            if (preserve_content) {
                if (size > this->size()) {
                    cuda_check(cuMemcpyDtoD(
                        new_begin_ptr, begin_ptr, this->size() * sizeof(T)));
                } else {
                    cuda_check(cuMemcpyDtoD(
                        new_begin_ptr, begin_ptr, size * sizeof(T)));
                }
            }
            buffer = std::move(new_buffer);
            begin_ptr = new_begin_ptr;
            end_ptr = begin_ptr + (size * sizeof(T));
        } else {
            *this = CUDAArray(size);
        }
    }

    /// @brief Expand the array to at least a given size
    /// @param size The minimum size of the array
    /// @param preserve_content If true, the elements currently in the array
    /// will be copied to the beginning of the new array, up to the minimum of
    /// the old and new sizes
    /// @param round_up If true, the size will be rounded up to the nearest
    /// power of 2
    void
    expand(size_t size, bool preserve_content = false, bool round_up = true) {
        if (size > this->size()) {
            if (round_up)
                size = pow2_round_up(size);
            resize(size, preserve_content);
        }
    }

    void clear() {
        begin_ptr = 0;
        end_ptr = 0;
        buffer = nullptr;
    }
};

} // namespace radfoam
