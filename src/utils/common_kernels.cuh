#pragma once

#include <iostream>

#include "cuda_helpers.h"

#include <cub/iterator/arg_index_input_iterator.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/discard_output_iterator.cuh>

#include "unenumerate_iterator.cuh"

namespace radfoam {

template <typename InputIterator, typename UnaryFunction>
__global__ void for_n_kernel(InputIterator begin, size_t n, UnaryFunction f) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    while (i < n) {
        f(begin[i]);
        i += stride;
    }
}

template <int block_size, typename Kernel, typename... Args>
void launch_kernel_1d(Kernel kernel,
                      size_t n,
                      const void *stream,
                      Args... args) {
    if (n == 0) {
        return;
    }
    size_t num_blocks = (n + block_size - 1) / block_size;
    if (stream) {
        cudaStream_t s = *reinterpret_cast<const cudaStream_t *>(stream);
        kernel<<<num_blocks, block_size, 0, s>>>(args...);
    } else {
        kernel<<<num_blocks, block_size>>>(args...);
    }
    cuda_check(cudaGetLastError());
}

template <int block_size, typename InputIterator, typename UnaryFunction>
void for_n_b(InputIterator begin,
             size_t n,
             UnaryFunction f,
             bool strided = false,
             const void *stream = nullptr) {
    size_t num_threads = n;
    if (strided) {
        int mpc;
        cuda_check(
            cudaDeviceGetAttribute(&mpc, cudaDevAttrMultiProcessorCount, 0));
        num_threads = block_size * mpc;
    }

    launch_kernel_1d<block_size>(for_n_kernel<InputIterator, UnaryFunction>,
                                 num_threads,
                                 stream,
                                 begin,
                                 n,
                                 f);
}

template <typename InputIterator, typename UnaryFunction>
void for_n(InputIterator begin,
           size_t n,
           UnaryFunction f,
           bool strided = false,
           const void *stream = nullptr) {
    for_n_b<256>(begin, n, f, strided, stream);
}

template <typename InputIterator, typename UnaryFunction>
void for_range(InputIterator begin,
               InputIterator end,
               UnaryFunction f,
               bool strided = false,
               const void *stream = nullptr) {
    size_t n = end - begin;
    for_n(begin, n, f, strided, stream);
}

template <typename InputIterator,
          typename OutputIterator,
          typename UnaryFunction>
struct TransformFunctor {
    InputIterator begin;
    OutputIterator result;
    UnaryFunction f;

    __device__ void operator()(decltype(*begin) x) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        result[i] = f(x);
    }
};

template <typename InputIterator,
          typename OutputIterator,
          typename UnaryFunction>
void transform_range(InputIterator begin,
                     InputIterator end,
                     OutputIterator result,
                     UnaryFunction f,
                     bool strided = false,
                     const void *stream = nullptr) {
    size_t n = end - begin;
    TransformFunctor<InputIterator, OutputIterator, UnaryFunction> func = {
        begin, result, f};
    for_n(begin, n, func, strided, stream);
}

template <typename InputIterator, typename OutputIterator>
void copy_range(InputIterator begin,
                InputIterator end,
                OutputIterator result,
                bool strided = false,
                const void *stream = nullptr) {
    transform_range(
        begin,
        end,
        result,
        [] __device__(auto x) { return x; },
        strided,
        stream);
}

inline cub::CountingInputIterator<uint32_t> u32zero() {
    return cub::CountingInputIterator<uint32_t>(0);
}

inline cub::CountingInputIterator<uint64_t> u64zero() {
    return cub::CountingInputIterator<uint64_t>(0);
}

template <typename T>
inline cub::ArgIndexInputIterator<T *> enumerate(T *begin) {
    return cub::ArgIndexInputIterator<T *>(begin);
}

template <typename T>
inline UnenumerateIterator<T> unenumerate(T *begin) {
    return UnenumerateIterator<T>(begin);
}

inline cub::DiscardOutputIterator<> discard() {
    return cub::DiscardOutputIterator();
}

} // namespace radfoam