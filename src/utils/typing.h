#pragma once

#include <string>

#include <cuda_fp16.h>

#ifdef __CUDACC__
#define RADFOAM_HD __host__ __device__
#else
#define RADFOAM_HD
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

namespace radfoam {

enum ScalarType {
    Float16,
    Float32,
    Float64,
    UInt32,
    Int32,
    Int64,
};

inline std::string scalar_to_string(ScalarType type) {
    switch (type) {
    case Float16:
        return "float16";
    case Float32:
        return "float32";
    case Float64:
        return "float64";
    case UInt32:
        return "uint32";
    case Int32:
        return "int32";
    case Int64:
        return "int64";
    default:
        return "unknown";
    }
}

inline size_t scalar_size(ScalarType type) {
    switch (type) {
    case Float16:
        return 2;
    case Float32:
        return 4;
    case Float64:
        return 8;
    case UInt32:
        return 4;
    case Int32:
        return 4;
    case Int64:
        return 8;
    default:
        return 0;
    }
}

template <typename T>
constexpr ScalarType scalar_code() = delete;

template <>
constexpr ScalarType scalar_code<__half>() {
    return Float16;
}

template <>
constexpr ScalarType scalar_code<float>() {
    return Float32;
}

template <>
constexpr ScalarType scalar_code<double>() {
    return Float64;
}

template <>
constexpr ScalarType scalar_code<uint32_t>() {
    return UInt32;
}

template <>
constexpr ScalarType scalar_code<int32_t>() {
    return Int32;
}

template <>
constexpr ScalarType scalar_code<int64_t>() {
    return Int64;
}

template <typename T>
constexpr const char *scalar_cxx_name() = delete;

template <>
constexpr const char *scalar_cxx_name<uint32_t>() {
    return "uint32_t";
}

template <>
constexpr const char *scalar_cxx_name<__half>() {
    return "__half";
}

template <>
constexpr const char *scalar_cxx_name<float>() {
    return "float";
}

template <>
constexpr const char *scalar_cxx_name<double>() {
    return "double";
}

template <>
constexpr const char *scalar_cxx_name<int32_t>() {
    return "int32_t";
}

template <>
constexpr const char *scalar_cxx_name<int64_t>() {
    return "int64_t";
}

enum ColorMap {
    Gray = 0,
    Viridis = 1,
    Inferno = 2,
    Turbo = 3,
};

struct CMapTable {
    const float *const *data;
    const int *sizes;
};

template <typename T>
RADFOAM_HD void swap(T &a, T &b) {
    typename std::decay<T>::type tmp = a;
    a = b;
    b = tmp;
}

/// @brief Compute the base-2 logarithm of an integer
inline RADFOAM_HD uint32_t log2(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return (x > 0) ? 31 - __clz(x) : 0;
#else
    uint32_t result = 0;
    while (x >>= 1) {
        result++;
    }
    return result;
#endif
}

/// @brief Compute the smallest power of 2 greater than or equal to x
inline RADFOAM_HD uint32_t pow2_round_up(uint32_t x) {
    return (x > 1) ? 1 << (log2(x - 1) + 1) : 1;
}

} // namespace radfoam