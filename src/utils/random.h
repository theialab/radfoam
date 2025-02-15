#pragma once

#include <cmath>
#include <limits>

#include "typing.h"

namespace radfoam {

// https://github.com/skeeto/hash-prospector
inline RADFOAM_HD uint32_t mix(uint32_t x) {
    x ^= x >> 17;
    x *= 0xed5ad4bb;
    x ^= x >> 11;
    x *= 0xac4c1b51;
    x ^= x >> 15;
    x *= 0x31848bab;
    x ^= x >> 14;
    return x;
}

struct RNGState {
    uint32_t bits;
};

/// @brief Create a new RNG state with the given seed
inline RADFOAM_HD RNGState make_rng(uint32_t seed) {
    return RNGState{mix(seed ^ 0x2815db5b)};
}

#ifdef __CUDACC__
/// @brief Create an RNG with state unique to the current thread
inline __device__ RNGState thread_rng() {
    uint32_t seed =
        threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z) +
        blockDim.x * blockDim.y * blockDim.z *
            (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));
    return make_rng(seed);
}
#endif

/// @brief Generate a random integer in the range [0, 0xffffffff]
inline RADFOAM_HD uint32_t randint(RNGState &rngstate) {
    uint32_t x = rngstate.bits;
    rngstate.bits = mix(rngstate.bits + 1);
    return x;
}

/// @brief Generate a random integer in the range [min, max)
inline RADFOAM_HD uint32_t randint(RNGState &rngstate,
                                   uint32_t min,
                                   uint32_t max) {
    uint32_t diff = max - min;
    uint32_t x = randint(rngstate);
    x /= (0xffffffff / diff);
    return std::min(x, diff - 1) + min;
}

/// @brief Generate a random float in the range [0, 1]
inline RADFOAM_HD float rand(RNGState &rngstate) {
    return float(randint(rngstate)) / float(0xffffffff);
}

/// @brief Generate a random float from a unit normal distribution
inline RADFOAM_HD float randn(RNGState &rngstate) {
    // sample normal distribution using Box - Muller transform
    float u1 = std::max(rand(rngstate), std::numeric_limits<float>::min());
    float u2 = rand(rngstate);
#ifdef __CUDA_ARCH__
    float result = sqrtf(-2 * logf(u1)) * cosf(2 * M_PIf * u2);
#else
    float result = std::sqrt(-2 * std::log(u1)) * std::cos(2 * M_PIf * u2);
#endif
    return float(result);
}

} // namespace radfoam