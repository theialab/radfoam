#include <iostream>
#include <vector>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include "../utils/cuda_helpers.h"

#include "../utils/common_kernels.cuh"
#include "aabb_tree.cuh"

#define BUILD_AABB_BLOCK_SIZE 256
#define SORT_BLOCK_SIZE 1024
#define VORONOI_BLOCK_SIZE 256

namespace radfoam {

struct SortValue {
    CVec3<int32_t> point;
    uint32_t perm;
};

template <int block_size>
__global__ void block_sort_kernel(const uint32_t *__restrict__ keys_in,
                                  uint32_t *__restrict__ keys_out,
                                  const SortValue *__restrict__ values_in,
                                  SortValue *__restrict__ values_out,
                                  uint32_t num_points) {
    typedef cub::BlockRadixSort<uint32_t, block_size, 1, SortValue> Sorter;
    __shared__ typename Sorter::TempStorage temp_storage;

    uint32_t block_offset = blockIdx.x * block_size;
    uint32_t block_end = min(block_offset + block_size, num_points);

    uint32_t key[1] = {0xffffffff};
    SortValue perm[1];
    if (threadIdx.x + block_offset < num_points) {
        key[0] = keys_in[threadIdx.x + block_offset];
        perm[0] = values_in[threadIdx.x + block_offset];
    }

    Sorter(temp_storage).Sort(key, perm);

    if (threadIdx.x + block_offset < num_points) {
        keys_out[threadIdx.x + block_offset] = key[0];
        values_out[threadIdx.x + block_offset] = perm[0];
    }
}

void launch_block_sort_kernel(const uint32_t *keys_in,
                              uint32_t *keys_out,
                              const SortValue *values_in,
                              SortValue *values_out,
                              uint32_t num_points) {
    uint32_t num_blocks = (num_points + SORT_BLOCK_SIZE - 1) / SORT_BLOCK_SIZE;
    block_sort_kernel<SORT_BLOCK_SIZE><<<num_blocks, SORT_BLOCK_SIZE>>>(
        keys_in, keys_out, values_in, values_out, num_points);
}

void sort_points(CUDAArray<Vec3f> &points_buffer,
                 uint32_t num_points,
                 CUDAArray<uint32_t> &permutation) {

    CUDAArray<uint32_t> keys_in_buffer(num_points);
    CUDAArray<uint32_t> keys_out_buffer(num_points);
    CUDAArray<SortValue> values_in_buffer(num_points);
    CUDAArray<SortValue> values_out_buffer(num_points);

    uint32_t *keys_in = keys_in_buffer.begin();
    uint32_t *keys_out = keys_out_buffer.begin();
    SortValue *values_in = values_in_buffer.begin();
    SortValue *values_out = values_out_buffer.begin();

    auto points_begin = points_buffer.begin();

    transform_range(u32zero(),
                    u32zero() + num_points,
                    values_in,
                    [=] __device__(uint32_t i) {
                        Vec3f p = points_begin[i];
                        SortValue value;

                        for (int j = 0; j < 3; ++j) {
                            uint32_t u;
                            memcpy(&u, &p[j], sizeof(uint32_t));

                            if (u & 0x80000000) {
                                u = ~u;
                            } else {
                                u ^= 0x80000000;
                            }
                            value.point.data[j] = (int32_t)u;
                        }
                        value.perm = i;
                        return value;
                    });

    uint32_t segment_size = pow2_round_up(num_points);
    int dim = 0;

    while (segment_size > 1) {
        uint32_t num_segments = (num_points + segment_size - 1) / segment_size;

        if (segment_size > SORT_BLOCK_SIZE) {
            if (num_segments > 1024) {
                // CUB segmented sort
                for_n(u32zero(), num_points, [=] __device__(uint32_t i) {
                    keys_in[i] = values_in[i].point.data[dim];
                });

                auto get_segment_offset = [=] __device__(const uint32_t &i) {
                    return min(i * segment_size, num_points);
                };

                auto segment_offset_begin =
                    cub::TransformInputIterator<uint32_t,
                                                decltype(get_segment_offset),
                                                decltype(u32zero())>(
                        u32zero(), get_segment_offset);

                CUB_CALL(cub::DeviceSegmentedRadixSort::SortPairs(
                    temp_data,
                    temp_bytes,
                    keys_in,
                    keys_out,
                    values_in,
                    values_out,
                    num_points,
                    num_segments,
                    segment_offset_begin,
                    segment_offset_begin + 1));
            } else {
                // CUB global sort
                uint32_t segment_bits = log2(pow2_round_up(num_segments));

                for_n(u32zero(), num_points, [=] __device__(uint32_t i) {
                    keys_in[i] = values_in[i].point.data[dim];
                });

                CUB_CALL(cub::DeviceRadixSort::SortPairs(temp_data,
                                                         temp_bytes,
                                                         keys_in,
                                                         keys_out,
                                                         values_in,
                                                         values_out,
                                                         num_points));
            }

        } else {
            // Block sort
            uint32_t segment_bits = log2(SORT_BLOCK_SIZE) - log2(segment_size);

            for_n(u32zero(), num_points, [=] __device__(uint32_t i) {
                keys_in[i] = values_in[i].point.data[dim];
            });

            launch_block_sort_kernel(
                keys_in, keys_out, values_in, values_out, num_points);
        }

        segment_size >>= 1;
        dim = (dim + 1) % 3;

        std::swap(values_in, values_out);
    }

    permutation.expand(num_points);

    uint32_t *permutation_begin = permutation.begin();
    for_n(u32zero(), num_points, [=] __device__(uint32_t i) {
        Vec3f p;

        for (int j = 0; j < 3; ++j) {
            uint32_t u = (uint32_t)values_in[i].point.data[j];

            if (u & 0x80000000) {
                u ^= 0x80000000;
            } else {
                u = ~u;
            }

            memcpy(&p[j], &u, sizeof(float));
        }

        points_begin[i] = p;
        permutation_begin[i] = values_in[i].perm;
    });
}

template <typename scalar>
__global__ void build_leaves_kernel(const Vec3<scalar> *__restrict__ points,
                                    uint32_t num_points,
                                    AABB<scalar> *__restrict__ aabb_tree) {

    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t point_idx = thread_idx * 2;
    uint32_t num_leaves = pow2_round_up(num_points) / 2;

    if (thread_idx >= num_leaves) {
        return;
    }

    Vec3<scalar> leaf0;
    if (point_idx < num_points) {
        leaf0 = points[point_idx];
    } else {
        leaf0 = points[num_points - 1];
    }
    Vec3<scalar> leaf1;
    if (point_idx + 1 < num_points) {
        leaf1 = points[point_idx + 1];
    } else {
        leaf1 = leaf0;
    }
    Vec3<scalar> min_p = leaf0.cwiseMin(leaf1);
    Vec3<scalar> max_p = leaf0.cwiseMax(leaf1);
    AABB<scalar> leaf_aabb(min_p, max_p);

    aabb_tree[thread_idx] = leaf_aabb;

    uint32_t max_level = log2(min(num_leaves, BUILD_AABB_BLOCK_SIZE));
    AABB<scalar> *prev_level_begin = aabb_tree;
    AABB<scalar> *next_level_begin = aabb_tree + num_leaves;

    uint32_t thread_idx_in_block = threadIdx.x;

    for (uint32_t i = 0; i < max_level; ++i) {
        if (thread_idx_in_block >=
            (min(num_leaves, BUILD_AABB_BLOCK_SIZE) >> (i + 1))) {
            return;
        }
        uint32_t prev_level_idx =
            blockIdx.x * (blockDim.x >> i) + thread_idx_in_block * 2;
        uint32_t next_level_idx =
            blockIdx.x * (blockDim.x >> (i + 1)) + thread_idx_in_block;

        __syncthreads();

        AABB<scalar> prev_0 = prev_level_begin[prev_level_idx];
        AABB<scalar> prev_1 = prev_level_begin[prev_level_idx + 1];

        AABB<scalar> next = prev_0.merge(prev_1);

        next_level_begin[next_level_idx] = next;

        uint64_t next_level_width = (next_level_begin - prev_level_begin) >> 1;
        prev_level_begin = next_level_begin;
        next_level_begin += next_level_width;
    }
}

template <typename scalar>
__global__ void build_tree_kernel(AABB<scalar> *__restrict__ start_level,
                                  uint32_t width) {

    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= width / 2) {
        return;
    }

    uint32_t max_level = log2(min(width, BUILD_AABB_BLOCK_SIZE));
    AABB<scalar> *prev_level_begin = start_level;
    AABB<scalar> *next_level_begin = start_level + width;

    uint32_t thread_idx_in_block = threadIdx.x;

    for (uint32_t i = 0; i < max_level; ++i) {
        if (thread_idx_in_block >=
            (min(width / 2, BUILD_AABB_BLOCK_SIZE) >> i)) {
            return;
        }
        uint32_t prev_level_idx =
            blockIdx.x * 2 * (blockDim.x >> i) + thread_idx_in_block * 2;
        uint32_t next_level_idx =
            blockIdx.x * 2 * (blockDim.x >> (i + 1)) + thread_idx_in_block;

        __syncthreads();

        AABB<scalar> prev_0 = prev_level_begin[prev_level_idx];
        AABB<scalar> prev_1 = prev_level_begin[prev_level_idx + 1];

        AABB<scalar> next = prev_0.merge(prev_1);

        next_level_begin[next_level_idx] = next;

        uint64_t next_level_width = (next_level_begin - prev_level_begin) >> 1;
        prev_level_begin = next_level_begin;
        next_level_begin += next_level_width;
    }
}

template <typename scalar>
void build_aabb_tree(const Vec3<scalar> *points,
                     uint32_t num_points,
                     AABB<scalar> *aabb_tree) {

    uint32_t num_leaves = pow2_round_up(num_points) / 2;

    uint32_t num_blocks =
        (num_leaves + BUILD_AABB_BLOCK_SIZE - 1) / BUILD_AABB_BLOCK_SIZE;

    build_leaves_kernel<<<num_blocks, BUILD_AABB_BLOCK_SIZE>>>(
        points, num_points, aabb_tree);

    uint32_t width = num_leaves / BUILD_AABB_BLOCK_SIZE;
    AABB<scalar> *level_start = aabb_tree;
    level_start +=
        2 * (BUILD_AABB_BLOCK_SIZE - 1) * (num_leaves / BUILD_AABB_BLOCK_SIZE);

    while (width > 1) {
        uint32_t num_blocks =
            (width / 2 + BUILD_AABB_BLOCK_SIZE - 1) / BUILD_AABB_BLOCK_SIZE;
        build_tree_kernel<<<num_blocks, BUILD_AABB_BLOCK_SIZE>>>(level_start,
                                                                 width);

        for (uint32_t i = 0; i < log2(BUILD_AABB_BLOCK_SIZE); ++i) {
            level_start += width;
            width /= 2;
        }
    }
}

void build_aabb_tree(ScalarType scalar_type,
                     const void *points,
                     uint32_t num_points,
                     void *aabb_tree) {

    if (scalar_type == ScalarType::Int32) {
        build_aabb_tree(static_cast<const Vec3<int32_t> *>(points),
                        num_points,
                        static_cast<AABB<int32_t> *>(aabb_tree));
    } else if (scalar_type == ScalarType::Float32) {
        build_aabb_tree(static_cast<const Vec3<float> *>(points),
                        num_points,
                        static_cast<AABB<float> *>(aabb_tree));
    } else {
        throw std::runtime_error("unsupported scalar type");
    }
}

template <typename coord_scalar>
__global__ void nn_kernel(const Vec3<coord_scalar> *__restrict__ points,
                          const AABB<coord_scalar> *__restrict__ aabb_tree,
                          const Vec3<coord_scalar> *__restrict__ queries,
                          uint32_t num_points,
                          uint32_t num_queries,
                          uint32_t *__restrict__ indices) {
    uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    uint32_t lane = threadIdx.x % 32;

    if (i >= num_queries) {
        return;
    }

    Vec3<coord_scalar> query_point = queries[i];

    coord_scalar dist;
    uint32_t index;

    warp_knn<VORONOI_BLOCK_SIZE>(
        points, aabb_tree, num_points, query_point, 1, &dist, &index);

    if (lane != 0) {
        return;
    }

    indices[i] = index;
}

template <typename coord_scalar>
void nn(const Vec3<coord_scalar> *points,
        const AABB<coord_scalar> *aabb_tree,
        const Vec3<coord_scalar> *queries,
        uint32_t num_points,
        uint32_t num_queries,
        uint32_t *indices,
        const void *stream) {
    launch_kernel_1d<VORONOI_BLOCK_SIZE>(nn_kernel<coord_scalar>,
                                         32 * num_queries,
                                         stream,
                                         points,
                                         aabb_tree,
                                         queries,
                                         num_points,
                                         num_queries,
                                         indices);
}

void nn(ScalarType coord_scalar_type,
        const void *points,
        const void *aabb_tree,
        const void *queries,
        uint32_t num_points,
        uint32_t num_queries,
        uint32_t *indices,
        const void *stream) {

    if (num_points < 32) {
        throw std::runtime_error("number of points must be at least 32");
    }

    if (coord_scalar_type == ScalarType::Float32) {
        nn(static_cast<const Vec3<float> *>(points),
           static_cast<const AABB<float> *>(aabb_tree),
           static_cast<const Vec3<float> *>(queries),
           num_points,
           num_queries,
           indices,
           stream);
    } else {
        throw std::runtime_error("unsupported scalar type");
    }
}

template <typename coord_scalar>
__host__ __device__ uint32_t nn_cpu(const Vec3<coord_scalar> *points,
                                    const AABB<coord_scalar> *aabb_tree,
                                    const Vec3<coord_scalar> &query,
                                    uint32_t num_points) {
    uint32_t base_width = pow2_round_up(num_points);
    uint32_t tree_depth = log2(base_width);

    coord_scalar min_dist = FLT_MAX;
    uint32_t index = UINT32_MAX;

    auto functor = [&](uint32_t current_depth, uint32_t current_node) {
        AABB<coord_scalar> node =
            get_node(aabb_tree, tree_depth, current_depth, current_node);

        if (current_depth == tree_depth - 1) {
            uint32_t point_start_idx = current_node * 2;

            for (uint32_t i = 0; i < 2; ++i) {
                uint32_t point_idx = point_start_idx + i;
                point_idx = min(point_idx, num_points - 1);

                Vec3f point = points[point_idx];

                float dist = (query - point).norm();

                if (dist < min_dist) {
                    min_dist = dist;
                    index = point_idx;
                }
            }
            return TraversalAction::Continue;
        }

        float dist = node.sdf(query);

        if (dist < min_dist) {
            return TraversalAction::Continue;
        } else {
            return TraversalAction::SkipSubtree;
        }
    };

    traverse(num_points, tree_depth - 1, functor);

    return index;
}

uint32_t nn_cpu(ScalarType coord_scalar_type,
                const void *coords,
                const void *aabb_tree,
                const Vec3f &query,
                uint32_t num_points) {
    if (coord_scalar_type == ScalarType::Float32) {
        return nn_cpu(static_cast<const Vec3<float> *>(coords),
                      static_cast<const AABB<float> *>(aabb_tree),
                      query,
                      num_points);
    } else {
        throw std::runtime_error("unsupported scalar type");
    }
}

} // namespace radfoam