#pragma once

#include <cub/warp/warp_merge_sort.cuh>

#include "../utils/cuda_array.h"
#include "../utils/geometry.h"
#include "../utils/random.h"
#include "aabb_tree.h"

namespace radfoam {

/// @brief Sort the point set into an order appropriate for the leaves of an
/// AABB tree
void sort_points(CUDAArray<Vec3f> &points_buffer,
                 uint32_t num_points,
                 CUDAArray<uint32_t> &permutation);

/// @brief Query a node from the specified level of the AABB tree
template <typename scalar>
__forceinline__ __host__ __device__ AABB<scalar>
get_node(const AABB<scalar> *aabb_tree,
         uint32_t tree_depth,
         uint32_t node_depth,
         uint32_t node_idx) {
    auto *level_start = aabb_tree + ((1 << tree_depth) - (1 << node_depth + 1));
    return *(level_start + node_idx);
}

enum TraversalAction {
    Continue,
    SkipSubtree,
    Terminate,
};

/// @brief Traverse the AABB tree in a depth-first manner
template <typename Functor>
__forceinline__ __host__ __device__ void
traverse(uint32_t num_points, uint32_t max_depth, Functor functor) {

    uint32_t current_depth = 0;
    uint32_t current_node = 0;
    uint32_t tree_depth = log2(pow2_round_up(num_points));

    for (;;) {
        auto action = functor(current_depth, current_node);

        if (action == TraversalAction::Terminate) {
            break;
        } else if (action == TraversalAction::Continue &&
                   current_depth != max_depth) {
            current_node = 2 * current_node;
            current_depth++;
            continue;
        }

        current_node++;
#ifdef __CUDA_ARCH__
        uint32_t step_up_amount =
            min(__ffs(current_node) - 1, (int)current_depth);
#else
        uint32_t step_up_amount =
            min(__builtin_ctz(current_node), (int)current_depth);
#endif
        current_depth -= step_up_amount;
        current_node = current_node >> step_up_amount;

        uint32_t div = tree_depth - current_depth;
        uint32_t current_width = (num_points + (1 << div) - 1) >> div;

        if (current_node >= current_width)
            break;
    }
}

/// @brief Traverse the AABB tree cooperatively within a warp
template <int block_size, typename NodeFunctor, typename LeafFunctor>
__forceinline__ __device__ void warp_traverse(uint32_t num_points,
                                              NodeFunctor node_functor,
                                              LeafFunctor leaf_functor) {

    uint32_t base_width = pow2_round_up(num_points);
    uint32_t tree_depth = log2(base_width);

    uint32_t warp_idx = threadIdx.x / 32;
    uint8_t idx_in_warp = threadIdx.x % 32;
    uint32_t current_depth = tree_depth % 5;
    uint32_t current_node = 0;

    __shared__ uint32_t mask_stack_array[5 * (block_size / 32)];
    __shared__ uint8_t offset_stack_array[5 * (block_size / 32)];
    uint32_t *mask_stack = mask_stack_array + 5 * warp_idx;
    uint8_t *offset_stack = offset_stack_array + 5 * warp_idx;
    int8_t stack_idx = -1;

    uint8_t current_offset = 0;
    uint32_t current_mask = 0xffffffff;

    for (;;) {
        uint32_t div = tree_depth - current_depth;
        uint32_t current_width = (num_points + (1 << div) - 1) >> div;

        if (current_node >= current_width) {
            break;
        }

        bool maskbit = (current_mask >> current_offset) & 1;

        uint32_t thread_node = current_node + idx_in_warp;

        if (maskbit && current_depth == tree_depth) {
            leaf_functor(thread_node);
        } else if (maskbit) {
            TraversalAction action = node_functor(current_depth, thread_node);

            if (__any_sync(0xffffffff, action == TraversalAction::Terminate)) {
                break;
            }
            __syncwarp(0xffffffff);
            if (idx_in_warp == 0 && stack_idx >= 0) {
                mask_stack[stack_idx] = current_mask;
                offset_stack[stack_idx] = current_offset;
            }
            __syncwarp(0xffffffff);
            stack_idx++;
            current_offset = 0;
            current_mask =
                __ballot_sync(0xffffffff, action == TraversalAction::Continue);

            current_node = 32 * current_node;
            current_depth += 5;
            continue;
        }

        current_node += 32;
        current_offset++;

        while (current_offset == 32) {
            if (stack_idx < 0) {
                break;
            }

            stack_idx--;
            current_offset = offset_stack[stack_idx];
            current_mask = mask_stack[stack_idx];

            current_offset++;
            current_node = current_node >> 5;
            current_depth -= 5;
        }
    }
}

/// @brief Perform a warp-cooperative k-nearest neighbor search
template <int block_size, typename scalar>
__forceinline__ __device__ void warp_knn(const Vec3<scalar> *points,
                                         const AABB<scalar> *aabb_tree,
                                         uint32_t num_points,
                                         const Vec3<scalar> &query,
                                         uint32_t k,
                                         scalar *distance_out,
                                         uint32_t *index_out) {
    uint32_t base_width = pow2_round_up(num_points);
    uint32_t tree_depth = log2(base_width);

    uint32_t warp_idx = threadIdx.x / 32;
    uint8_t idx_in_warp = threadIdx.x % 32;

    using Sort = cub::WarpMergeSort<scalar, 2, 32, uint32_t>;
    __shared__ typename Sort::TempStorage temp_storage[block_size / 32];

    auto compare = [](auto a, auto b) { return a < b; };

    scalar distance = std::numeric_limits<scalar>::max();
    uint32_t index = 0;
    scalar max_distance = std::numeric_limits<scalar>::max();

    scalar local_keys[2];
    uint32_t local_values[2];

    uint32_t current_depth = 6;
    uint32_t current_node = 0;

    uint32_t steps = 0;

    auto leaf_functor = [&](uint32_t point_idx) {
        scalar new_dist = std::numeric_limits<scalar>::max();

        if (point_idx < num_points) {
            Vec3<scalar> point = points[point_idx];
            new_dist = (point - query).norm();
        }

        local_keys[0] = distance;
        local_keys[1] = new_dist;
        local_values[0] = index;
        local_values[1] = point_idx;

        Sort(temp_storage[warp_idx]).Sort(local_keys, local_values, compare);

        scalar d = __shfl_sync(0xffffffff, local_keys[0], idx_in_warp / 2);
        uint32_t i = __shfl_sync(0xffffffff, local_values[0], idx_in_warp / 2);
        if (idx_in_warp % 2 == 0) {
            distance = d;
            index = i;
        }
        d = __shfl_sync(0xffffffff, local_keys[1], idx_in_warp / 2);
        i = __shfl_sync(0xffffffff, local_values[1], idx_in_warp / 2);
        if (idx_in_warp % 2 == 1) {
            distance = d;
            index = i;
        }

        scalar new_max_distance = __shfl_sync(0xffffffff, distance, k - 1);
        if (new_max_distance < max_distance) {
            max_distance = new_max_distance;
        }

        steps++;
    };

    while (current_depth < tree_depth) {
        AABB<scalar> node0 = get_node(
            aabb_tree, tree_depth, current_depth, current_node + idx_in_warp);
        AABB<scalar> node1 = get_node(aabb_tree,
                                      tree_depth,
                                      current_depth,
                                      current_node + idx_in_warp + 32);

        scalar dist0 = node0.sdf(query);
        scalar dist1 = node1.sdf(query);

        local_keys[0] = dist0;
        local_keys[1] = dist1;
        local_values[0] = current_node + idx_in_warp;
        local_values[1] = current_node + idx_in_warp + 32;

        Sort(temp_storage[warp_idx]).Sort(local_keys, local_values, compare);

        uint32_t best_node = __shfl_sync(0xffffffff, local_values[0], 0);

        uint32_t levels_to_step = min(tree_depth - current_depth, 6);

        current_node = best_node << levels_to_step;
        current_depth += levels_to_step;
    }

    current_node = max(current_node - 16, 0u);
    leaf_functor(current_node + idx_in_warp);

    distance = std::numeric_limits<scalar>::max();
    index = 0;

    auto node_functor = [&](uint32_t current_depth, uint32_t current_node) {
        if (current_node >= (1 << current_depth))
            return TraversalAction::SkipSubtree;

        AABB<scalar> node =
            get_node(aabb_tree, tree_depth, current_depth, current_node);

        scalar dist = (node.min - query)
                          .cwiseMax(Vec3<scalar>::Zero())
                          .cwiseMax(query - node.max)
                          .norm();

        if (dist > max_distance) {
            return TraversalAction::SkipSubtree;
        }

        return TraversalAction::Continue;
    };

    warp_traverse<block_size>(num_points, node_functor, leaf_functor);

    *distance_out = distance;
    *index_out = index;
}

} // namespace radfoam