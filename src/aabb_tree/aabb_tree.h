#pragma once

#include "../utils/cuda_helpers.h"
#include "../utils/geometry.h"

namespace radfoam {

/// @brief Build an AABB tree from a set of points, assuming that the points are
/// already sorted
void build_aabb_tree(ScalarType scalar_type,
                     const void *points,
                     uint32_t num_points,
                     void *aabb_tree);

/// @brief Find the nearest neighbor of each query point
void nn(ScalarType coord_scalar_type,
        const void *coords,
        const void *aabb_tree,
        const void *query_points,
        uint32_t num_points,
        uint32_t num_queries,
        uint32_t *indices,
        const void *stream = nullptr);

/// @brief Find the nearest neighbor of a single query point
uint32_t nn_cpu(ScalarType coord_scalar_type,
                const void *coords,
                const void *aabb_tree,
                const Vec3f &query,
                uint32_t num_points);

} // namespace radfoam