#pragma once

#include <vector>

#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_partition.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include "../utils/cuda_helpers.h"
#include "../utils/geometry.h"
#include "delaunay.h"

#include "../utils/common_kernels.cuh"
#include "exact_tree_ops.cuh"
#include "sorted_map.cuh"

namespace radfoam {

/// @brief Sample num_samples Delaunay tets randomly from the point set
void sample_initial_tets(const Vec3f *points,
                         uint32_t num_points,
                         const AABB<float> *aabb_tree,
                         SortedMap<IndexedTet, uint32_t> &tets_table,
                         SortedMap<IndexedTriangle, uint32_t> &faces_table,
                         uint32_t num_samples);

/// @brief Grow the Delaunay mesh by finding tets adjacent to the frontier
uint32_t growth_iteration(const Vec3f *points,
                          uint32_t num_points,
                          const AABB<float> *aabb_tree,
                          SortedMap<IndexedTet, uint32_t> &tets_table,
                          SortedMap<IndexedTriangle, uint32_t> &faces_table,
                          CUDAArray<IndexedTriangle> &frontier,
                          uint32_t num_frontier);

/// @brief Delete tets that violate the Delaunay condition
uint32_t
delete_delaunay_violations(const Vec3f *points,
                           uint32_t num_points,
                           const AABB<float> *aabb_tree,
                           SortedMap<IndexedTet, uint32_t> &tets_table,
                           SortedMap<IndexedTriangle, uint32_t> &faces_table,
                           CUDAArray<IndexedTriangle> &frontier,
                           const uint32_t *face_to_tet);

} // namespace radfoam