#include "triangulation_ops.h"

#include "../utils/common_kernels.cuh"
#include "exact_tree_ops.cuh"

namespace radfoam {

template <typename coord_scalar>
__global__ void
farthest_neighbor_kernel(const Vec3<coord_scalar> *__restrict__ points,
                         const uint32_t *point_adjacency,
                         const uint32_t *point_adjacency_offsets,
                         uint32_t num_points,
                         uint32_t *__restrict__ indices,
                         float *__restrict__ cell_radius) {
    uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= num_points) {
        return;
    }

    Vec3f primal_point = points[i];
    uint32_t point_adjacency_begin = point_adjacency_offsets[i];
    uint32_t point_adjacency_end = point_adjacency_offsets[i + 1];
    uint32_t num_faces = point_adjacency_end - point_adjacency_begin;
    uint32_t farthest_idx = UINT32_MAX;
    float sum_distance = 0.0f;
    float max_distance = 0.0f;

    for (uint32_t i = 0; i < num_faces; ++i) {
        uint32_t opposite_point_idx =
            point_adjacency[point_adjacency_begin + i];
        Vec3f opposite_point = points[opposite_point_idx];

        float distance = (opposite_point - primal_point).norm();
        sum_distance += 0.5 * distance;
        if (distance > max_distance) {
            max_distance = distance;
            farthest_idx = opposite_point_idx;
        }
    }

    indices[i] = farthest_idx;
    cell_radius[i] = sum_distance / num_faces;
}

template <typename coord_scalar>
void farthest_neighbor(const Vec3<coord_scalar> *points,
                       const uint32_t *point_adjacency,
                       const uint32_t *point_adjacency_offsets,
                       uint32_t num_points,
                       uint32_t *indices,
                       float *cell_radius,
                       const void *stream) {
    launch_kernel_1d<1024>(farthest_neighbor_kernel<coord_scalar>,
                           num_points,
                           stream,
                           points,
                           point_adjacency,
                           point_adjacency_offsets,
                           num_points,
                           indices,
                           cell_radius);
}

void farthest_neighbor(ScalarType coord_scalar_type,
                       const void *points,
                       const void *point_adjacency,
                       const void *point_adjacency_offsets,
                       uint32_t num_points,
                       void *indices,
                       void *cell_radius,
                       const void *stream) {

    if (coord_scalar_type == ScalarType::Float32) {
        farthest_neighbor(
            static_cast<const Vec3<float> *>(points),
            static_cast<const uint32_t *>(point_adjacency),
            static_cast<const uint32_t *>(point_adjacency_offsets),
            num_points,
            static_cast<uint32_t *>(indices),
            static_cast<float *>(cell_radius),
            stream);
    } else {
        throw std::runtime_error("unsupported scalar type");
    }
}

} // namespace radfoam