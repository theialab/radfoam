#pragma once

#include "../utils/geometry.h"

namespace radfoam {

/// @brief Find the farthest neighbor of each point
void farthest_neighbor(ScalarType coord_scalar_type,
                       const void *points,
                       const void *point_adjacency,
                       const void *point_adjacency_offsets,
                       uint32_t num_points,
                       void *indices,
                       void *cell_radius,
                       const void *stream = nullptr);

} // namespace radfoam