#pragma once

#include <memory>

#include "../utils/geometry.h"

namespace radfoam {

class TriangulationFailedError : public std::runtime_error {
  public:
    TriangulationFailedError(const std::string &message)
        : std::runtime_error(message) {}
};

class Triangulation {
  public:
    virtual ~Triangulation() = default;

    virtual const uint32_t *permutation() const = 0;

    virtual uint32_t num_points() const = 0;

    virtual const IndexedTet *tets() const = 0;

    virtual uint32_t num_tets() const = 0;

    virtual uint32_t num_faces() const = 0;

    virtual const uint32_t *tet_adjacency() const = 0;

    virtual const uint32_t *point_adjacency() const = 0;

    virtual uint32_t point_adjacency_size() const = 0;

    virtual const uint32_t *point_adjacency_offsets() const = 0;

    virtual const uint32_t *vert_to_tet() const = 0;

    virtual bool
    rebuild(const void *points, uint32_t num_points, bool incremental) = 0;

    static std::unique_ptr<Triangulation>
    create_triangulation(const void *points, uint32_t num_points);
};

} // namespace radfoam