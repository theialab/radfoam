#include <vector>

#include "../aabb_tree/aabb_tree.h"

#include "delaunay.cuh"

namespace radfoam {

void check_duplicates(const Vec3f *points, uint32_t num_points) {

    CUDAArray<bool> error_flag_buffer(1);
    auto error_flag = error_flag_buffer.begin();

    cuda_check(cuMemsetD8((CUdeviceptr)error_flag, 0, 1));

    for_n(u32zero(), num_points - 1, [=] __device__(uint32_t i) {
        bool duplicate_point = points[i] == points[i + 1];
        if (duplicate_point) {
            *error_flag = true;
        }
    });

    bool error_flag_host;
    cuda_check(
        cuMemcpyDtoH(&error_flag_host, (CUdeviceptr)error_flag, sizeof(bool)));

    if (error_flag_host) {
        throw TriangulationFailedError("duplicate points found");
    }
}

uint32_t find_adjacency(SortedMap<IndexedTet, uint32_t> &tets_map,
                        SortedMap<IndexedTriangle, uint32_t> &faces_map,
                        uint32_t num_points,
                        CUDAArray<uint32_t> &tet_adjacency,
                        CUDAArray<uint32_t> &point_adjacency,
                        CUDAArray<uint32_t> &point_adjacency_offset,
                        CUDAArray<uint32_t> &face_to_tet,
                        CUDAArray<uint32_t> &vert_to_tet) {

    auto tets = tets_map.unique_keys.begin();
    auto num_tets = tets_map.num_unique_keys;

    auto faces = faces_map.unique_keys.begin();
    auto num_faces = faces_map.num_unique_keys;

    vert_to_tet.resize(num_points);
    auto vert_to_tet_begin = vert_to_tet.begin();

    cuda_check(
        cuMemsetD32((CUdeviceptr)vert_to_tet_begin, UINT32_MAX, num_points));

    auto write_vert_to_tet = [=] __device__(uint32_t i) {
        IndexedTet tet = tets[i];
        for (uint32_t j = 0; j < 4; ++j) {
            atomicMin(vert_to_tet_begin + tet.vertices[j], i);
        }
    };

    for_n(u32zero(), num_tets, write_vert_to_tet);

    CUDAArray<uint32_t> face_adjacent_count(num_faces);
    auto face_adjacent_count_begin = face_adjacent_count.begin();

    cuda_check(
        cuMemsetD32((CUdeviceptr)face_adjacent_count_begin, 0, num_faces));

    CUDAArray<uint32_t> tet_face_index(4 * num_tets);
    auto tet_face_index_begin = tet_face_index.begin();

    face_to_tet.expand(2 * num_faces);
    auto face_to_tet_begin = face_to_tet.begin_as<uint32_t>();

    cuda_check(
        cuMemsetD32((CUdeviceptr)face_to_tet_begin, UINT32_MAX, 2 * num_faces));

    CUDAArray<bool> error_flag_buffer(1);
    auto error_flag = error_flag_buffer.begin();

    cuda_check(cuMemsetD8((CUdeviceptr)error_flag, 0, 1));

    auto write_face_to_tet = [=] __device__(uint32_t i) {
        IndexedTet tet = tets[i];

        for (uint32_t j = 0; j < 4; ++j) {
            IndexedTriangle face = tet.face(j);

            auto it = binary_search(faces, faces + num_faces, face);

            if (it == faces + num_faces) {
                *error_flag = true;
                return;
            }

            size_t k = it - faces;
            uint32_t offset = atomicAdd(face_adjacent_count_begin + k, 1);

            if (offset >= 2) {
                *error_flag = true;
                return;
            }

            face_to_tet_begin[2 * k + offset] = 4 * i + j;
            tet_face_index_begin[4 * i + j] = k;
        }
    };

    for_n(u32zero(), num_tets, write_face_to_tet);

    bool error_flag_host;
    cuda_check(
        cuMemcpyDtoH(&error_flag_host, (CUdeviceptr)error_flag, sizeof(bool)));

    if (error_flag_host) {
        throw TriangulationFailedError("ambiguous triangulation");
    }

    tet_adjacency.expand(num_tets * 4);
    auto tet_adjacency_begin = tet_adjacency.begin();

    auto collect_tet_adjacency = [=] __device__(uint32_t i) {
        uint32_t tet_adjacency[4] = {
            UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};

        for (uint32_t j = 0; j < 4; ++j) {
            uint32_t face_index = tet_face_index_begin[4 * i + j];
            uint32_t face_adjacent_count =
                face_adjacent_count_begin[face_index];

            for (uint32_t k = 0; k < face_adjacent_count; ++k) {
                uint32_t adjacent = face_to_tet_begin[2 * face_index + k];
                if (adjacent != (4 * i + j)) {
                    tet_adjacency[j] = adjacent;
                    break;
                }
            }
        }

        for (uint32_t j = 0; j < 4; ++j) {
            tet_adjacency_begin[4 * i + j] = tet_adjacency[j];
        }
    };

    for_n(u32zero(), num_tets, collect_tet_adjacency);

    CUDAArray<IndexedEdge> edges(6 * num_tets);
    auto edges_begin = edges.begin();

    auto write_edges = [=] __device__(uint32_t i) {
        IndexedTet tet = tets[i];

        edges_begin[6 * i + 0] = IndexedEdge(tet.vertices[0], tet.vertices[1]);
        edges_begin[6 * i + 1] = IndexedEdge(tet.vertices[0], tet.vertices[2]);
        edges_begin[6 * i + 2] = IndexedEdge(tet.vertices[0], tet.vertices[3]);
        edges_begin[6 * i + 3] = IndexedEdge(tet.vertices[1], tet.vertices[2]);
        edges_begin[6 * i + 4] = IndexedEdge(tet.vertices[1], tet.vertices[3]);
        edges_begin[6 * i + 5] = IndexedEdge(tet.vertices[2], tet.vertices[3]);
    };

    for_n(u32zero(), num_tets, write_edges);

    CUDAArray<IndexedEdge> edges_temp(6 * num_tets);

    CUB_CALL(cub::DeviceMergeSort::SortKeysCopy(temp_data,
                                                temp_bytes,
                                                edges_begin,
                                                edges_temp.begin(),
                                                6 * num_tets,
                                                std::less<IndexedEdge>()));

    CUDAArray<uint32_t> num_unique_edges(1);

    CUB_CALL(cub::DeviceSelect::Unique(temp_data,
                                       temp_bytes,
                                       edges_temp.begin(),
                                       edges_begin,
                                       num_unique_edges.begin(),
                                       6 * num_tets));

    uint32_t num_edges;

    cuda_check(cuMemcpyDtoH(
        &num_edges, (CUdeviceptr)num_unique_edges.begin(), sizeof(uint32_t)));

    CUDAArray<uint32_t> point_adjacency_keys(num_edges * 2);
    point_adjacency.resize(num_edges * 2);
    auto point_adjacency_keys_begin = point_adjacency_keys.begin();
    auto point_adjacency_begin = point_adjacency.begin();

    auto write_point_adjacency_keyval = [=] __device__(uint32_t i) {
        IndexedEdge edge = edges_begin[i];
        point_adjacency_keys_begin[2 * i + 0] = edge.vertices[0];
        point_adjacency_keys_begin[2 * i + 1] = edge.vertices[1];
        point_adjacency_begin[2 * i + 0] = edge.vertices[1];
        point_adjacency_begin[2 * i + 1] = edge.vertices[0];
    };

    for_n(u32zero(), num_edges, write_point_adjacency_keyval);

    CUB_CALL(cub::DeviceMergeSort::SortPairs(temp_data,
                                             temp_bytes,
                                             point_adjacency_keys_begin,
                                             point_adjacency_begin,
                                             num_edges * 2,
                                             std::less<uint32_t>()));

    point_adjacency_offset.resize(num_points + 1);
    auto point_adjacency_offset_begin = point_adjacency_offset.begin();

    cuda_check(cuMemsetD32(
        (CUdeviceptr)point_adjacency_offset_begin, UINT32_MAX, num_points + 1));

    auto write_point_adjacency_offset = [=] __device__(uint32_t i) {
        if (i == 0) {
            point_adjacency_offset_begin[0] = 0;
            point_adjacency_offset_begin[num_points] = num_edges * 2;
        } else {
            uint32_t key0 = point_adjacency_keys_begin[i - 1];
            uint32_t key1 = point_adjacency_keys_begin[i];
            if (key0 != key1) {
                point_adjacency_offset_begin[key0 + 1] = i;
            }
        }
    };

    for_n(u32zero(), num_edges * 2, write_point_adjacency_offset);

    return num_edges * 2;
}

class DelaunayTriangulation : public Triangulation {
  public:
    virtual ~DelaunayTriangulation() = default;

    explicit DelaunayTriangulation(const Vec3f *points, uint32_t num_points) {
        rebuild(points, num_points, false);
    }

    const uint32_t *permutation() const override {
        return permutation_buffer.begin();
    }

    uint32_t num_points() const override { return _num_points; }

    const IndexedTet *tets() const override {
        return tets_map.unique_keys.begin();
    }

    uint32_t num_tets() const override { return tets_map.num_unique_keys; }

    uint32_t num_faces() const override { return faces_map.num_unique_keys; }

    const uint32_t *tet_adjacency() const override {
        return tet_adjacency_buffer.begin();
    }

    const uint32_t *point_adjacency() const override {
        return point_adjacency_buffer.begin();
    }

    uint32_t point_adjacency_size() const override {
        return _point_adjacency_size;
    }

    const uint32_t *point_adjacency_offsets() const override {
        return point_adjacency_offset_buffer.begin();
    }

    const uint32_t *vert_to_tet() const override {
        return vert_to_tet_buffer.begin();
    }

    bool rebuild(const void *points_,
                 uint32_t num_points,
                 bool incremental) override {

        if (num_points < 32) {
            throw std::runtime_error(
                "Delaunay triangulation does not support less than 32 points");
        }

        bool sorted;

        const Vec3f *points = static_cast<const Vec3f *>(points_);

        points_buffer.expand(num_points);
        copy_range(points, points + num_points, points_buffer.begin());

        uint32_t num_frontier;

        if (incremental && num_points == _num_points) {
            sorted = false;

            check_duplicates(points_buffer.begin(), num_points);

            aabb_tree_buffer.expand(pow2_round_up(num_points));
            build_aabb_tree(ScalarType::Float32,
                            points_buffer.begin(),
                            num_points,
                            aabb_tree_buffer.begin());

            num_frontier =
                delete_delaunay_violations(points_buffer.begin(),
                                           num_points,
                                           aabb_tree_buffer.begin(),
                                           tets_map,
                                           faces_map,
                                           frontier_buffer,
                                           face_to_tet_buffer.begin());
        } else {
            _num_points = num_points;

            sort_points(points_buffer, num_points, permutation_buffer);

            check_duplicates(points_buffer.begin(), num_points);

            sorted = true;

            aabb_tree_buffer.expand(pow2_round_up(num_points));
            build_aabb_tree(ScalarType::Float32,
                            points_buffer.begin(),
                            num_points,
                            aabb_tree_buffer.begin());

            sample_initial_tets(points_buffer.begin(),
                                num_points,
                                aabb_tree_buffer.begin(),
                                tets_map,
                                faces_map,
                                num_points);

            num_frontier = faces_map.num_unique_keys;
            frontier_buffer.expand(num_frontier);

            cuda_check(cuMemcpyDtoD((CUdeviceptr)frontier_buffer.begin(),
                                    (CUdeviceptr)faces_map.unique_keys.begin(),
                                    num_frontier * sizeof(IndexedTriangle)));
        }

        uint32_t num_iterations = 0;
        while (num_frontier > 0) {
            num_frontier = growth_iteration(points_buffer.begin(),
                                            num_points,
                                            aabb_tree_buffer.begin(),
                                            tets_map,
                                            faces_map,
                                            frontier_buffer,
                                            num_frontier);
            num_iterations++;

            if (num_iterations > 500) {
                throw TriangulationFailedError(
                    "growth iteration limit exceeded");
            }
            if (tets_map.num_unique_keys > num_points * 20) {
                throw TriangulationFailedError("divergent growth iterations");
            }
        }

        _point_adjacency_size = find_adjacency(tets_map,
                                               faces_map,
                                               _num_points,
                                               tet_adjacency_buffer,
                                               point_adjacency_buffer,
                                               point_adjacency_offset_buffer,
                                               face_to_tet_buffer,
                                               vert_to_tet_buffer);

        return sorted;
    }

  private:
    uint32_t _num_points;
    uint32_t _num_convex_hull_points;
    uint32_t _point_adjacency_size;

    CUDAArray<Vec3f> points_buffer;
    CUDAArray<uint32_t> permutation_buffer;
    CUDAArray<AABB<float>> aabb_tree_buffer;
    SortedMap<IndexedTet, uint32_t> tets_map;
    SortedMap<IndexedTriangle, uint32_t> faces_map;
    CUDAArray<IndexedTriangle> frontier_buffer;
    CUDAArray<uint32_t> tet_adjacency_buffer;
    CUDAArray<uint32_t> point_adjacency_buffer;
    CUDAArray<uint32_t> point_adjacency_offset_buffer;
    CUDAArray<uint32_t> face_to_tet_buffer;
    CUDAArray<uint32_t> vert_to_tet_buffer;
};

std::unique_ptr<Triangulation>
Triangulation::create_triangulation(const void *points, uint32_t num_points) {

    return std::make_unique<DelaunayTriangulation>(
        static_cast<const Vec3f *>(points), num_points);
}

} // namespace radfoam