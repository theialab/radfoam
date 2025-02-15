#include "delaunay.cuh"

namespace radfoam {

constexpr int SAMPLE_INITIAL_TETS_BLOCK_SIZE = 128;

__global__ void sample_initial_tets_kernel(const Vec3f *points,
                                           uint32_t num_points,
                                           const AABB<float> *aabb_tree,
                                           IndexedTet *tets,
                                           IndexedTriangle *faces,
                                           uint32_t num_samples) {

    uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    uint32_t lane = threadIdx.x % 32;

    if (i >= num_samples) {
        return;
    }

    uint32_t steps = 0;

    IndexedTet tet;
    RNGState rng = make_rng(i);

    Vec3f v0, v1, v2, v3;

    bool found = false;
    for (;;) {
        tet.vertices[0] = i % num_points;

        uint32_t nn_idx;

        if (lane == 0) {
            nn_idx = vertex_nearest_neighbour(
                points, aabb_tree, num_points, tet.vertices[0], &steps);
        }

        tet.vertices[1] = __shfl_sync(0xffffffff, nn_idx, 0);

        v0 = points[tet.vertices[0]];
        v1 = points[tet.vertices[1]];

        Vec3f c = (v0 + v1) / 2;
        Vec3f d0 = v1 - v0;
        float r_squared = d0.squaredNorm() / 4;
        Vec3f dr = randn3(rng);
        Vec3f n = d0.cross(dr.normalized());
        Vec3f d1 = n.cross(d0);
        d1.normalize();
        v2 = c + d1 * sqrt(r_squared);

        tet.vertices[2] = maximal_empty_sphere<SAMPLE_INITIAL_TETS_BLOCK_SIZE>(
            points,
            aabb_tree,
            num_points,
            v0,
            v1,
            v2,
            tet.vertices[0],
            tet.vertices[1],
            UINT32_MAX,
            2,
            &steps,
            &found);
        if (!found) {
            break;
        }

        v2 = points[tet.vertices[2]];

        tet.vertices[3] = maximal_empty_sphere<SAMPLE_INITIAL_TETS_BLOCK_SIZE>(
            points,
            aabb_tree,
            num_points,
            v0,
            v1,
            v2,
            tet.vertices[0],
            tet.vertices[1],
            tet.vertices[2],
            3,
            &steps,
            &found);

        if (!found) {
            break;
        }

        v3 = points[tet.vertices[3]];

        if (!found) {
            break;
        }

        found = check_delaunay<SAMPLE_INITIAL_TETS_BLOCK_SIZE>(
            points, aabb_tree, num_points, tet, nullptr);

        break;
    }

    if (lane != 0) {
        return;
    }

    IndexedTriangle _faces[4];

    if (found) {
        Vec3f n = tet.face(0).normal(v0, v1, v2);
        if (n.dot(v0 - v3) < 0) {
            swap(tet.vertices[0], tet.vertices[1]);
        }

        _faces[0] = tet.face(0);
        _faces[1] = tet.face(1);
        _faces[2] = tet.face(2);
        _faces[3] = tet.face(3);
    } else {
        uint32_t max = UINT32_MAX;

        tet.vertices[0] = max;
        tet.vertices[1] = max;
        tet.vertices[2] = max;
        tet.vertices[3] = max;

        _faces[0] = IndexedTriangle(max, max, max);
        _faces[1] = IndexedTriangle(max, max, max);
        _faces[2] = IndexedTriangle(max, max, max);
        _faces[3] = IndexedTriangle(max, max, max);
    }
    tets[i] = tet;
    faces[i * 4 + 0] = _faces[0];
    faces[i * 4 + 1] = _faces[1];
    faces[i * 4 + 2] = _faces[2];
    faces[i * 4 + 3] = _faces[3];
}

void sample_initial_tets(const Vec3f *points,
                         uint32_t num_points,
                         const AABB<float> *aabb_tree,
                         SortedMap<IndexedTet, uint32_t> &tets_map,
                         SortedMap<IndexedTriangle, uint32_t> &faces_map,
                         uint32_t num_samples) {

    CUDAArray<IndexedTet> temp_tets(num_samples);
    IndexedTet *temp_tets_begin = temp_tets.begin();

    CUDAArray<IndexedTriangle> temp_faces(num_samples * 4);
    IndexedTriangle *temp_faces_begin = temp_faces.begin();

    launch_kernel_1d<SAMPLE_INITIAL_TETS_BLOCK_SIZE>(sample_initial_tets_kernel,
                                                     num_samples * 32,
                                                     nullptr,
                                                     points,
                                                     num_points,
                                                     aabb_tree,
                                                     temp_tets_begin,
                                                     temp_faces_begin,
                                                     num_samples);

    tets_map.insert(temp_tets_begin, nullptr, num_samples);
    faces_map.insert(temp_faces_begin, nullptr, num_samples * 4);
}

} // namespace radfoam