#include "delaunay.cuh"

namespace radfoam {

constexpr int GROWTH_ITERATION_BLOCK_SIZE = 128;

__global__ void
growth_iteration_kernel(const Vec3f *__restrict__ points,
                        uint32_t num_points,
                        const AABB<float> *__restrict__ aabb_tree,
                        const IndexedTriangle *__restrict__ frontier,
                        uint32_t num_frontier,
                        IndexedTet *__restrict__ new_tets,
                        IndexedTriangle *__restrict__ new_faces) {
    uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    uint32_t lane = threadIdx.x % 32;

    if (i >= num_frontier) {
        return;
    }

    IndexedTriangle seed_face = frontier[i];
    Vec3f v0 = points[seed_face.vertices[0]];
    Vec3f v1 = points[seed_face.vertices[1]];
    Vec3f v2 = points[seed_face.vertices[2]];

    IndexedTet tet;
    tet.vertices[0] = seed_face.vertices[0];
    tet.vertices[1] = seed_face.vertices[1];
    tet.vertices[2] = seed_face.vertices[2];

    uint32_t steps = 0;
    bool found;
    tet.vertices[3] =
        maximal_empty_sphere<GROWTH_ITERATION_BLOCK_SIZE>(points,
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

    if (lane != 0) {
        return;
    }

    Vec3f v3 = points[tet.vertices[3]];

    IndexedTriangle faces[3];
    uint32_t j = 0;
    if (found) {
        Vec3f n = (v1 - v0).cross((v2 - v1));
        if (n.dot((v0 - v3)) < 0) {
            swap(tet.vertices[0], tet.vertices[1]);
        }

        for (uint32_t k = 0; k < 4; ++k) {
            auto face = tet.face(k);
            if (face != seed_face) {
                switch (j) { // Avoid spilling to local memory
                case 0:
                    faces[0] = face;
                    break;
                case 1:
                    faces[1] = face;
                    break;
                case 2:
                    faces[2] = face;
                    break;
                }
                j++;
            }
        }
    } else {
        uint32_t max = UINT32_MAX;

        tet.vertices[0] = max;
        tet.vertices[1] = max;
        tet.vertices[2] = max;
        tet.vertices[3] = max;

        faces[0] = IndexedTriangle(max, max, max);
        faces[1] = IndexedTriangle(max, max, max);
        faces[2] = IndexedTriangle(max, max, max);
    }

    new_tets[i] = tet;
    new_faces[i * 3 + 0] = faces[0];
    new_faces[i * 3 + 1] = faces[1];
    new_faces[i * 3 + 2] = faces[2];
}

uint32_t growth_iteration(const Vec3f *points,
                          uint32_t num_points,
                          const AABB<float> *aabb_tree,
                          SortedMap<IndexedTet, uint32_t> &tets_map,
                          SortedMap<IndexedTriangle, uint32_t> &faces_map,
                          CUDAArray<IndexedTriangle> &frontier,
                          uint32_t num_frontier) {

    frontier.expand(num_frontier * 3, true);
    auto frontier_begin = frontier.begin();

    CUDAArray<IndexedTet> new_tets(num_frontier * 2);
    auto new_tets_begin = new_tets.begin();

    CUDAArray<IndexedTriangle> new_faces(num_frontier * 6);
    auto new_faces_begin = new_faces.begin();

    launch_kernel_1d<GROWTH_ITERATION_BLOCK_SIZE>(growth_iteration_kernel,
                                                  num_frontier * 32,
                                                  nullptr,
                                                  points,
                                                  num_points,
                                                  aabb_tree,
                                                  frontier_begin,
                                                  num_frontier,
                                                  new_tets_begin,
                                                  new_faces_begin);
    SortedMap<IndexedTet, uint32_t> new_tets_map;
    new_tets_map.num_unique_keys = num_frontier;
    new_tets_map.insert(new_tets_begin, nullptr, num_frontier);
    auto unique_new_tets_begin = new_tets_map.unique_keys.begin();
    uint32_t num_new_tets = new_tets_map.num_unique_keys;

    new_tets.clear();

    SortedMap<IndexedTriangle, uint32_t> new_faces_map;
    new_faces_map.num_unique_keys = num_frontier * 3;
    new_faces_map.insert(new_faces_begin, nullptr, num_frontier * 3);
    auto unique_new_faces_begin = new_faces_map.unique_keys.begin();
    uint32_t num_new_faces = new_faces_map.num_unique_keys;

    new_faces.clear();

    CUDAArray<uint32_t> num_selected(2);
    auto num_selected_begin = num_selected.begin();

    uint32_t num_tets = tets_map.num_unique_keys;

    CUDAArray<IndexedTet> tets(num_tets + num_frontier);
    auto tets_begin = tets.begin();
    IndexedTet *tets_end = tets_begin + num_tets;

    copy_range(tets_map.unique_keys.begin(),
               tets_map.unique_keys.begin() + num_tets,
               tets_begin);

    CUDAArray<bool> flags(3 * num_frontier);
    auto flags_begin = flags.begin();

    auto check_tets = [=] __device__(uint32_t i) {
        IndexedTet tet = unique_new_tets_begin[i];
        if (tet.vertices[0] == UINT32_MAX) {
            flags_begin[i] = false;
            return;
        }

        auto it = binary_search(tets_begin, tets_end, tet);

        flags_begin[i] = it == tets_end;
    };

    for_n(u32zero(), num_new_tets, check_tets);

    CUB_CALL(cub::DeviceSelect::Flagged(temp_data,
                                        temp_bytes,
                                        unique_new_tets_begin,
                                        flags_begin,
                                        tets_end,
                                        num_selected_begin,
                                        num_new_tets));

    cuda_check(cuMemcpyDtoH(
        &num_new_tets, (CUdeviceptr)num_selected_begin, sizeof(uint32_t)));
    num_tets += num_new_tets;
    tets_map.insert(tets_begin, nullptr, num_tets);

    tets.clear();

    uint32_t num_faces = faces_map.num_unique_keys;

    CUDAArray<IndexedTriangle> faces(num_faces + 3 * num_frontier);
    auto faces_begin = faces.begin();
    copy_range(faces_map.unique_keys.begin(),
               faces_map.unique_keys.begin() + num_faces,
               faces_begin);
    IndexedTriangle *faces_end = faces_begin + num_faces;

    auto check_faces = [=] __device__(uint32_t i) {
        IndexedTriangle face = unique_new_faces_begin[i];
        if (face.vertices[0] == UINT32_MAX) {
            flags_begin[i] = false;
            return;
        }

        auto it = binary_search(faces_begin, faces_end, face);

        flags_begin[i] = it == faces_end;
    };
    for_n(u32zero(), num_new_faces, check_faces);

    CUB_CALL(cub::DeviceSelect::Flagged(temp_data,
                                        temp_bytes,
                                        unique_new_faces_begin,
                                        flags_begin,
                                        frontier_begin,
                                        num_selected_begin + 1,
                                        num_new_faces));

    cuda_check(cuMemcpyDtoH(&num_new_faces,
                            (CUdeviceptr)(num_selected_begin + 1),
                            sizeof(uint32_t)));

    num_frontier = num_new_faces;
    num_faces += num_frontier;

    copy_range(frontier_begin, frontier_begin + num_frontier, faces_end);

    faces_map.insert(faces_begin, nullptr, num_faces);

    return num_frontier;
}

} // namespace radfoam