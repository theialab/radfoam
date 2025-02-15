#include "delaunay.cuh"

namespace radfoam {

constexpr int DELAUNAY_VIOLATIONS_BLOCK_SIZE = 128;

__global__ void
delaunay_violations_kernel(const Vec3f *__restrict__ points,
                           uint32_t num_points,
                           const AABB<float> *__restrict__ aabb_tree,
                           IndexedTet *__restrict__ tets,
                           uint32_t num_tets,
                           bool *__restrict__ conditions) {
    uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    uint32_t lane = threadIdx.x % 32;

    if (i >= num_tets) {
        return;
    }

    uint32_t steps = 0;

    IndexedTet tet = tets[i];
    swap(tet.vertices[0], tet.vertices[1]);

    bool condition = check_delaunay_warp<DELAUNAY_VIOLATIONS_BLOCK_SIZE>(
        points, aabb_tree, num_points, tet, &steps);

    if (lane == 0) {
        conditions[i] = condition;
    }
}

uint32_t
delete_delaunay_violations(const Vec3f *points,
                           uint32_t num_points,
                           const AABB<float> *aabb_tree,
                           SortedMap<IndexedTet, uint32_t> &tets_map,
                           SortedMap<IndexedTriangle, uint32_t> &faces_map,
                           CUDAArray<IndexedTriangle> &frontier,
                           const uint32_t *face_to_tet) {
    auto tets_begin = tets_map.unique_keys.begin();
    auto num_tets = tets_map.num_unique_keys;

    CUDAArray<bool> conditions(num_tets);
    auto conditions_begin = conditions.begin();

    launch_kernel_1d<DELAUNAY_VIOLATIONS_BLOCK_SIZE>(delaunay_violations_kernel,
                                                     num_tets * 32,
                                                     nullptr,
                                                     points,
                                                     num_points,
                                                     aabb_tree,
                                                     tets_begin,
                                                     num_tets,
                                                     conditions_begin);

    auto faces_begin = faces_map.unique_keys.begin();
    auto num_faces = faces_map.num_unique_keys;

    CUDAArray<uint8_t> face_status(num_faces);
    auto face_status_begin = face_status.begin();

    constexpr uint8_t FACE_UNCHANGED = 0;
    constexpr uint8_t FACE_NOW_FRONTIER = 1;
    constexpr uint8_t FACE_DELETED = 2;

    auto check_face_status = [=] __device__(uint32_t i) {
        uint32_t tet_0 = face_to_tet[2 * i + 0] / 4;
        uint32_t tet_1 = face_to_tet[2 * i + 1] / 4;

        IndexedTriangle face = faces_begin[i];
        uint32_t adjacent_tet;

        uint8_t status;
        if (tet_1 == UINT32_MAX / 4) {
            status = conditions_begin[tet_0] ? FACE_NOW_FRONTIER : FACE_DELETED;
            adjacent_tet = tet_0;
        } else {
            bool cond_0 = conditions_begin[tet_0];
            bool cond_1 = conditions_begin[tet_1];

            if (cond_0 && cond_1) {
                status = FACE_UNCHANGED;
            } else if (cond_0) {
                status = FACE_NOW_FRONTIER;
                adjacent_tet = tet_0;
            } else if (cond_1) {
                status = FACE_NOW_FRONTIER;
                adjacent_tet = tet_1;
            } else {
                status = FACE_DELETED;
            }
        }

        if (status == FACE_NOW_FRONTIER) {
            IndexedTet adjacent = tets_begin[adjacent_tet];

            for (uint32_t j = 0; j < 4; ++j) {
                IndexedTriangle adjacent_face = adjacent.face(j);
                if (face == adjacent_face) {
                    faces_begin[i] = adjacent_face;
                }
            }
        }

        face_status_begin[i] = status;
    };

    for_n(u32zero(), num_faces, check_face_status);

    auto enumerated_faces = enumerate<IndexedTriangle>(faces_begin);

    CUDAArray<IndexedTriangle> unchanged_faces_buffer(num_faces);
    auto unchanged_faces_out =
        unenumerate<IndexedTriangle>(unchanged_faces_buffer.begin());

    auto select_unchanged_and_frontier =
        [=] __device__(cub::KeyValuePair<ptrdiff_t, IndexedTriangle> kv) {
            return face_status_begin[kv.key] == FACE_UNCHANGED ||
                   face_status_begin[kv.key] == FACE_NOW_FRONTIER;
        };

    CUDAArray<uint32_t> num_selected_buffer(2);
    auto num_selected_begin = num_selected_buffer.begin();

    CUB_CALL(cub::DevicePartition::If(temp_data,
                                      temp_bytes,
                                      enumerated_faces,
                                      unchanged_faces_out,
                                      num_selected_begin,
                                      num_faces,
                                      select_unchanged_and_frontier));

    frontier.expand(num_faces);
    auto frontier_faces_out = unenumerate<IndexedTriangle>(frontier.begin());

    auto select_frontier =
        [=] __device__(cub::KeyValuePair<ptrdiff_t, IndexedTriangle> kv) {
            return face_status_begin[kv.key] == FACE_NOW_FRONTIER;
        };

    CUB_CALL(cub::DevicePartition::If(temp_data,
                                      temp_bytes,
                                      enumerated_faces,
                                      frontier_faces_out,
                                      num_selected_begin + 1,
                                      num_faces,
                                      select_frontier));

    face_status.clear();

    uint32_t num_selected_host[2];
    cuda_check(cuMemcpyDtoH(&num_selected_host,
                            (CUdeviceptr)num_selected_begin,
                            2 * sizeof(uint32_t)));

    uint32_t num_frontier = num_selected_host[1];
    faces_map.insert(
        unchanged_faces_buffer.begin(), nullptr, num_selected_host[0]);

    unchanged_faces_buffer.clear();

    CUDAArray<IndexedTet> temp_tets_buffer(num_tets);
    auto temp_tets_begin = temp_tets_buffer.begin();

    CUB_CALL(cub::DevicePartition::Flagged(temp_data,
                                           temp_bytes,
                                           tets_begin,
                                           conditions_begin,
                                           temp_tets_begin,
                                           num_selected_begin,
                                           num_tets));

    cuda_check(cuMemcpyDtoH(
        &num_selected_host, (CUdeviceptr)num_selected_begin, sizeof(uint32_t)));

    tets_map.insert(temp_tets_begin, nullptr, num_selected_host[0]);

    return num_frontier;
}

} // namespace radfoam