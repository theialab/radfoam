#include <memory>

#include "triangulation_bindings.h"

#include "aabb_tree/aabb_tree.h"
#include "delaunay/delaunay.h"
#include "delaunay/triangulation_ops.h"

namespace radfoam_bindings {

std::unique_ptr<Triangulation> create_triangulation(torch::Tensor points) {
    if (points.size(-1) != 3) {
        throw std::runtime_error("points must have 3 as the last dimension");
    }
    if (points.device().type() != at::kCUDA) {
        throw std::runtime_error("points must be on CUDA device");
    }
    if (points.scalar_type() != torch::kFloat32) {
        throw std::runtime_error("points must have float32 dtype");
    }

    uint32_t num_points = points.numel() / 3;

    set_default_stream();

    return Triangulation::create_triangulation(points.data_ptr(), num_points);
}

bool rebuild(Triangulation &triangulation,
             torch::Tensor points,
             bool incremental) {
    if (points.size(-1) != 3) {
        throw std::runtime_error("points must have 3 as the last dimension");
    }
    if (points.device().type() != at::kCUDA) {
        throw std::runtime_error("points must be on CUDA device");
    }
    if (points.scalar_type() != torch::kFloat32) {
        throw std::runtime_error("points must have float32 dtype");
    }

    set_default_stream();

    return triangulation.rebuild(
        points.data_ptr(), points.numel() / 3, incremental);
}

torch::Tensor permutation(const Triangulation &triangulation) {
    const uint32_t *permutation = triangulation.permutation();
    uint32_t num_points = triangulation.num_points();

    at::TensorOptions options =
        at::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA);

    return torch::from_blob(
        const_cast<uint32_t *>(permutation), {num_points}, options);
}

torch::Tensor get_tets(const Triangulation &triangulation) {
    const IndexedTet *tets = triangulation.tets();
    uint32_t num_tets = triangulation.num_tets();

    at::TensorOptions options =
        at::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA);

    return torch::from_blob(
        const_cast<IndexedTet *>(tets), {num_tets, 4}, options);
}

torch::Tensor get_tet_adjacency(const Triangulation &triangulation) {
    const uint32_t *tet_adjacency = triangulation.tet_adjacency();
    uint32_t num_tets = triangulation.num_tets();

    at::TensorOptions options =
        at::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA);

    return torch::from_blob(
        const_cast<uint32_t *>(tet_adjacency), {num_tets, 4}, options);
}

torch::Tensor get_point_adjacency(const Triangulation &triangulation) {
    const uint32_t *point_adjacency = triangulation.point_adjacency();
    uint32_t point_adjacency_size = triangulation.point_adjacency_size();

    at::TensorOptions options =
        at::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA);

    return torch::from_blob(const_cast<uint32_t *>(point_adjacency),
                            {point_adjacency_size},
                            options);
}

torch::Tensor get_point_adjacency_offsets(const Triangulation &triangulation) {
    const uint32_t *point_adjacency_offsets =
        triangulation.point_adjacency_offsets();
    uint32_t num_points = triangulation.num_points();

    at::TensorOptions options =
        at::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA);

    return torch::from_blob(const_cast<uint32_t *>(point_adjacency_offsets),
                            {num_points + 1},
                            options);
}

torch::Tensor get_vert_to_tet(const Triangulation &triangulation) {
    const uint32_t *vert_to_tet = triangulation.vert_to_tet();
    uint32_t num_points = triangulation.num_points();

    at::TensorOptions options =
        at::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA);

    return torch::from_blob(
        const_cast<uint32_t *>(vert_to_tet), {num_points}, options);
}

torch::Tensor build_aabb_tree(torch::Tensor points) {
    if (points.size(-1) != 3) {
        throw std::runtime_error("points must have 3 as the last dimension");
    }
    if (points.dim() != 2) {
        throw std::runtime_error("points must have 2 dimensions");
    }
    if (points.device().type() != at::kCUDA) {
        throw std::runtime_error("points must be on CUDA device");
    }

    ScalarType scalar_type = dtype_to_scalar_type(points.scalar_type());

    uint32_t num_points = points.numel() / 3;

    torch::Tensor aabb_tree = torch::empty(
        {pow2_round_up(num_points), 2, 3},
        torch::TensorOptions().dtype(points.dtype()).device(points.device()));

    radfoam::build_aabb_tree(
        scalar_type, points.data_ptr(), num_points, aabb_tree.data_ptr());

    return aabb_tree;
}

torch::Tensor
nn(torch::Tensor points, torch::Tensor tree, torch::Tensor queries) {
    uint32_t num_points = points.numel() / 3;
    uint32_t num_queries = queries.numel() / 3;

    if (points.scalar_type() != queries.scalar_type()) {
        throw std::runtime_error("points and queries must have the same dtype");
    }
    if (points.scalar_type() != tree.scalar_type()) {
        throw std::runtime_error("points and tree must have the same dtype");
    }
    if (points.device().type() != at::kCUDA) {
        throw std::runtime_error("points must be on CUDA device");
    }
    if (tree.device().type() != at::kCUDA) {
        throw std::runtime_error("tree must be on CUDA device");
    }
    if (queries.device().type() != at::kCUDA) {
        throw std::runtime_error("queries must be on CUDA device");
    }

    std::vector<int64_t> indices_shape;

    for (int64_t i = 0; i < queries.dim() - 1; i++) {
        indices_shape.push_back(queries.size(i));
    }

    torch::Tensor indices = torch::zeros(
        indices_shape, torch::dtype(torch::kUInt32).device(queries.device()));

    radfoam::nn(dtype_to_scalar_type(points.scalar_type()),
                points.data_ptr(),
                tree.data_ptr(),
                queries.data_ptr(),
                num_points,
                num_queries,
                static_cast<uint32_t *>(indices.data_ptr()));

    return indices;
}

std::tuple<torch::Tensor, torch::Tensor>
farthest_neighbor(torch::Tensor points_in,
                  torch::Tensor point_adjacency_in,
                  torch::Tensor point_adjacency_offsets_in) {
    uint32_t num_points = points_in.size(0);
    torch::Tensor points = points_in.contiguous();
    torch::Tensor point_adjacency = point_adjacency_in.contiguous();
    torch::Tensor point_adjacency_offsets =
        point_adjacency_offsets_in.contiguous();

    if (points.device().type() != at::kCUDA) {
        throw std::runtime_error("points must be on CUDA device");
    }

    std::vector<int64_t> indices_shape;

    for (int64_t i = 0; i < points.dim() - 1; i++) {
        indices_shape.push_back(points.size(i));
    }

    torch::Tensor indices = torch::zeros(
        indices_shape, torch::dtype(torch::kUInt32).device(points.device()));
    torch::Tensor cell_radius = torch::zeros(
        indices_shape, torch::dtype(torch::kFloat32).device(points.device()));

    radfoam::farthest_neighbor(dtype_to_scalar_type(points.scalar_type()),
                               points.data_ptr(),
                               point_adjacency.data_ptr(),
                               point_adjacency_offsets.data_ptr(),
                               num_points,
                               static_cast<uint32_t *>(indices.data_ptr()),
                               static_cast<float *>(cell_radius.data_ptr()));

    return std::make_tuple(indices, cell_radius);
}

void init_triangulation_bindings(py::module &module) {
    radfoam::global_cuda_init();

    py::register_exception<radfoam::TriangulationFailedError>(
        module, "TriangulationFailedError");

    py::class_<Triangulation, std::unique_ptr<Triangulation>>(module,
                                                              "Triangulation")
        .def(py::init(&create_triangulation), py::arg("points"))
        .def("tets", &get_tets)
        .def("tet_adjacency", &get_tet_adjacency)
        .def("point_adjacency", &get_point_adjacency)
        .def("point_adjacency_offsets", &get_point_adjacency_offsets)
        .def("vert_to_tet", &get_vert_to_tet)
        .def("rebuild",
             &rebuild,
             py::arg("points"),
             py::arg("incremental") = false)
        .def("permutation", &permutation);

    module.def("build_aabb_tree", &build_aabb_tree, py::arg("points"));

    module.def(
        "nn", &nn, py::arg("points"), py::arg("tree"), py::arg("queries"));

    module.def("farthest_neighbor",
               &farthest_neighbor,
               py::arg("points"),
               py::arg("point_adjacency"),
               py::arg("point_adjacency_offsets"));
}

} // namespace radfoam_bindings