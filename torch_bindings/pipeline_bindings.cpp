#include "pipeline_bindings.h"

#include "tracing/pipeline.h"
#include "viewer/viewer.h"

namespace radfoam_bindings {

void validate_scene_data(const Pipeline &pipeline,
                         torch::Tensor points,
                         torch::Tensor attributes,
                         torch::Tensor point_adjacency,
                         torch::Tensor point_adjacency_offsets) {

    if (points.size(-1) != 3) {
        throw std::runtime_error("points had dimension " +
                                 std::to_string(points.size(-1)) +
                                 " along axis -1, expected 3");
    }
    if (dtype_to_scalar_type(points.scalar_type()) != ScalarType::Float32) {
        throw std::runtime_error(
            "points had dtype " +
            std::string(c10::toString(points.scalar_type())) + ", expected " +
            std::string(scalar_to_string(ScalarType::Float32)));
    }
    if (points.device().type() != at::kCUDA) {
        throw std::runtime_error("points must be on CUDA device");
    }
    uint32_t num_points = points.numel() / 3;

    if (attributes.size(-1) != pipeline.attribute_dim()) {
        throw std::runtime_error("attributes had dimension " +
                                 std::to_string(attributes.size(-1)) +
                                 " along axis -1, expected " +
                                 std::to_string(pipeline.attribute_dim()));
    }
    if (attributes.numel() / pipeline.attribute_dim() != num_points) {
        throw std::runtime_error("attributes must have the same number of "
                                 "rows as points");
    }
    if (dtype_to_scalar_type(attributes.scalar_type()) !=
        pipeline.attribute_type()) {
        throw std::runtime_error(
            "attributes had dtype " +
            std::string(c10::toString(attributes.scalar_type())) +
            ", expected " +
            std::string(scalar_to_string(pipeline.attribute_type())));
    }
    if (attributes.device().type() != at::kCUDA) {
        throw std::runtime_error("attributes must be on CUDA device");
    }

    if (point_adjacency_offsets.scalar_type() != at::kUInt32) {
        throw std::runtime_error(
            "point_adjacency_offsets must have uint32 dtype");
    }
    if (point_adjacency_offsets.device().type() != at::kCUDA) {
        throw std::runtime_error(
            "point_adjacency_offsets must be on CUDA device");
    }
    if (point_adjacency_offsets.numel() != num_points + 1) {
        throw std::runtime_error("point_adjacency_offsets must have num_points "
                                 "+ 1 elements");
    }

    if (point_adjacency.scalar_type() != at::kUInt32) {
        throw std::runtime_error("point_adjacency must have uint32 dtype");
    }
    if (point_adjacency.device().type() != at::kCUDA) {
        throw std::runtime_error("point_adjacency must be on CUDA device");
    }
}

void update_scene(Viewer &self,
                  torch::Tensor points_in,
                  torch::Tensor attributes_in,
                  torch::Tensor point_adjacency_in,
                  torch::Tensor point_adjacency_offsets_in,
                  torch::Tensor aabb_tree_in) {
    torch::Tensor points = points_in.contiguous();
    torch::Tensor attributes = attributes_in.contiguous();
    torch::Tensor point_adjacency = point_adjacency_in.contiguous();
    torch::Tensor point_adjacency_offsets =
        point_adjacency_offsets_in.contiguous();
    torch::Tensor aabb_tree = aabb_tree_in.contiguous();

    validate_scene_data(self.get_pipeline(),
                        points,
                        attributes,
                        point_adjacency,
                        point_adjacency_offsets);

    set_default_stream();

    uint32_t num_points = points.size(0);
    uint32_t num_attrs = attributes.size(0);
    uint32_t num_point_adjacency = point_adjacency.size(0);
    self.update_scene(num_points,
                      num_attrs,
                      num_point_adjacency,
                      points.data_ptr(),
                      attributes.data_ptr(),
                      point_adjacency.data_ptr(),
                      point_adjacency_offsets.data_ptr(),
                      aabb_tree.data_ptr());
}

py::object trace_forward(Pipeline &self,
                         torch::Tensor points_in,
                         torch::Tensor attributes_in,
                         torch::Tensor point_adjacency_in,
                         torch::Tensor point_adjacency_offsets_in,
                         torch::Tensor rays_in,
                         torch::Tensor start_point_in,
                         std::optional<torch::Tensor> depth_quantiles_in,
                         py::object weight_threshold,
                         py::object max_intersections,
                         bool return_contribution) {
    torch::Tensor points = points_in.contiguous();
    torch::Tensor attributes = attributes_in.contiguous();
    torch::Tensor point_adjacency = point_adjacency_in.contiguous();
    torch::Tensor point_adjacency_offsets =
        point_adjacency_offsets_in.contiguous();
    torch::Tensor rays = rays_in.contiguous();
    torch::Tensor start_point = start_point_in.contiguous();

    validate_scene_data(self,
                        points_in,
                        attributes_in,
                        point_adjacency_in,
                        point_adjacency_offsets_in);

    bool return_depth = depth_quantiles_in.has_value();

    uint32_t num_points = points.size(0);
    uint32_t point_adjacency_size = point_adjacency.size(0);
    uint32_t num_rays = rays.numel() / 6;
    uint32_t num_depth_quantiles = 0;

    if (rays.size(-1) != 6) {
        throw std::runtime_error("rays must have 6 as the last dimension");
    }
    if (rays.scalar_type() != at::kFloat) {
        throw std::runtime_error("rays must have float32 dtype");
    }
    if (rays.device().type() != at::kCUDA) {
        throw std::runtime_error("rays must be on CUDA device");
    }

    if (start_point.numel() != num_rays) {
        throw std::runtime_error("start_point must have the same batch size "
                                 "as rays");
    }
    if (start_point.scalar_type() != at::kUInt32) {
        throw std::runtime_error("start_point must have uint32 dtype");
    }
    if (start_point.device().type() != at::kCUDA) {
        throw std::runtime_error("start_point must be on CUDA device");
    }

    torch::Tensor depth_quantiles;
    if (return_depth) {
        depth_quantiles = depth_quantiles_in.value().contiguous();
        num_depth_quantiles = depth_quantiles.size(-1);

        if (depth_quantiles.scalar_type() != at::kFloat) {
            throw std::runtime_error("depth_quantiles must have float32 dtype");
        }
        if (depth_quantiles.device().type() != at::kCUDA) {
            throw std::runtime_error("depth_quantiles must be on CUDA device");
        }
        if (depth_quantiles.numel() / num_depth_quantiles != num_rays) {
            throw std::runtime_error("depth_quantiles must have the same batch "
                                     "size as rays");
        }
    }

    TraceSettings settings = default_trace_settings();
    if (!weight_threshold.is_none()) {
        settings.weight_threshold = weight_threshold.cast<float>();
    }
    if (!max_intersections.is_none()) {
        settings.max_intersections = max_intersections.cast<uint32_t>();
    }

    std::vector<int64_t> output_shape;
    for (int i = 0; i < rays.dim() - 1; i++) {
        output_shape.push_back(rays.size(i));
    }
    auto output_rgba_shape = output_shape;
    output_rgba_shape.push_back(4);
    torch::Tensor output_rgba =
        torch::empty(output_rgba_shape,
                     torch::dtype(scalar_to_type_meta(self.attribute_type()))
                         .device(rays.device()));

    auto output_num_intersections_shape = output_shape;
    output_num_intersections_shape.push_back(1);
    torch::Tensor num_intersections =
        torch::empty(output_num_intersections_shape,
                     torch::dtype(scalar_to_type_meta(ScalarType::UInt32))
                         .device(rays.device()));

    torch::Tensor output_contribution;
    if (return_contribution) {
        output_contribution = torch::zeros(
            {num_points, 1},
            torch::dtype(scalar_to_type_meta(self.attribute_type()))
                .device(rays.device()));
    }

    auto output_depth_shape = output_shape;
    output_depth_shape.push_back(num_depth_quantiles);
    torch::Tensor output_depth;
    torch::Tensor output_depth_indices;
    if (return_depth) {
        output_depth =
            torch::zeros(output_depth_shape,
                         torch::dtype(scalar_to_type_meta(ScalarType::Float32))
                             .device(rays.device()));
        output_depth_indices =
            torch::zeros(output_depth_shape,
                         torch::dtype(scalar_to_type_meta(ScalarType::UInt32))
                             .device(rays.device()));
    }

    set_default_stream();

    self.trace_forward(
        settings,
        num_points,
        reinterpret_cast<const radfoam::Vec3f *>(points.data_ptr()),
        attributes.data_ptr(),
        point_adjacency_size,
        reinterpret_cast<const uint32_t *>(point_adjacency.data_ptr()),
        reinterpret_cast<const uint32_t *>(point_adjacency_offsets.data_ptr()),
        num_rays,
        reinterpret_cast<const radfoam::Ray *>(rays.data_ptr()),
        reinterpret_cast<const uint32_t *>(start_point.data_ptr()),
        num_depth_quantiles,
        return_depth
            ? reinterpret_cast<const float *>(depth_quantiles.data_ptr())
            : nullptr,
        output_rgba.data_ptr(),
        return_depth ? reinterpret_cast<float *>(output_depth.data_ptr())
                     : nullptr,
        return_depth
            ? reinterpret_cast<uint32_t *>(output_depth_indices.data_ptr())
            : nullptr,
        reinterpret_cast<uint32_t *>(num_intersections.data_ptr()),
        return_contribution ? output_contribution.data_ptr() : nullptr);

    py::dict output_dict;

    output_dict["rgba"] = output_rgba;
    if (return_depth) {
        output_dict["depth"] = output_depth;
        output_dict["depth_indices"] = output_depth_indices;
    }
    if (return_contribution) {
        output_dict["contribution"] = output_contribution;
    }
    output_dict["num_intersections"] = num_intersections;

    return output_dict;
}

py::object trace_backward(Pipeline &self,
                          torch::Tensor points_in,
                          torch::Tensor attributes_in,
                          torch::Tensor point_adjacency_in,
                          torch::Tensor point_adjacency_offsets_in,
                          torch::Tensor rays_in,
                          torch::Tensor start_point_in,
                          torch::Tensor rgb_out,
                          torch::Tensor rgb_grad_in,
                          std::optional<torch::Tensor> depth_quantiles_in,
                          std::optional<torch::Tensor> depth_indices_in,
                          std::optional<torch::Tensor> depth_grad_in,
                          std::optional<torch::Tensor> ray_error_in,
                          py::object weight_threshold,
                          py::object max_intersections) {
    torch::Tensor points = points_in.contiguous();
    torch::Tensor attributes = attributes_in.contiguous();
    torch::Tensor point_adjacency = point_adjacency_in.contiguous();
    torch::Tensor point_adjacency_offsets =
        point_adjacency_offsets_in.contiguous();
    torch::Tensor rays = rays_in.contiguous();
    torch::Tensor start_point = start_point_in.contiguous();

    validate_scene_data(self,
                        points_in,
                        attributes_in,
                        point_adjacency_in,
                        point_adjacency_offsets_in);

    bool return_depth = depth_quantiles_in.has_value();
    bool return_error = ray_error_in.has_value();

    uint32_t num_points = points.size(0);
    uint32_t point_adjacency_size = point_adjacency.size(0);
    uint32_t num_rays = rays.numel() / 6;
    uint32_t num_depth_quantiles = 0;

    if (rays.size(-1) != 6) {
        throw std::runtime_error("rays must have 6 as the last dimension");
    }
    if (rays.scalar_type() != at::kFloat) {
        throw std::runtime_error("rays must have float32 dtype");
    }
    if (rays.device().type() != at::kCUDA) {
        throw std::runtime_error("rays must be on CUDA device");
    }

    if (start_point.numel() != num_rays) {
        throw std::runtime_error("start_point must have the same batch size "
                                 "as rays");
    }
    if (start_point.scalar_type() != at::kUInt32) {
        throw std::runtime_error("start_point must have uint32 dtype");
    }
    if (start_point.device().type() != at::kCUDA) {
        throw std::runtime_error("start_point must be on CUDA device");
    }

    torch::Tensor rgb_grad_in_c = rgb_grad_in.contiguous();
    if (rgb_grad_in_c.size(-1) != 4) {
        throw std::runtime_error("rgb_grad_in must have 4 as "
                                 "the last dimension");
    }
    if (dtype_to_scalar_type(rgb_grad_in_c.scalar_type()) !=
        self.attribute_type()) {
        throw std::runtime_error(
            "rgb_grad_in had dtype " +
            std::string(c10::toString(rgb_grad_in_c.scalar_type())) +
            ", expected " +
            std::string(scalar_to_string(self.attribute_type())));
    }
    if (rgb_grad_in_c.device().type() != at::kCUDA) {
        throw std::runtime_error("rgb_grad_in must be on CUDA device");
    }
    if (rgb_grad_in_c.numel() / 4 != num_rays) {
        throw std::runtime_error("rgb_grad_in must have the same batch size "
                                 "as rays");
    }

    torch::Tensor depth_quantiles;
    torch::Tensor depth_indices;
    torch::Tensor depth_grad;
    if (return_depth) {
        depth_quantiles = depth_quantiles_in.value().contiguous();
        num_depth_quantiles = depth_quantiles.size(-1);

        if (depth_quantiles.scalar_type() != at::kFloat) {
            throw std::runtime_error("depth_quantiles must have float32 dtype");
        }
        if (depth_quantiles.device().type() != at::kCUDA) {
            throw std::runtime_error("depth_quantiles must be on CUDA device");
        }
        if (depth_quantiles.numel() != num_rays * num_depth_quantiles) {
            throw std::runtime_error("depth_quantiles must have the same batch "
                                     "size as rays");
        }

        if (!depth_grad_in.has_value()) {
            throw std::runtime_error("depth_grad must be provided if "
                                     "depth_quantiles is provided");
        }

        depth_indices = depth_indices_in.value().contiguous();

        if (depth_indices.scalar_type() != at::kUInt32) {
            throw std::runtime_error("depth_indices must have uint32 dtype");
        }
        if (depth_indices.device().type() != at::kCUDA) {
            throw std::runtime_error("depth_indices must be on CUDA device");
        }
        if (depth_indices.numel() != num_rays * num_depth_quantiles) {
            throw std::runtime_error("depth_indices must have the same batch "
                                     "size as rays");
        }

        depth_grad = depth_grad_in.value().contiguous();

        if (depth_grad.size(-1) != num_depth_quantiles) {
            throw std::runtime_error("depth_grad must have the same number of "
                                     "depth quantiles as depth_quantiles");
        }
        if (dtype_to_scalar_type(depth_grad.scalar_type()) !=
            ScalarType::Float32) {
            throw std::runtime_error(
                "depth_grad had dtype " +
                std::string(c10::toString(depth_grad.scalar_type())) +
                ", expected " +
                std::string(scalar_to_string(ScalarType::Float32)));
        }
        if (depth_grad.device().type() != at::kCUDA) {
            throw std::runtime_error("depth_grad must be on CUDA device");
        }
        if (depth_grad.numel() != num_rays * num_depth_quantiles) {
            throw std::runtime_error("depth_grad must have the same batch "
                                     "size as rays");
        }
    }

    torch::Tensor ray_error;
    torch::Tensor point_error;
    if (return_error) {
        ray_error = ray_error_in.value().contiguous();

        if (dtype_to_scalar_type(ray_error.scalar_type()) !=
            self.attribute_type()) {
            throw std::runtime_error(
                "ray_error had dtype " +
                std::string(c10::toString(ray_error.scalar_type())) +
                ", expected " +
                std::string(scalar_to_string(self.attribute_type())));
        }
        if (ray_error.device().type() != at::kCUDA) {
            throw std::runtime_error("ray_error must be on CUDA device");
        }
        if (ray_error.numel() != num_rays) {
            std::cout << ray_error.numel() << " " << num_rays << std::endl;
            throw std::runtime_error("ray_error must have the same batch size "
                                     "as rays");
        }

        point_error = torch::zeros(
            {num_points, 1},
            torch::dtype(scalar_to_type_meta(self.attribute_type()))
                .device(rays.device()));
    }

    TraceSettings settings = default_trace_settings();
    if (!weight_threshold.is_none()) {
        settings.weight_threshold = weight_threshold.cast<float>();
    }
    if (!max_intersections.is_none()) {
        settings.max_intersections = max_intersections.cast<uint32_t>();
    }

    int64_t num_attr = attributes.size(0);

    std::vector<int64_t> attr_grad_shape = {num_attr, self.attribute_dim()};

    torch::Tensor attr_grad =
        torch::zeros(attr_grad_shape,
                     torch::dtype(scalar_to_type_meta(self.attribute_type()))
                         .device(rays.device()));

    std::vector<int64_t> points_grad_shape = {(int64_t)num_points, 3};

    torch::Tensor points_grad = torch::zeros(
        points_grad_shape, torch::dtype(rays.dtype()).device(rays.device()));

    torch::Tensor ray_grad = torch::empty_like(rays);

    set_default_stream();

    self.trace_backward(
        settings,
        num_points,
        reinterpret_cast<const radfoam::Vec3f *>(points.data_ptr()),
        attributes.data_ptr(),
        point_adjacency_size,
        reinterpret_cast<const uint32_t *>(point_adjacency.data_ptr()),
        reinterpret_cast<const uint32_t *>(point_adjacency_offsets.data_ptr()),
        num_rays,
        reinterpret_cast<const radfoam::Ray *>(rays.data_ptr()),
        reinterpret_cast<const uint32_t *>(start_point.data_ptr()),
        num_depth_quantiles,
        return_depth
            ? reinterpret_cast<const float *>(depth_quantiles.data_ptr())
            : nullptr,
        return_depth
            ? reinterpret_cast<const uint32_t *>(depth_indices.data_ptr())
            : nullptr,
        rgb_out.data_ptr(),
        rgb_grad_in_c.data_ptr(),
        return_depth ? reinterpret_cast<const float *>(depth_grad.data_ptr())
                     : nullptr,
        return_error ? ray_error.data_ptr() : nullptr,
        reinterpret_cast<radfoam::Ray *>(ray_grad.data_ptr()),
        reinterpret_cast<radfoam::Vec3f *>(points_grad.data_ptr()),
        attr_grad.data_ptr(),
        return_error ? point_error.data_ptr() : nullptr);

    py::dict output_dict;

    output_dict["points_grad"] = points_grad;
    output_dict["attr_grad"] = attr_grad;
    output_dict["ray_grad"] = ray_grad;
    if (return_error) {
        output_dict["point_error"] = point_error;
    }

    return output_dict;
}

void trace_benchmark(Pipeline &self,
                     torch::Tensor points_in,
                     torch::Tensor attributes_in,
                     torch::Tensor point_adjacency_in,
                     torch::Tensor point_adjacency_offsets_in,
                     torch::Tensor adjacent_diff_in,
                     py::dict camera_in,
                     torch::Tensor start_point,
                     torch::Tensor output_rgba_in,
                     py::object weight_threshold,
                     py::object max_intersections) {
    torch::Tensor points = points_in.contiguous();
    torch::Tensor attributes = attributes_in.contiguous();
    torch::Tensor point_adjacency = point_adjacency_in.contiguous();
    torch::Tensor point_adjacency_offsets =
        point_adjacency_offsets_in.contiguous();
    torch::Tensor adjacent_diff = adjacent_diff_in.contiguous();

    validate_scene_data(self,
                        points_in,
                        attributes_in,
                        point_adjacency_in,
                        point_adjacency_offsets_in);

    uint32_t num_points = points.size(0);

    radfoam::Camera camera;
    camera.position = radfoam::Vec3f(
        camera_in["position"].cast<torch::Tensor>().data_ptr<float>());
    camera.forward = radfoam::Vec3f(
        camera_in["forward"].cast<torch::Tensor>().data_ptr<float>());
    camera.up =
        radfoam::Vec3f(camera_in["up"].cast<torch::Tensor>().data_ptr<float>());
    camera.right = radfoam::Vec3f(
        camera_in["right"].cast<torch::Tensor>().data_ptr<float>());
    camera.fov = camera_in["fov"].cast<float>();
    camera.width = camera_in["width"].cast<int>();
    camera.height = camera_in["height"].cast<int>();
    if (camera_in["model"].cast<std::string>() == "pinhole") {
        camera.model = radfoam::CameraModel::Pinhole;
    } else if (camera_in["model"].cast<std::string>() == "fisheye") {
        camera.model = radfoam::CameraModel::Fisheye;
    } else {
        throw std::runtime_error("Invalid camera model");
    }

    if (start_point.numel() != 1) {
        throw std::runtime_error("start_point must have a single element");
    }
    if (start_point.scalar_type() != at::kUInt32) {
        throw std::runtime_error("start_point must have uint32 dtype");
    }
    if (start_point.device().type() != at::kCUDA) {
        throw std::runtime_error("start_point must be on CUDA device");
    }

    if (output_rgba_in.numel() != camera.width * camera.height) {
        throw std::runtime_error("output_rgba must have width * height "
                                 "elements");
    }
    if (output_rgba_in.scalar_type() != at::kUInt32) {
        throw std::runtime_error("output_rgba must have uint32 dtype");
    }
    if (output_rgba_in.device().type() != at::kCUDA) {
        throw std::runtime_error("output_rgba must be on CUDA device");
    }

    TraceSettings settings = default_trace_settings();
    if (!weight_threshold.is_none()) {
        settings.weight_threshold = weight_threshold.cast<float>();
    }
    if (!max_intersections.is_none()) {
        settings.max_intersections = max_intersections.cast<uint32_t>();
    }

    self.trace_benchmark(
        settings,
        num_points,
        reinterpret_cast<const radfoam::Vec3f *>(points.data_ptr()),
        attributes.data_ptr(),
        reinterpret_cast<const uint32_t *>(point_adjacency.data_ptr()),
        reinterpret_cast<const uint32_t *>(point_adjacency_offsets.data_ptr()),
        reinterpret_cast<const radfoam::Vec4h *>(adjacent_diff.data_ptr()),
        camera,
        reinterpret_cast<const uint32_t *>(start_point.data_ptr()),
        reinterpret_cast<uint32_t *>(output_rgba_in.data_ptr()));
}

std::shared_ptr<Pipeline> create_pipeline(int sh_degree,
                                          py::object attr_dtype) {
    return create_pipeline(sh_degree, dtype_to_scalar_type(attr_dtype));
}

void run_with_viewer(std::shared_ptr<Pipeline> pipeline,
                     std::function<void(std::shared_ptr<Viewer>)> callback,
                     std::optional<int> total_iterations,
                     std::optional<torch::Tensor> camera_pos,
                     std::optional<torch::Tensor> camera_forward,
                     std::optional<torch::Tensor> camera_up) {
    py::gil_scoped_release release;

    ViewerOptions options = default_viewer_options();
    if (total_iterations.has_value()) {
        options.total_iterations = total_iterations.value();
    }
    if (camera_pos.has_value()) {
        torch::Tensor camera_pos_cpu =
            camera_pos->contiguous().cpu().to(torch::kFloat);
        options.camera_pos = radfoam::Vec3f(camera_pos_cpu.data_ptr<float>());
    }
    if (camera_forward.has_value()) {
        torch::Tensor camera_forward_cpu =
            camera_forward->contiguous().cpu().to(torch::kFloat);
        options.camera_forward =
            radfoam::Vec3f(camera_forward_cpu.data_ptr<float>());
    }
    if (camera_up.has_value()) {
        torch::Tensor camera_up_cpu =
            camera_up->contiguous().cpu().to(torch::kFloat);
        options.camera_up = radfoam::Vec3f(camera_up_cpu.data_ptr<float>());
    }

    set_default_stream();

    run_with_viewer(std::move(pipeline), std::move(callback), options);
}

void init_pipeline_bindings(py::module &module) {
    py::class_<Pipeline, std::shared_ptr<Pipeline>>(module, "Pipeline")
        .def("trace_forward",
             trace_forward,
             py::arg("points"),
             py::arg("attributes"),
             py::arg("point_adjacency"),
             py::arg("point_adjacency_offsets"),
             py::arg("rays"),
             py::arg("start_point"),
             py::arg("depth_quantiles") = py::none(),
             py::arg("weight_threshold") = py::none(),
             py::arg("max_intersections") = py::none(),
             py::arg("return_contribution") = false)
        .def("trace_backward",
             trace_backward,
             py::arg("points"),
             py::arg("attributes"),
             py::arg("point_adjacency"),
             py::arg("point_adjacency_offsets"),
             py::arg("rays"),
             py::arg("start_point"),
             py::arg("rgb_out"),
             py::arg("grad_in"),
             py::arg("depth_quantiles") = py::none(),
             py::arg("depth_indices") = py::none(),
             py::arg("depth_grad_in") = py::none(),
             py::arg("ray_error") = py::none(),
             py::arg("weight_threshold") = py::none(),
             py::arg("max_intersections") = py::none())
        .def("trace_benchmark",
             trace_benchmark,
             py::arg("points"),
             py::arg("attributes"),
             py::arg("point_adjacency"),
             py::arg("point_adjacency_offsets"),
             py::arg("adjacent_diff"),
             py::arg("camera"),
             py::arg("start_point"),
             py::arg("output_rgba"),
             py::arg("weight_threshold") = py::none(),
             py::arg("max_intersections") = py::none());

    module.def("create_pipeline",
               create_pipeline,
               py::arg("sh_degree"),
               py::arg("attr_dtype") = "float32");

    py::class_<Viewer, std::shared_ptr<Viewer>>(module, "Viewer")
        .def("update_scene",
             update_scene,
             py::arg("points"),
             py::arg("attributes"),
             py::arg("point_adjacency"),
             py::arg("point_adjacency_offsets"),
             py::arg("aabb_tree"))
        .def("step", &Viewer::step)
        .def("is_closed", &Viewer::is_closed);

    module.def("run_with_viewer",
               run_with_viewer,
               py::arg("pipeline"),
               py::arg("callback"),
               py::arg("total_iterations") = py::none(),
               py::arg("camera_pos") = py::none(),
               py::arg("camera_forward") = py::none(),
               py::arg("camera_up") = py::none());
}

} // namespace radfoam_bindings