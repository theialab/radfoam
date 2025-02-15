#pragma once

#include <array>
#include <string>

#include <c10/cuda/CUDAStream.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "utils/geometry.h"

namespace radfoam_bindings {

namespace py = pybind11;
using namespace radfoam;

inline void set_default_stream() {
    auto stream = at::cuda::getCurrentCUDAStream();
    at::cuda::setCurrentCUDAStream(stream);
}

inline ScalarType dtype_to_scalar_type(py::object dtype) {
    std::string dtype_str = py::str(dtype).cast<std::string>();

    if (dtype_str == "float32") {
        return ScalarType::Float32;
    } else if (dtype_str == "torch.float32") {
        return ScalarType::Float32;
    } else if (dtype_str == "float64") {
        return ScalarType::Float64;
    } else if (dtype_str == "torch.float64") {
        return ScalarType::Float64;
    } else if (dtype_str == "float16") {
        return ScalarType::Float16;
    } else if (dtype_str == "torch.float16") {
        return ScalarType::Float16;
    } else {
        throw std::runtime_error("unsupported dtype '" + dtype_str + "'");
    }
}

inline ScalarType dtype_to_scalar_type(at::ScalarType dtype) {
    if (dtype == at::kFloat) {
        return ScalarType::Float32;
    } else if (dtype == at::kDouble) {
        return ScalarType::Float64;
    } else if (dtype == at::kHalf) {
        return ScalarType::Float16;
    } else {
        throw std::runtime_error("unsupported dtype '" +
                                 std::string(c10::toString(dtype)) + "'");
    }
}

inline caffe2::TypeMeta scalar_to_type_meta(ScalarType scalar) {
    switch (scalar) {
    case ScalarType::Float32:
        return caffe2::TypeMeta::Make<float>();
    case ScalarType::Float64:
        return caffe2::TypeMeta::Make<double>();
    case ScalarType::Float16:
        return caffe2::TypeMeta::Make<at::Half>();
    case ScalarType::UInt32:
        return caffe2::TypeMeta::Make<uint32_t>();
    default:
        throw std::runtime_error("unsupported scalar type");
    }
}

inline std::array<uint32_t, 3> get_3d_shape(const torch::Tensor &tensor,
                                            int feature_dims) {
    std::array<uint32_t, 3> shape = {1, 1, 1};
    uint32_t product = 1;
    for (int i = 0; i < feature_dims; i++) {
        product *= tensor.size(-(i + 1));
    }
    for (int i = 0; i < 2; i++) {
        if (i + feature_dims + 1 > tensor.dim()) {
            break;
        }
        product *= tensor.size(-(i + feature_dims + 1));
        shape[i] = tensor.size(-(i + feature_dims + 1));
    }
    shape[2] = tensor.numel() / product;
    return shape;
}

} // namespace radfoam_bindings