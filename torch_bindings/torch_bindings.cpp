
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "pipeline_bindings.h"
#include "triangulation_bindings.h"
#include "utils/batch_fetcher.h"
#include "utils/cuda_array.h"

namespace radfoam {

struct TorchBuffer : public OpaqueBuffer {
    torch::Tensor tensor;

    TorchBuffer(size_t bytes) {
        // allocate on CUDA device
        // int64 dtype for alignment
        size_t num_words = (bytes + sizeof(int64_t) - 1) / sizeof(int64_t);
        tensor = torch::empty({(int64_t)num_words},
                              torch::dtype(torch::kInt64).device(torch::kCUDA));
    }

    void *data() override { return tensor.data_ptr(); }
};

std::unique_ptr<OpaqueBuffer> allocate_buffer(size_t bytes) {
    return std::make_unique<TorchBuffer>(bytes);
}

struct TorchBatchFetcher {
    std::unique_ptr<BatchFetcher> fetcher;
    torch::Tensor data;
    size_t batch_size;

    TorchBatchFetcher(torch::Tensor _data, size_t _batch_size, bool shuffle)
        : data(_data), batch_size(_batch_size) {
        size_t num_bytes = data.numel() * data.element_size();
        size_t num_elems = data.size(0);
        size_t stride = num_bytes / num_elems;
        fetcher = create_batch_fetcher(
            data.data_ptr(), num_bytes, stride, batch_size, shuffle);
    }

    torch::Tensor next() {
        void *batch = fetcher->next();
        std::vector<int64_t> shape;
        shape.push_back(batch_size);
        for (int i = 1; i < data.dim(); i++) {
            shape.push_back(data.size(i));
        }
        return torch::from_blob(batch,
                                shape,
                                torch::dtype(data.dtype()).device(torch::kCUDA))
            .clone();
    }
};

std::unique_ptr<TorchBatchFetcher> create_torch_batch_fetcher(
    torch::Tensor data, size_t batch_size, bool shuffle) {
    return std::make_unique<TorchBatchFetcher>(data, batch_size, shuffle);
}

} // namespace radfoam

namespace radfoam_bindings {

PYBIND11_MODULE(torch_bindings, module) {
    using namespace radfoam_bindings;

    module.doc() = "radfoam pytorch bindings module";

    init_pipeline_bindings(module);
    init_triangulation_bindings(module);

    py::class_<TorchBatchFetcher, std::unique_ptr<TorchBatchFetcher>>(
        module, "BatchFetcher")
        .def(py::init(&create_torch_batch_fetcher),
             py::arg("data"),
             py::arg("batch_size"),
             py::arg("shuffle"))
        .def("next", &TorchBatchFetcher::next);
}

} // namespace radfoam_bindings
