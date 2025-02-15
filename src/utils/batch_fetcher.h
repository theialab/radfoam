#pragma once

#include <memory>

namespace radfoam {

class BatchFetcher {
  public:
    virtual ~BatchFetcher() = default;

    virtual void *next() = 0;
};

std::unique_ptr<BatchFetcher> create_batch_fetcher(const void *data,
                                                   size_t num_bytes,
                                                   size_t stride,
                                                   size_t batch_size,
                                                   bool shuffle);

} // namespace radfoam