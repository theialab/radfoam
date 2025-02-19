#include <atomic>
#include <cstring>
#include <exception>
#include <iostream>
#include <thread>
#include <vector>

#include <atomic_queue/atomic_queue.h>

#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_TBB
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "batch_fetcher.h"
#include "cuda_helpers.h"
#include "random.h"

namespace radfoam {

constexpr int buffer_size = 4;

struct Batch {
    CUdeviceptr data;
    CUevent data_ready_event;
};

class BatchFetcherImpl : public BatchFetcher {
  public:
    BatchFetcherImpl(const uint8_t *data,
                     size_t num_bytes,
                     size_t stride,
                     size_t batch_size,
                     bool shuffle)
        : worker_exception(nullptr), done(false) {

        CUcontext context;
        cuda_check(cuCtxGetCurrent(&context));

        if (context == nullptr) {
            throw std::runtime_error("No CUDA context found");
        }

        worker = std::thread([=] {
            try {
                size_t num_elemnts = num_bytes / stride;
                if (num_elemnts > (size_t)__UINT32_MAX__) {
                    throw std::runtime_error("Too many elements");
                }
                uint32_t batch_idx = 0;

                cuda_check(cuCtxSetCurrent(context));

                CUstream stream;

                std::vector<uint8_t> cpu_batch_buffer[buffer_size];
                CUdeviceptr gpu_batch_buffer[buffer_size];
                CUevent events[buffer_size];

                cuda_check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

                auto upload_batch = [&](int i) {
                    auto copy_element = [&](int j) {
                        size_t idx;
                        if (shuffle) {
                            auto rng = make_rng(batch_idx * batch_size + j);
                            idx = randint(rng, 0, num_elemnts);
                        } else {
                            idx = (batch_idx * batch_size + j) % num_elemnts;
                        }
                        memcpy(cpu_batch_buffer[i].data() + j * stride,
                               data + idx * stride,
                               stride);
                    };
                    thrust::for_each(thrust::host,
                                     thrust::counting_iterator<int>(0),
                                     thrust::counting_iterator<int>(batch_size),
                                     copy_element);
                    batch_idx += 1;
                    cuda_check(cuMemcpyHtoDAsync(gpu_batch_buffer[i],
                                                 cpu_batch_buffer[i].data(),
                                                 batch_size * stride,
                                                 stream));
                    cuda_check(cuEventRecord(events[i], stream));

                    return Batch{gpu_batch_buffer[i], events[i]};
                };

                for (int i = 0; i < buffer_size; ++i) {
                    cpu_batch_buffer[i].resize(batch_size * stride);
                    cuda_check(
                        cuMemAlloc(&gpu_batch_buffer[i], batch_size * stride));
                    cuda_check(
                        cuEventCreate(&events[i], CU_EVENT_BLOCKING_SYNC));
                }

                int i = 0;
                while (!this->done) {
                    auto batch = upload_batch(i);
                    while (!queue.try_push(batch) && !this->done) {
                        std::this_thread::yield();
                    }
                    i = (i + 1) % buffer_size;
                }

                // Free resources
                cuda_check(cuStreamSynchronize(stream));
                for (int i = 0; i < buffer_size; i++) {
                    cuda_check(cuMemFree(gpu_batch_buffer[i]));
                    cuda_check(cuEventDestroy(events[i]));
                }
                cuda_check(cuStreamDestroy(stream));
            } catch (...) {
                this->worker_exception = std::current_exception();
                this->done = true;
            }
        });
    }

    ~BatchFetcherImpl() {
        done = true;
        worker.join();
    }

    void *next() override {
        Batch batch = {};
        while (!done && !queue.try_pop(batch)) {
            std::this_thread::yield();
        }
        if (done) {
            worker.join();
            std::rethrow_exception(worker_exception);
        }
        cuda_check(cuEventSynchronize(batch.data_ready_event));
        return reinterpret_cast<void *>(batch.data);
    }

  private:
    std::exception_ptr worker_exception;
    std::thread worker;
    std::atomic_bool done;
    atomic_queue::AtomicQueue2<Batch, buffer_size - 2> queue;
};

std::unique_ptr<BatchFetcher> create_batch_fetcher(const void *data,
                                                   size_t num_bytes,
                                                   size_t stride,
                                                   size_t batch_size,
                                                   bool shuffle) {
    return std::make_unique<BatchFetcherImpl>(
        static_cast<const uint8_t *>(data),
        num_bytes,
        stride,
        batch_size,
        shuffle);
}

} // namespace radfoam