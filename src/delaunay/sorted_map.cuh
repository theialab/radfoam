#pragma once

#include "../utils/cuda_array.h"

namespace radfoam {

template <typename K, typename V>
struct SortedMap {
    CUDAArray<K> unique_keys;
    CUDAArray<V> unique_values;
    uint32_t num_unique_keys;

    SortedMap() { num_unique_keys = 0; }

    SortedMap(K *keys, V *values, uint32_t num_keys) {
        insert(keys, values, num_keys);
    }

    void insert(K *keys, V *values, uint32_t num_keys) {

        if (values != nullptr) {
            unique_values.resize(num_keys);

            CUB_CALL(cub::DeviceMergeSort::SortPairs(
                temp_data, temp_bytes, keys, values, num_keys, std::less<K>()));
        } else {
            CUB_CALL(cub::DeviceMergeSort::SortKeys(
                temp_data, temp_bytes, keys, num_keys, std::less<K>()));
        }

        auto sorted_keys_begin = keys;

        auto is_valid_unique_key =
            [=] __device__(cub::KeyValuePair<ptrdiff_t, K> pair) {
                auto i = pair.key;
                K key = sorted_keys_begin[i];
                if (!key.is_valid()) {
                    return false;
                }
                if (i > 0) {
                    K prev_key = sorted_keys_begin[i - 1];
                    if (key == prev_key) {
                        return false;
                    }
                }
                return true;
            };

        auto is_valid_unique_value =
            [=] __device__(cub::KeyValuePair<ptrdiff_t, V> pair) {
                auto i = pair.key;
                K key = sorted_keys_begin[i];
                if (!key.is_valid()) {
                    return false;
                }
                if (i > 0) {
                    K prev_key = sorted_keys_begin[i - 1];
                    if (key == prev_key) {
                        return false;
                    }
                }
                return true;
            };

        size_t *num_unique_keys_device;
        cuda_check(
            cuMemAlloc(reinterpret_cast<CUdeviceptr *>(&num_unique_keys_device),
                       sizeof(size_t)));

        unique_keys.resize(num_keys);

        auto enumerated_sorted_keys = enumerate<K>(sorted_keys_begin);
        auto unenumerated_unique_keys = unenumerate<K>(unique_keys.begin());

        CUB_CALL(cub::DeviceSelect::If(temp_data,
                                       temp_bytes,
                                       enumerated_sorted_keys,
                                       unenumerated_unique_keys,
                                       num_unique_keys_device,
                                       num_keys,
                                       is_valid_unique_key));

        if (values != nullptr) {
            unique_values.resize(num_keys);

            auto enumerated_sorted_values = enumerate<V>(values);
            auto unenumerated_unique_values =
                unenumerate<V>(unique_values.begin());

            CUB_CALL(cub::DeviceSelect::If(temp_data,
                                           temp_bytes,
                                           enumerated_sorted_values,
                                           unenumerated_unique_values,
                                           num_unique_keys_device,
                                           num_keys,
                                           is_valid_unique_value));
        }

        cuda_check(
            cuMemcpyDtoH(&num_unique_keys,
                         reinterpret_cast<CUdeviceptr>(num_unique_keys_device),
                         sizeof(size_t)));
        cuda_check(
            cuMemFree(reinterpret_cast<CUdeviceptr>(num_unique_keys_device)));
    }

    void clear() {
        num_unique_keys = 0;
        unique_keys.clear();
        unique_values.clear();
    }
};

} // namespace radfoam