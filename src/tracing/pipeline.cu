#include "../aabb_tree/aabb_tree.h"
#include "../delaunay/triangulation_ops.h"
#include "../utils/cuda_array.h"
#include "../utils/cuda_helpers.h"
#include "../utils/geometry.h"
#include "pipeline.h"

#include "../utils/common_kernels.cuh"
#include "sh_utils.cuh"
#include "tracing_utils.cuh"

namespace radfoam {

template <typename attr_scalar, int sh_degree, int block_size>
__global__ void forward(TraceSettings settings,
                        const Vec3f *__restrict__ points,
                        const attr_scalar *__restrict__ attributes,
                        const uint32_t *__restrict__ point_adjacency,
                        const uint32_t *__restrict__ point_adjacency_offsets,
                        const Vec4h *__restrict__ adjacent_diff,
                        const Ray *__restrict__ rays,
                        uint32_t num_rays,
                        const uint32_t *__restrict__ start_point_index,
                        uint32_t num_depth_quantiles,
                        const float *__restrict__ depth_quantiles,
                        attr_scalar *__restrict__ ray_rgba,
                        float *__restrict__ quantile_depths,
                        uint32_t *__restrict__ quantile_point_indices,
                        uint32_t *__restrict__ num_intersections,
                        attr_scalar *__restrict__ point_contribution) {

    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_rays)
        return;

    constexpr int sh_dim = 3 * (1 + sh_degree) * (1 + sh_degree);
    constexpr int attr_memory_size = 1 + sh_dim;

    Ray ray = rays[thread_idx];
    ray.direction /= ray.direction.norm();

    const float *ray_depth_quantiles =
        depth_quantiles + thread_idx * num_depth_quantiles;

    auto sh_coeffs = sh_coefficients<sh_degree>(ray.direction);

    auto load_attributes = [&](uint32_t v_idx, Vec3f &rgb, float &s) {
        const attr_scalar *attr_ptr = attributes + v_idx * attr_memory_size;
        s = (float)attr_ptr[attr_memory_size - 1];
        if (s > 1e-6f) {
            rgb = load_sh_as_rgb<attr_scalar, sh_degree>(sh_coeffs, attr_ptr);
        } else {
            rgb = Vec3f::Zero();
        }
    };

    float transmittance = 1.0f;
    Vec3f accumulated_rgb = Vec3f::Zero();

    uint32_t current_quantile_idx = 0;
    float current_quantile;
    if (depth_quantiles) {
        current_quantile = ray_depth_quantiles[current_quantile_idx];
    }

    auto functor = [&](uint32_t point_idx,
                       float t_0,
                       float t_1,
                       const Vec3f &current_point,
                       const Vec3f &next_point) {
        Vec3f rgb_primal;
        float s_primal;

        load_attributes(point_idx, rgb_primal, s_primal);

        float delta_t = fmaxf(t_1 - t_0, 0.0f);
        float alpha = 1 - expf(-s_primal * delta_t);
        float weight = transmittance * alpha;

        if (point_contribution) {
            atomicAdd(point_contribution + point_idx, (attr_scalar)weight);
        }
        accumulated_rgb += weight * rgb_primal;

        float next_transmittance = transmittance * (1 - alpha);
        while (current_quantile_idx < num_depth_quantiles &&
               next_transmittance < current_quantile) {
            quantile_depths[thread_idx * num_depth_quantiles +
                            current_quantile_idx] =
                t_0 + logf(transmittance / current_quantile) / s_primal;
            quantile_point_indices[thread_idx * num_depth_quantiles +
                                   current_quantile_idx] = point_idx;
            current_quantile_idx++;
            if (current_quantile_idx < num_depth_quantiles) {
                current_quantile = ray_depth_quantiles[current_quantile_idx];
            }
        }

        transmittance = next_transmittance;

        return transmittance > settings.weight_threshold;
    };

    uint32_t start_point = start_point_index[thread_idx];

    uint32_t n = trace<block_size, 4>(ray,
                                      points,
                                      point_adjacency,
                                      point_adjacency_offsets,
                                      adjacent_diff,
                                      start_point,
                                      settings.max_intersections,
                                      functor);

    while (current_quantile_idx < num_depth_quantiles) {
        quantile_depths[thread_idx * num_depth_quantiles +
                        current_quantile_idx] = -1.0f;
        quantile_point_indices[thread_idx * num_depth_quantiles +
                               current_quantile_idx] = UINT32_MAX;
        current_quantile_idx++;
    }

    for (uint32_t i = 0; i < 3; ++i) {
        ray_rgba[thread_idx * 4 + i] = attr_scalar(accumulated_rgb[i]);
    }
    ray_rgba[thread_idx * 4 + 3] = attr_scalar(1 - transmittance);

    if (num_intersections)
        num_intersections[thread_idx] = n;
}

template <typename attr_scalar, int sh_degree, int block_size>
__global__ void backward(TraceSettings settings,
                         const Vec3f *__restrict__ points,
                         const attr_scalar *__restrict__ attributes,
                         const uint32_t *__restrict__ point_adjacency,
                         const uint32_t *__restrict__ point_adjacency_offsets,
                         const Vec4h *__restrict__ adjacent_diff,
                         const Ray *__restrict__ rays,
                         uint32_t num_rays,
                         const uint32_t *__restrict__ start_point_index,
                         uint32_t num_depth_quantiles,
                         const float *__restrict__ depth_quantiles,
                         const uint32_t *__restrict__ quantile_point_indices,
                         const attr_scalar *__restrict__ ray_rgba,
                         const attr_scalar *__restrict__ ray_rgba_grad,
                         const float *__restrict__ depth_grad,
                         const attr_scalar *__restrict__ ray_error,
                         Ray *__restrict__ ray_grad,
                         Vec3f *__restrict__ points_grad,
                         attr_scalar *__restrict__ attribute_grad,
                         attr_scalar *__restrict__ point_error) {

    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_rays)
        return;

    constexpr int sh_dim = 3 * (1 + sh_degree) * (1 + sh_degree);
    constexpr int attr_memory_size = 1 + sh_dim;

    Ray ray = rays[thread_idx];
    ray.direction /= ray.direction.norm();

    const float *ray_depth_grad = depth_grad + thread_idx * num_depth_quantiles;
    const float *ray_depth_quantiles =
        depth_quantiles + thread_idx * num_depth_quantiles;

    auto sh_coeffs = sh_coefficients<sh_degree>(ray.direction);

    auto load_attributes = [&](uint32_t v_idx, Vec3f &rgb, float &s) {
        const attr_scalar *attr_ptr = attributes + v_idx * attr_memory_size;
        s = (float)attr_ptr[attr_memory_size - 1];
        if (s > 1e-6f) {
            rgb = load_sh_as_rgb<attr_scalar, sh_degree>(sh_coeffs, attr_ptr);
        } else {
            rgb = Vec3f::Zero();
        }
    };

    Vec4f rgba_grad, rgba;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        rgba_grad[i] = (float)ray_rgba_grad[thread_idx * 4 + i];
        rgba[i] = (float)ray_rgba[thread_idx * 4 + i];
    }

    float error;
    if (ray_error) {
        error = (float)ray_error[thread_idx];
    }

    uint32_t current_quantile_idx = 0;
    float current_quantile;
    if (depth_quantiles) {
        current_quantile = ray_depth_quantiles[current_quantile_idx];
    }
    float current_depth_grad = 0.0f;
    for (uint32_t i = 0; i < num_depth_quantiles; ++i) {
        if (quantile_point_indices[thread_idx * num_depth_quantiles + i] !=
            UINT32_MAX) {
            uint32_t point_idx =
                quantile_point_indices[thread_idx * num_depth_quantiles + i];
            float s = (float)
                attributes[point_idx * attr_memory_size + attr_memory_size - 1];
            current_depth_grad += ray_depth_grad[i] / s;
        }
    }

    float transmittance = 1.0f;
    Vec3f accumulated_rgb = Vec3f::Zero();

    uint32_t prev_point_idx = UINT32_MAX;
    Vec3f prev_point = Vec3f::Zero();
    Vec3f prev_point_grad = Vec3f::Zero();

    Vec3f current_point_grad = Vec3f::Zero();
    Vec3f next_point_grad = Vec3f::Zero();

    auto functor = [&](uint32_t point_idx,
                       float t_0,
                       float t_1,
                       const Vec3f &current_point,
                       const Vec3f &next_point) {
        Vec3f rgb_primal;
        float s_primal;

        load_attributes(point_idx, rgb_primal, s_primal);

        float delta_t = fmaxf(t_1 - t_0, 0.0f);
        float alpha = 1 - expf(-s_primal * delta_t);
        float weight = transmittance * alpha;
        float dalpha_ds_primal = delta_t * (1 - alpha);
        float dalpha_ddelta_t = 0.0f;
        if (delta_t > 0.0f) {
            dalpha_ddelta_t = s_primal * (1 - alpha);
        }

        accumulated_rgb += weight * rgb_primal;
        if (point_error) {
            atomicAdd(point_error + point_idx, (attr_scalar)(weight * error));
        }

        Vec3f dL_drgb_primal = rgba_grad.template head<3>() * weight;

        Vec3f rgb_rest = rgba.template head<3>() - accumulated_rgb;
        rgb_rest /= (transmittance * (1 - alpha + 1e-6f));

        float dL_dalpha =
            transmittance *
            (rgb_primal - rgb_rest).dot(rgba_grad.template head<3>());
        dL_dalpha += (1 - rgba[3]) * rgba_grad[3] / (1 - alpha + 1e-6f);

        float dL_ds_primal = dL_dalpha * dalpha_ds_primal;
        float dL_ddelta_t = dL_dalpha * dalpha_ddelta_t;

        float dL_dt0 = 0.0f;

        float next_transmittance = transmittance * (1 - alpha);
        while (current_quantile_idx < num_depth_quantiles &&
               next_transmittance < current_quantile) {

            float depth_grad_i =
                ray_depth_grad[current_quantile_idx] / s_primal;
            dL_dt0 += depth_grad_i;
            dL_ds_primal += -depth_grad_i *
                            logf(transmittance / current_quantile) / s_primal;

            current_depth_grad -= depth_grad_i;

            current_quantile_idx++;
            if (current_quantile_idx < num_depth_quantiles) {
                current_quantile = ray_depth_quantiles[current_quantile_idx];
            }
        }

        if (current_quantile_idx < num_depth_quantiles) {
            dL_ds_primal += -delta_t * current_depth_grad;
            dL_ddelta_t += -s_primal * current_depth_grad;
        }

        dL_dt0 += -dL_ddelta_t;
        float dL_dt1 = dL_ddelta_t;

        Vec3f dt0_dprev_point;
        if (prev_point_idx != UINT32_MAX) {
            dt0_dprev_point =
                cell_intersection_grad(prev_point, current_point, ray);
        } else {
            dt0_dprev_point = Vec3f::Zero();
        }

        Vec3f dt1_dcurrent_point =
            cell_intersection_grad(current_point, next_point, ray);
        Vec3f dt0_dcurrent_point =
            cell_intersection_grad(current_point, prev_point, ray);

        Vec3f dt1_dnext_point =
            cell_intersection_grad(next_point, current_point, ray);

        prev_point_grad += dL_dt0 * dt0_dprev_point;
        current_point_grad +=
            dL_dt0 * dt0_dcurrent_point + dL_dt1 * dt1_dcurrent_point;
        next_point_grad += dL_dt1 * dt1_dnext_point;

        if (prev_point_idx != UINT32_MAX) {
            atomic_add_vec(points_grad + prev_point_idx, prev_point_grad);
        }
        prev_point = current_point;
        prev_point_idx = point_idx;
        prev_point_grad = current_point_grad;

        current_point_grad = next_point_grad;
        next_point_grad = Vec3f::Zero();

        transmittance = next_transmittance;

        for (uint32_t i = 0; i < 3; ++i) {
            if (rgb_primal[i] == 0.0f) {
                dL_drgb_primal[i] = 0.0f;
            }
        }
        write_rgb_grad_to_sh<attr_scalar, sh_degree>(
            sh_coeffs,
            dL_drgb_primal,
            attribute_grad + point_idx * attr_memory_size);
        atomicAdd(attribute_grad + point_idx * attr_memory_size +
                      (attr_memory_size - 1),
                  (attr_scalar)dL_ds_primal);

        return transmittance > settings.weight_threshold;
    };

    uint32_t start_point = start_point_index[thread_idx];

    trace<block_size, 2>(ray,
                         points,
                         point_adjacency,
                         point_adjacency_offsets,
                         adjacent_diff,
                         start_point,
                         settings.max_intersections,
                         functor);
}

template <typename attr_scalar, int sh_degree, int block_size>
__global__ void
visualization(TraceSettings settings,
              const Vec3f *__restrict__ points,
              const attr_scalar *__restrict__ attributes,
              const uint32_t *__restrict__ point_adjacency,
              const uint32_t *__restrict__ point_adjacency_offsets,
              const Vec4h *__restrict__ adjacent_diff,
              VisualizationSettings vis_settings,
              CMapTable cmap_table,
              Camera camera,
              cudaSurfaceObject_t output_rgba,
              uint32_t start_point_index) {

    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pix_i = thread_idx % camera.width;
    uint32_t pix_j = thread_idx / camera.width;

    if (pix_i >= camera.width || pix_j >= camera.height)
        return;

    constexpr int sh_dim = 3 * (1 + sh_degree) * (1 + sh_degree);
    constexpr int attr_memory_size = 1 + sh_dim;

    Ray ray = cast_ray(camera, pix_i, pix_j);
    if (ray.direction.norm() < 0.1f) {
        surf2Dwrite(0, output_rgba, 4 * pix_i, camera.height - 1 - pix_j);
        return;
    }

    auto sh_coeffs = sh_coefficients<sh_degree>(ray.direction);

    auto load_attributes = [&](uint32_t v_idx, Vec3f &rgb, float &s) {
        const attr_scalar *attr_ptr = attributes + v_idx * attr_memory_size;
        s = (float)attr_ptr[attr_memory_size - 1];
        if (s > 1e-6f) {
            rgb = load_sh_as_rgb<attr_scalar, sh_degree>(sh_coeffs, attr_ptr);
        } else {
            rgb = Vec3f::Zero();
        }
    };

    float transmittance = 1.0f;
    Vec3f accumulated_rgb = Vec3f::Zero();
    float depth = 0.0f;
    bool depth_quantile_passed = false;

    auto functor = [&](uint32_t point_idx,
                       float t_0,
                       float t_1,
                       const Vec3f &current_point,
                       const Vec3f &next_point) {
        Vec3f rgb_primal;
        float s_primal;

        load_attributes(point_idx, rgb_primal, s_primal);

        float delta_t = fmaxf(t_1 - t_0, 0.0f);
        float alpha = 1 - expf(-s_primal * delta_t);

        accumulated_rgb += transmittance * alpha * rgb_primal;

        float next_transmittance = transmittance * (1 - alpha);
        if (!depth_quantile_passed &&
            next_transmittance < vis_settings.depth_quantile) {
            depth = t_0 + logf(transmittance / vis_settings.depth_quantile) /
                              s_primal;
            depth_quantile_passed = true;
        }

        transmittance = next_transmittance;

        return transmittance > settings.weight_threshold;
    };

    uint32_t n = trace<block_size, 4>(ray,
                                      points,
                                      point_adjacency,
                                      point_adjacency_offsets,
                                      adjacent_diff,
                                      start_point_index,
                                      settings.max_intersections,
                                      functor);

    uint32_t out;

    if (vis_settings.mode == VisualizationMode::RGB) {
        Vec3f color = accumulated_rgb;

        Vec3f bg_color;
        if (vis_settings.checker_bg) {
            int is = 2 * ((pix_i / 20) % 2) - 1;
            int js = 2 * ((pix_j / 20) % 2) - 1;
            if (is * js > 0) {
                bg_color = Vec3f(0.3f, 0.3f, 0.3f);
            } else {
                bg_color = Vec3f(0.5f, 0.5f, 0.5f);
            }
        } else {
            bg_color = *vis_settings.bg_color;
        }

        color += transmittance * bg_color;

        out = make_rgba8(color[0], color[1], color[2], 1.0f);
    } else if (vis_settings.mode == VisualizationMode::Depth) {
        float val = depth / vis_settings.max_depth;

        Vec3f color = colormap(val, vis_settings.color_map, cmap_table);

        out = make_rgba8(color[0], color[1], color[2], 1.0f);
    } else if (vis_settings.mode == VisualizationMode::Alpha) {
        out = make_rgba8(1.0f - transmittance,
                         1.0f - transmittance,
                         1.0f - transmittance,
                         1.0f);
    } else if (vis_settings.mode == VisualizationMode::Intersections) {
        float val = float(n - 1) / float(settings.max_intersections);

        Vec3f color = colormap(val, vis_settings.color_map, cmap_table);

        out = make_rgba8(color[0], color[1], color[2], 1.0f);
    }

    surf2Dwrite(out, output_rgba, 4 * pix_i, camera.height - 1 - pix_j);
}

template <typename attr_scalar, int sh_degree, int block_size>
__global__ void benchmark(TraceSettings settings,
                          const Vec3f *__restrict__ points,
                          const attr_scalar *__restrict__ attributes,
                          const uint32_t *__restrict__ point_adjacency,
                          const uint32_t *__restrict__ point_adjacency_offsets,
                          const Vec4h *__restrict__ adjacent_diff,
                          Camera camera,
                          const uint32_t *__restrict__ start_point_index,
                          uint32_t *__restrict__ output_rgba) {

    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pix_i = thread_idx % camera.width;
    uint32_t pix_j = thread_idx / camera.width;

    if (pix_i >= camera.width || pix_j >= camera.height)
        return;

    constexpr int sh_dim = 3 * (1 + sh_degree) * (1 + sh_degree);
    constexpr int attr_memory_size = 1 + sh_dim;

    Ray ray = cast_ray(camera, pix_i, pix_j);
    if (ray.direction.norm() < 0.1f) {
        output_rgba[thread_idx] = 0;
        return;
    }

    auto sh_coeffs = sh_coefficients<sh_degree>(ray.direction);

    auto load_attributes = [&](uint32_t v_idx, Vec3f &rgb, float &s) {
        const attr_scalar *attr_ptr = attributes + v_idx * attr_memory_size;
        s = (float)attr_ptr[attr_memory_size - 1];
        if (s > 1e-6f) {
            rgb = load_sh_as_rgb<attr_scalar, sh_degree>(sh_coeffs, attr_ptr);
        } else {
            rgb = Vec3f::Zero();
        }
    };

    float transmittance = 1.0f;
    Vec3f accumulated_rgb = Vec3f::Zero();

    auto functor = [&](uint32_t point_idx,
                       float t_0,
                       float t_1,
                       const Vec3f &current_point,
                       const Vec3f &next_point) {
        Vec3f rgb_primal;
        float s_primal;

        load_attributes(point_idx, rgb_primal, s_primal);

        float delta_t = fmaxf(t_1 - t_0, 0.0f);
        float alpha = 1 - expf(-s_primal * delta_t);

        accumulated_rgb += transmittance * alpha * rgb_primal;
        transmittance = transmittance * (1 - alpha);

        return transmittance > settings.weight_threshold;
    };

    uint32_t n = trace<block_size, 4>(ray,
                                      points,
                                      point_adjacency,
                                      point_adjacency_offsets,
                                      adjacent_diff,
                                      *start_point_index,
                                      settings.max_intersections,
                                      functor);

    output_rgba[thread_idx] = make_rgba8(
        accumulated_rgb[0], accumulated_rgb[1], accumulated_rgb[2], 1.0f);
}

__global__ void prefetch_adjacent_diff_kernel(
    const Vec3f *__restrict__ points,
    uint32_t num_points,
    uint32_t point_adjacency_size,
    const uint32_t *__restrict__ point_adjacency,
    const uint32_t *__restrict__ point_adjacency_offsets,
    Vec4h *__restrict__ adjacent_diff) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points)
        return;

    Vec3f p = points[i];
    uint32_t offset_start = point_adjacency_offsets[i];
    uint32_t offset_end = point_adjacency_offsets[i + 1];
    uint32_t num_adjacent = offset_end - offset_start;

    for (uint32_t j = 0; j < num_adjacent; ++j) {
        uint32_t adjacent_idx = point_adjacency[offset_start + j];
        Vec3f q = points[adjacent_idx];
        Vec3f diff = q - p;
        adjacent_diff[offset_start + j] = Vec4h(diff[0], diff[1], diff[2], 0);
    }
}

void prefetch_adjacent_diff(const Vec3f *points,
                            uint32_t num_points,
                            uint32_t point_adjacency_size,
                            const uint32_t *point_adjacency,
                            const uint32_t *point_adjacency_offsets,
                            Vec4h *adjacent_diff,
                            const void *stream) {
    launch_kernel_1d<256>(prefetch_adjacent_diff_kernel,
                          num_points,
                          stream,
                          points,
                          num_points,
                          point_adjacency_size,
                          point_adjacency,
                          point_adjacency_offsets,
                          adjacent_diff);
}

template <typename attr_scalar, int sh_degree>
class CUDATracingPipeline : public Pipeline {
  public:
    CUDATracingPipeline() = default;

    virtual ~CUDATracingPipeline() {}

    void trace_forward(const TraceSettings &settings,
                       uint32_t num_points,
                       const Vec3f *points,
                       const void *attributes,
                       uint32_t point_adjacency_size,
                       const uint32_t *point_adjacency,
                       const uint32_t *point_adjacency_offsets,
                       uint32_t num_rays,
                       const Ray *rays,
                       const uint32_t *start_point_index,
                       uint32_t num_depth_quantiles,
                       const float *depth_quantiles,
                       void *ray_rgba,
                       float *quantile_dpeths,
                       uint32_t *quantile_point_indices,
                       uint32_t *num_intersections,
                       void *point_contribution) override {

        CUDAArray<Vec4h> adjacent_diff(point_adjacency_size + 32);
        prefetch_adjacent_diff(reinterpret_cast<const Vec3f *>(points),
                               num_points,
                               point_adjacency_size,
                               point_adjacency,
                               point_adjacency_offsets,
                               adjacent_diff.begin(),
                               nullptr);

        constexpr uint32_t block_size = 128;
        launch_kernel_1d<block_size>(
            forward<attr_scalar, sh_degree, block_size>,
            num_rays,
            nullptr,
            settings,
            points,
            reinterpret_cast<const attr_scalar *>(attributes),
            point_adjacency,
            point_adjacency_offsets,
            adjacent_diff.begin(),
            rays,
            num_rays,
            start_point_index,
            num_depth_quantiles,
            depth_quantiles,
            static_cast<attr_scalar *>(ray_rgba),
            quantile_dpeths,
            quantile_point_indices,
            num_intersections,
            static_cast<attr_scalar *>(point_contribution));
    }

    void trace_backward(const TraceSettings &settings,
                        uint32_t num_points,
                        const Vec3f *points,
                        const void *attributes,
                        uint32_t point_adjacency_size,
                        const uint32_t *point_adjacency,
                        const uint32_t *point_adjacency_offsets,
                        uint32_t num_rays,
                        const Ray *rays,
                        const uint32_t *start_point_index,
                        uint32_t num_depth_quantiles,
                        const float *depth_quantiles,
                        const uint32_t *quantile_point_indices,
                        const void *ray_rgba,
                        const void *ray_rgba_grad,
                        const float *depth_grad,
                        const void *ray_error,
                        Ray *ray_grad,
                        Vec3f *points_grad,
                        void *attribute_grad,
                        void *point_error) override {

        CUDAArray<Vec4h> adjacent_diff(point_adjacency_size + 32);
        prefetch_adjacent_diff(reinterpret_cast<const Vec3f *>(points),
                               num_points,
                               point_adjacency_size,
                               point_adjacency,
                               point_adjacency_offsets,
                               adjacent_diff.begin(),
                               nullptr);

        constexpr uint32_t block_size = 128;
        launch_kernel_1d<block_size>(
            backward<attr_scalar, sh_degree, block_size>,
            num_rays,
            nullptr,
            settings,
            points,
            reinterpret_cast<const attr_scalar *>(attributes),
            point_adjacency,
            point_adjacency_offsets,
            adjacent_diff.begin(),
            rays,
            num_rays,
            start_point_index,
            num_depth_quantiles,
            depth_quantiles,
            quantile_point_indices,
            static_cast<const attr_scalar *>(ray_rgba),
            static_cast<const attr_scalar *>(ray_rgba_grad),
            depth_grad,
            static_cast<const attr_scalar *>(ray_error),
            ray_grad,
            points_grad,
            static_cast<attr_scalar *>(attribute_grad),
            static_cast<attr_scalar *>(point_error));
    }

    void trace_visualization(const TraceSettings &settings,
                             const VisualizationSettings &vis_settings,
                             const Camera &camera,
                             CMapTable cmap_table,
                             uint32_t num_points,
                             uint32_t num_tets,
                             const void *points,
                             const void *attributes,
                             const void *point_adjacency,
                             const void *point_adjacency_offsets,
                             const void *adjacent_diff,
                             uint32_t start_index,
                             uint64_t output_surface,
                             const void *stream) override {

        uint32_t num_rays = camera.width * camera.height;
        constexpr uint32_t block_size = 128;

        launch_kernel_1d<block_size>(
            visualization<attr_scalar, sh_degree, block_size>,
            num_rays,
            stream,
            settings,
            reinterpret_cast<const Vec3f *>(points),
            reinterpret_cast<const attr_scalar *>(attributes),
            reinterpret_cast<const uint32_t *>(point_adjacency),
            reinterpret_cast<const uint32_t *>(point_adjacency_offsets),
            reinterpret_cast<const Vec4h *>(adjacent_diff),
            vis_settings,
            cmap_table,
            camera,
            output_surface,
            start_index);
    }

    void trace_benchmark(const TraceSettings &settings,
                         uint32_t num_points,
                         const Vec3f *points,
                         const void *attributes,
                         const uint32_t *point_adjacency,
                         const uint32_t *point_adjacency_offsets,
                         const Vec4h *adjacent_diff,
                         Camera camera,
                         const uint32_t *start_point_index,
                         uint32_t *ray_rgba) override {

        uint32_t num_rays = camera.width * camera.height;

        constexpr uint32_t block_size = 512;
        launch_kernel_1d<block_size>(
            benchmark<attr_scalar, sh_degree, block_size>,
            num_rays,
            nullptr,
            settings,
            points,
            reinterpret_cast<const attr_scalar *>(attributes),
            point_adjacency,
            point_adjacency_offsets,
            adjacent_diff,
            camera,
            start_point_index,
            ray_rgba);
    }

    uint32_t attribute_dim() const override {
        return 1 + 3 * (1 + sh_degree) * (1 + sh_degree);
    }

    ScalarType attribute_type() const override {
        return scalar_code<attr_scalar>();
    }
};

std::shared_ptr<Pipeline> create_pipeline(int sh_degree, ScalarType attr_type) {

    if (attr_type == ScalarType::Float32) {
        if (sh_degree == 1) {
            return std::make_shared<CUDATracingPipeline<float, 1>>();
        } else if (sh_degree == 2) {
            return std::make_shared<CUDATracingPipeline<float, 2>>();
        } else if (sh_degree == 3) {
            return std::make_shared<CUDATracingPipeline<float, 3>>();
        } else {
            throw std::runtime_error("Unsupported SH degree");
        }
    } else if (attr_type == ScalarType::Float16) {
        if (sh_degree == 1) {
            return std::make_shared<CUDATracingPipeline<__half, 1>>();
        } else if (sh_degree == 2) {
            return std::make_shared<CUDATracingPipeline<__half, 2>>();
        } else if (sh_degree == 3) {
            return std::make_shared<CUDATracingPipeline<__half, 3>>();
        } else {
            throw std::runtime_error("Unsupported SH degree");
        }
    } else {
        throw std::runtime_error("Unsupported attribute type");
    }
}

} // namespace radfoam