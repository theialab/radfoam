#pragma once

#include <memory>

#include "../utils/typing.h"
#include "camera.h"

namespace radfoam {

struct TraceSettings {
    float weight_threshold;
    uint32_t max_intersections;
};

inline TraceSettings default_trace_settings() {
    TraceSettings settings;
    settings.weight_threshold = 0.001f;
    settings.max_intersections = 1024;
    return settings;
}

enum VisualizationMode {
    RGB = 0,
    Depth = 1,
    Alpha = 2,
    Intersections = 3,
};

struct VisualizationSettings {
    VisualizationMode mode;
    ColorMap color_map;
    CVec3f bg_color;
    bool checker_bg;
    float max_depth;
    float depth_quantile;
};

inline VisualizationSettings default_visualization_settings() {
    VisualizationSettings settings;
    settings.mode = RGB;
    settings.color_map = Turbo;
    settings.bg_color = Vec3f(1.0f, 1.0f, 1.0f);
    settings.checker_bg = false;
    settings.max_depth = 10.0f;
    settings.depth_quantile = 0.5f;
    return settings;
}

/// @brief Prefetch offset for each edge in the adjacency matrix
void prefetch_adjacent_diff(const Vec3f *points,
                            uint32_t num_points,
                            uint32_t point_adjacency_size,
                            const uint32_t *point_adjacency,
                            const uint32_t *point_adjacency_offsets,
                            Vec4h *adjacent_diff,
                            const void *stream);

class Pipeline {
  public:
    virtual ~Pipeline() = default;

    virtual void trace_forward(const TraceSettings &settings,
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
                               void *point_contribution) = 0;

    virtual void trace_backward(const TraceSettings &settings,
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
                                void *point_error) = 0;

    virtual void trace_visualization(const TraceSettings &settings,
                                     const VisualizationSettings &vis_settings,
                                     const Camera &camera,
                                     CMapTable cmap_table,
                                     uint32_t num_points,
                                     uint32_t num_tets,
                                     const void *points,
                                     const void *attributes,
                                     const void *point_adjacency,
                                     const void *point_adjacency_offsets,
                                     const void *adjacent_points,
                                     uint32_t start_index,
                                     uint64_t output_surface,
                                     const void *stream = nullptr) = 0;

    virtual void trace_benchmark(const TraceSettings &settings,
                                 uint32_t num_points,
                                 const Vec3f *points,
                                 const void *attributes,
                                 const uint32_t *point_adjacency,
                                 const uint32_t *point_adjacency_offsets,
                                 const Vec4h *adjacent_diff,
                                 Camera camera,
                                 const uint32_t *start_point_index,
                                 uint32_t *ray_rgba) = 0;

    virtual uint32_t attribute_dim() const = 0;

    virtual ScalarType attribute_type() const = 0;
};

std::shared_ptr<Pipeline> create_pipeline(int sh_degree, ScalarType attr_type);

} // namespace radfoam