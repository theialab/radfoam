#pragma once

#include <functional>
#include <memory>

#include "../tracing/pipeline.h"

namespace radfoam {

struct ViewerOptions {
    bool limit_framerate;
    int max_framerate;
    int total_iterations;
    Vec3f camera_pos;
    Vec3f camera_forward;
    Vec3f camera_up;
};

inline ViewerOptions default_viewer_options() {
    ViewerOptions options;
    options.limit_framerate = true;
    options.max_framerate = 20;
    options.total_iterations = 0;
    options.camera_pos = Vec3f(2.5f, 2.5f, 2.5f);
    options.camera_forward = Vec3f(-1.0f, -1.0f, -1.0f).normalized();
    options.camera_up = Vec3f(0.0f, 0.0f, 1.0f);
    return options;
}

class Viewer {
  public:
    ~Viewer() = default;

    virtual void update_scene(uint32_t num_points,
                              uint32_t num_attrs,
                              uint32_t num_point_adjacency,
                              const void *coords,
                              const void *attributes,
                              const void *point_adjacency,
                              const void *point_adjacency_offsets,
                              const void *aabb_tree) = 0;

    virtual void step(int iteration) = 0;

    virtual bool is_closed() const = 0;

    virtual const Pipeline &get_pipeline() const = 0;
};

void run_with_viewer(std::shared_ptr<Pipeline> pipeline,
                     std::function<void(std::shared_ptr<Viewer>)> callback,
                     ViewerOptions options);

} // namespace radfoam