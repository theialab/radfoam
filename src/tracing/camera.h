#pragma once

#include "../utils/geometry.h"

namespace radfoam {

struct Ray {
    Vec3f origin;
    Vec3f direction;
};

enum CameraModel {
    Pinhole,
    Fisheye,
};

struct Camera {
    CVec3f position;
    CVec3f forward;
    CVec3f right;
    CVec3f up;
    float fov;
    uint32_t width;
    uint32_t height;
    CameraModel model;

    RADFOAM_HD void rotate(Vec3f axis, float angle) {
        auto rotation = Eigen::AngleAxisf(angle, axis);
        forward = rotation * *forward;
        right = rotation * *right;
        up = rotation * *up;
    }
};

/// @brief Create a camera pointing from position to target
inline RADFOAM_HD Camera look_at(const Vec3f &position,
                                 const Vec3f &target,
                                 const Vec3f &up,
                                 float fov,
                                 uint32_t width,
                                 uint32_t height,
                                 CameraModel model = Pinhole) {
    Camera camera;
    camera.position = position;
    camera.forward = (target - position).normalized();
    camera.right = camera.forward->cross(up).normalized();
    camera.up = camera.right->cross(*camera.forward).normalized();
    camera.fov = fov;
    camera.width = width;
    camera.height = height;
    camera.model = model;
    return camera;
}

/// @brief Create a ray from the camera through pixel (i, j)
inline RADFOAM_HD Ray cast_ray(const Camera &camera, int i, int j) {
    Ray ray;
    ray.origin = *camera.position;
    float aspect_ratio = static_cast<float>(camera.width) / camera.height;
    float x = static_cast<float>(i) / camera.width;
    float y = static_cast<float>(j) / camera.height;

    float u = (2.0f * x - 1.0f) * aspect_ratio;
    float v = (1.0f - 2.0f * y);
    float mask = 1.0f;

    if (camera.model == Pinhole) {
        float w = 1.0f / tanf(camera.fov * 0.5f);
        ray.direction =
            w * *camera.forward + u * *camera.right + v * *camera.up;
    } else if (camera.model == Fisheye) {
        float theta = atan2f(v, u);
        float phi = camera.fov * sqrtf(u * u + v * v);
        if (phi >= M_PIf) {
            phi = M_PIf - 1e-6f;
            mask = 0.0f;
        }
        ray.direction = sinf(phi) * cosf(theta) * *camera.right +
                        sinf(phi) * sinf(theta) * *camera.up +
                        cosf(phi) * *camera.forward;
    }
    ray.direction = ray.direction.normalized() * mask;

    return ray;
}

} // namespace radfoam
