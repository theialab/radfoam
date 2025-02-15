#pragma once

#include "../utils/geometry.h"
#include "shewchuk.cuh"

namespace radfoam {

struct HalfspacePredicate {
    const Vec3f *v0;
    const Vec3f *v1;
    const Vec3f *v2;

    __forceinline__ __device__ HalfspacePredicate(const Vec3f &v0,
                                                  const Vec3f &v1,
                                                  const Vec3f &v2)
        : v0(&v0), v1(&v1), v2(&v2) {}

    __forceinline__ __device__ bool
    check_point(const Vec3f &v3, bool conservative = false) const {
        if (conservative) {
            return orient3dconservative(*v0, *v1, v3, *v2) ==
                   PredicateResult::Inside;
        } else {
            return orient3d(*v0, *v1, v3, *v2) == PredicateResult::Inside;
        }
    }

    __forceinline__ __device__ bool
    check_aabb_conservative(const AABB<float> &aabb) const {
        bool inside = false;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            Vec3f corner((i & 1) ? aabb.min[0] : aabb.max[0],
                         (i & 2) ? aabb.min[1] : aabb.max[1],
                         (i & 4) ? aabb.min[2] : aabb.max[2]);
            inside |= check_point(corner, true);
        }
        return inside;
    }
};

struct EmptyCircumspherePredicate {
    const Vec3f *v0;
    const Vec3f *v1;
    const Vec3f *v2;
    Vec3f v3;
    Vec3f c;

    __forceinline__ __device__ EmptyCircumspherePredicate(const Vec3f &v0,
                                                          const Vec3f &v1,
                                                          const Vec3f &v2)
        : v0(&v0), v1(&v1), v2(&v2), c(Vec3f::Zero()) {
        c = Vec3f(std::numeric_limits<float>::infinity(),
                  std::numeric_limits<float>::infinity(),
                  std::numeric_limits<float>::infinity());
        v3 = Vec3f(std::numeric_limits<float>::infinity(),
                   std::numeric_limits<float>::infinity(),
                   std::numeric_limits<float>::infinity());
    }

    inline __device__ bool check_point(const Vec3f &v3_new,
                                       bool conservative = false) const {
        if (!isfinite(v3[0])) {
            return true;
        }
        if (conservative) {
            return insphereconservative(*v0, *v1, *v2, v3_new, v3) ==
                   PredicateResult::Inside;
        } else {
            return insphere(*v0, *v1, *v2, v3_new, v3) ==
                   PredicateResult::Inside;
        }
    }

    inline __device__ bool
    check_aabb_conservative(const AABB<float> &aabb) const {
        if (!isfinite(c[0])) {
            return true;
        }
        Vec3f x = c.cwiseMin(aabb.max).cwiseMax(aabb.min);
        return check_point(x, true);
    }

    __forceinline__ __device__ bool
    warp_update(const Vec3f &v3_new, bool valid, bool &found) {
        uint32_t valid_mask = __ballot_sync(0xffffffff, valid);
        uint32_t lane_id = threadIdx.x % 32;
        bool is_best_lane = false;
        for (uint32_t i = 0; i < 32; ++i) {
            if (valid_mask & (1 << i)) {
                Vec3f v3i;
                v3i[0] = __shfl_sync(0xffffffff, v3_new[0], i);
                v3i[1] = __shfl_sync(0xffffffff, v3_new[1], i);
                v3i[2] = __shfl_sync(0xffffffff, v3_new[2], i);

                if (!found || check_point(v3i)) {
                    found = true;
                    update(v3i);
                    is_best_lane = lane_id == i;
                }
            }
        }
        return is_best_lane;
    }

    __forceinline__ __device__ void update(const Vec3f &v3_new) {
        v3 = v3_new;

        Vec3f u0 = *v1 - *v0;
        Vec3f u1 = *v2 - *v0;
        Vec3f u2 = v3 - *v0;

        Vec3f w0 = u0.cross(u1);
        Vec3f w1 = u1.cross(u2);
        Vec3f w2 = u2.cross(u0);

        float vol = u0.dot(w1) / 6;
        Vec3f num = u0.squaredNorm() * w1 + u1.squaredNorm() * w2 +
                    u2.squaredNorm() * w0;
        Vec3f x = num / (12 * vol);

        if (isfinite(x[0]) && isfinite(x[1]) && isfinite(x[2])) {
            c = x + *v0;
        }
    }
};

} // namespace radfoam