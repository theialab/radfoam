#pragma once

#include "../utils/geometry.h"
#include "../utils/random.h"

#include "../aabb_tree/aabb_tree.cuh"
#include "predicate.cuh"

namespace radfoam {

__forceinline__ __device__ uint32_t warp_broadcast(uint32_t value, bool pred) {
    uint32_t mask = __ballot_sync(0xffffffff, pred);
    uint32_t lane_id = __ffs(mask) - 1;
    return __shfl_sync(0xffffffff, value, lane_id);
}

/// @brief Find the nearest neighbours of points in the point set
inline __device__ uint32_t
vertex_nearest_neighbour(const Vec3f *points,
                         const AABB<float> *aabb_tree,
                         uint32_t num_points,
                         uint32_t q_idx,
                         uint32_t *steps_out) {
    uint32_t steps = 0;

    RNGState rng = thread_rng();

    uint32_t base_width = pow2_round_up(num_points);
    uint32_t tree_depth = log2(base_width);

    Vec3f q = points[q_idx];
    float dist = FLT_MAX;

    Vec3f tangent;
    uint32_t tangent_idx;

    uint32_t current_depth = 1;
    uint32_t current_node = 0;
    uint32_t points_per_child = base_width >> 2;
    for (;;) {
        if (current_depth >= tree_depth) {
            break;
        }

        uint32_t pivot = 2 * (current_node + 1) * points_per_child;
        if (q_idx < pivot) {
            current_node = 2 * current_node;
        } else {
            current_node = 2 * (current_node + 1);
        }

        uint32_t point_idx = randint(rng,
                                     current_node * points_per_child,
                                     (current_node + 1) * points_per_child);

        point_idx = min(point_idx, num_points - 1);

        current_depth++;
        points_per_child >>= 1;

        if (q_idx == point_idx)
            continue;

        Vec3f point = points[point_idx];

        float new_dist = (point - q).squaredNorm();

        if (new_dist < dist) {
            current_depth = 1;
            current_node = 0;
            points_per_child = base_width >> 2;

            dist = new_dist;
            tangent_idx = point_idx;
            tangent = point;
        }

        steps++;
    }

    float radius = __fsqrt_ru(dist);

    auto functor = [&](uint32_t current_depth, uint32_t current_node) {
        AABB<float> node =
            get_node(aabb_tree, tree_depth, current_depth, current_node);
        steps++;

        auto status = node.intersects_sphere(q, radius);

        if (status == IntersectionResult::Outside) {
            return TraversalAction::SkipSubtree;
        } else {
            if (current_depth == tree_depth - 1) {
                uint32_t point_start_idx = current_node * 2;

                for (uint32_t i = 0; i < 2; ++i) {
                    uint32_t point_idx = point_start_idx + i;
                    point_idx = min(point_idx, num_points - 1);

                    if (point_idx == q_idx)
                        continue;

                    Vec3f point = points[point_idx];

                    float new_dist = (q - point).squaredNorm();

                    if (new_dist < dist) {
                        dist = new_dist;
                        radius = __fsqrt_ru(dist);
                        tangent_idx = point_idx;
                    }
                    steps++;
                }
            }
            return TraversalAction::Continue;
        }
    };

    traverse(num_points, tree_depth - 1, functor);

    if (steps_out)
        *steps_out += steps;

    return tangent_idx;
}

/// @brief Check if tetrahedra satisfy the Delaunay condition
template <int block_size>
__forceinline__ __device__ bool check_delaunay(const Vec3f *points,
                                               const AABB<float> *aabb_tree,
                                               uint32_t num_points,
                                               const IndexedTet &tet,
                                               uint32_t *steps_out) {
    uint32_t steps = 0;

    Vec3f v0 = points[tet.vertices[0]];
    Vec3f v1 = points[tet.vertices[1]];
    Vec3f v2 = points[tet.vertices[2]];
    Vec3f v3 = points[tet.vertices[3]];

    EmptyCircumspherePredicate predicate(v0, v1, v2);
    predicate.update(v3);

    if (HalfspacePredicate(v0, v2, v1).check_point(v3)) {
        return false;
    }

    uint32_t base_width = pow2_round_up(num_points);
    uint32_t tree_depth = log2(base_width);

    bool condition = true;

    auto functor = [&](uint32_t current_depth, uint32_t current_node) {
        AABB<float> node =
            get_node(aabb_tree, tree_depth, current_depth, current_node);
        steps++;

        if (!predicate.check_aabb_conservative(node)) {
            return TraversalAction::SkipSubtree;
        }

        if (current_depth == tree_depth - 1) {
            uint32_t point_start_idx = current_node * 2;

            for (uint32_t i = 0; i < 2; ++i) {
                uint32_t point_idx = point_start_idx + i;
                point_idx = min(point_idx, num_points - 1);

                bool should_ignore = false;
                for (uint32_t j = 0; j < 4; ++j) {
                    if (point_idx == tet.vertices[j]) {
                        should_ignore = true;
                        break;
                    }
                }

                if (should_ignore)
                    continue;

                Vec3f point = points[point_idx];

                steps++;

                if (predicate.check_point(point)) {
                    condition = false;
                    return TraversalAction::Terminate;
                }
            }
        }
        return TraversalAction::Continue;
    };

    traverse(num_points, tree_depth - 1, functor);

    if (steps_out)
        *steps_out += steps;

    return condition;
}

/// @brief Check if tetrahedra satisfy the Delaunay condition (warp cooperative)
template <int block_size>
__forceinline__ __device__ bool
check_delaunay_warp(const Vec3f *points,
                    const AABB<float> *aabb_tree,
                    uint32_t num_points,
                    const IndexedTet &tet,
                    uint32_t *steps_out) {
    uint32_t steps = 0;

    uint32_t base_width = pow2_round_up(num_points);
    uint32_t tree_depth = log2(base_width);

    Vec3f v0 = points[tet.vertices[0]];
    Vec3f v1 = points[tet.vertices[1]];
    Vec3f v2 = points[tet.vertices[2]];
    Vec3f v3 = points[tet.vertices[3]];

    EmptyCircumspherePredicate predicate(v0, v1, v2);
    predicate.update(v3);

    if (HalfspacePredicate(v0, v2, v1).check_point(v3)) {
        return false;
    }

    __syncwarp(0xffffffff);

    bool condition = true;

    auto leaf_functor = [&](uint32_t point_idx) {
        point_idx = min(point_idx, num_points - 1);

        bool should_ignore = false;
        for (uint32_t j = 0; j < 4; ++j) {
            if (point_idx == tet.vertices[j]) {
                should_ignore = true;
                break;
            }
        }

        Vec3f point = points[point_idx];

        steps++;

        bool flag = predicate.check_point(point, true) && !should_ignore;

        if (!__any_sync(0xffffffff, flag)) {
            return;
        }

        flag = predicate.check_point(point) && !should_ignore;

        if (__any_sync(0xffffffff, flag)) {
            condition = false;
        }
    };

    auto node_functor = [&](uint32_t current_depth, uint32_t current_node) {
        if (!condition)
            return TraversalAction::Terminate;

        if (current_node >= (1 << current_depth))
            return TraversalAction::SkipSubtree;

        AABB<float> node =
            get_node(aabb_tree, tree_depth, current_depth, current_node);
        steps++;

        if (!predicate.check_aabb_conservative(node)) {
            return TraversalAction::SkipSubtree;
        }

        return TraversalAction::Continue;
    };

    warp_traverse<block_size>(num_points, node_functor, leaf_functor);

    if (steps_out)
        *steps_out += steps;

    return condition;
}

/// @brief Find the largest empty sphere that intersects three vertices from the
/// exterior of the triangulation (warp cooperative)
template <int block_size>
inline __device__ uint32_t maximal_empty_sphere(const Vec3f *points,
                                                const AABB<float> *aabb_tree,
                                                uint32_t num_points,
                                                const Vec3f &v0,
                                                const Vec3f &v1,
                                                const Vec3f &v2,
                                                uint32_t t0,
                                                uint32_t t1,
                                                uint32_t t2,
                                                uint32_t num_tangent,
                                                uint32_t *steps_out,
                                                bool *found_out) {
    uint32_t steps = 0;

    RNGState rng = thread_rng();

    uint32_t base_width = pow2_round_up(num_points);
    uint32_t tree_depth = log2(base_width);

    EmptyCircumspherePredicate sphere_predicate(v0, v1, v2);
    HalfspacePredicate plane_predicate(v0, v1, v2);

    __syncwarp(0xffffffff);

    uint32_t tangent_idx = 0;
    bool found = false;

    auto tangent_inds = [&](uint32_t k) {
        switch (k) {
        case 0:
            return t0;
        case 1:
            return t1;
        case 2:
            return t2;
        }
        return UINT32_MAX;
    };

    uint32_t k = 0;
    for (uint32_t i = 0; i < 16; ++i) {
        k = (k + 1) % num_tangent;
        Vec3i t = inverse_morton_code(tangent_inds(k));
        Vec3f offset = randn3(rng) * 4.0f;
        t += offset.template cast<int32_t>();
        t = t.cwiseMax(Vec3i::Zero());
        uint32_t point_idx = morton_code(t);
        point_idx = min(point_idx, num_points - 1);

        bool should_ignore = false;
        for (uint32_t j = 0; j < num_tangent; ++j) {
            if (point_idx == tangent_inds(j)) {
                should_ignore = true;
                break;
            }
        }

        Vec3f point = points[point_idx];

        steps++;

        bool flag = (plane_predicate.check_point(point, true) &&
                     sphere_predicate.check_point(point, true));
        flag &= !should_ignore;

        if (!__any_sync(0xffffffff, flag)) {
            continue;
        }

        flag = (plane_predicate.check_point(point) &&
                sphere_predicate.check_point(point));
        flag &= !should_ignore;

        if (__any_sync(0xffffffff, flag)) {
            bool pred = sphere_predicate.warp_update(point, flag, found);
            tangent_idx = warp_broadcast(point_idx, pred);
        }
    }

    auto leaf_functor = [&](uint32_t point_idx) {
        point_idx = min(point_idx, num_points - 1);

        bool should_ignore = false;
        for (uint32_t j = 0; j < num_tangent; ++j) {
            if (point_idx == tangent_inds(j)) {
                should_ignore = true;
                break;
            }
        }

        Vec3f point = points[point_idx];

        steps++;

        bool flag = (plane_predicate.check_point(point, true) &&
                     sphere_predicate.check_point(point, true));
        flag &= !should_ignore;

        if (!__any_sync(0xffffffff, flag)) {
            return;
        }

        flag = (plane_predicate.check_point(point) &&
                sphere_predicate.check_point(point));
        flag &= !should_ignore;

        if (__any_sync(0xffffffff, flag)) {
            bool pred = sphere_predicate.warp_update(point, flag, found);
            tangent_idx = warp_broadcast(point_idx, pred);
        }
    };

    auto node_functor = [&](uint32_t current_depth, uint32_t current_node) {
        if (current_node >= (1 << current_depth))
            return TraversalAction::SkipSubtree;

        AABB<float> node =
            get_node(aabb_tree, tree_depth, current_depth, current_node);
        steps++;

        if (!plane_predicate.check_aabb_conservative(node)) {
            return TraversalAction::SkipSubtree;
        }

        if (!sphere_predicate.check_aabb_conservative(node)) {
            return TraversalAction::SkipSubtree;
        }

        return TraversalAction::Continue;
    };

    warp_traverse<block_size>(num_points, node_functor, leaf_functor);

    if (steps_out)
        *steps_out += steps;

    if (found_out)
        *found_out = found;

    return tangent_idx;
}

} // namespace radfoam