#pragma once

#include <float.h>

#define EIGEN_MPL2_ONLY
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cuda_runtime.h>

#include "random.h"
#include "typing.h"

namespace radfoam {

template <typename scalar, int dim>
using Vec = Eigen::Matrix<scalar, dim, 1>;

template <typename scalar>
using Vec2 = Eigen::Matrix<scalar, 2, 1>;

template <typename scalar>
using Vec3 = Eigen::Matrix<scalar, 3, 1>;

template <typename scalar>
using Vec4 = Eigen::Matrix<scalar, 4, 1>;

template <typename scalar, int rows, int cols>
using Mat = Eigen::Matrix<scalar, rows, cols>;

template <typename scalar>
using Mat3 = Eigen::Matrix<scalar, 3, 3>;

template <typename scalar>
using Mat4 = Eigen::Matrix<scalar, 4, 4>;

template <int dim>
using Vecf = Vec<float, dim>;

using Vec2f = Vec2<float>;
using Vec3f = Vec3<float>;
using Vec4f = Vec4<float>;

using Mat3f = Mat3<float>;
using Mat4f = Mat4<float>;

template <int dim>
using Vecd = Vec<double, dim>;

using Vec2d = Vec2<double>;
using Vec3d = Vec3<double>;
using Vec4d = Vec4<double>;

using Mat3d = Mat3<double>;
using Mat4d = Mat4<double>;

template <int dim>
using Vech = Vec<__half, dim>;

using Vec2h = Vec2<__half>;
using Vec3h = Vec3<__half>;
using Vec4h = Vec4<__half>;

using Mat3h = Mat3<__half>;
using Mat4h = Mat4<__half>;

template <int dim>
using Veci = Vec<int32_t, dim>;

using Vec2i = Vec2<int32_t>;
using Vec3i = Vec3<int32_t>;
using Vec4i = Vec4<int32_t>;

using Mat3i = Mat<int32_t, 3, 3>;
using Mat4i = Mat<int32_t, 4, 4>;

template <int dim>
using Vecl = Vec<int64_t, dim>;

using Vec2l = Vec2<int64_t>;
using Vec3l = Vec3<int64_t>;
using Vec4l = Vec4<int64_t>;

using Mat3l = Mat<int64_t, 3, 3>;
using Mat4l = Mat<int64_t, 4, 4>;

/// @brief A default-initializable matrix type that can be dereferenced to an
/// Eigen matrix
template <typename scalar, int rows, int cols>
struct PODMat {
    scalar data[rows * cols];

    PODMat() = default;

    RADFOAM_HD PODMat(const Eigen::Matrix<scalar, rows, cols> &mat) {
        memcpy(data, mat.data(), sizeof(data));
    }

    RADFOAM_HD Eigen::Matrix<scalar, rows, cols> &operator*() {
        return *reinterpret_cast<Eigen::Matrix<scalar, rows, cols> *>(data);
    }

    RADFOAM_HD const Eigen::Matrix<scalar, rows, cols> &operator*() const {
        return *reinterpret_cast<const Eigen::Matrix<scalar, rows, cols> *>(
            data);
    }

    RADFOAM_HD Eigen::Matrix<scalar, rows, cols> *operator->() {
        return reinterpret_cast<Eigen::Matrix<scalar, rows, cols> *>(data);
    }

    RADFOAM_HD const Eigen::Matrix<scalar, rows, cols> *operator->() const {
        return reinterpret_cast<const Eigen::Matrix<scalar, rows, cols> *>(
            data);
    }
};

template <typename scalar>
using CVec2 = PODMat<scalar, 2, 1>;

template <typename scalar>
using CVec3 = PODMat<scalar, 3, 1>;

template <typename scalar>
using CVec4 = PODMat<scalar, 4, 1>;

template <typename scalar>
using CMat3 = PODMat<scalar, 3, 3>;

template <typename scalar>
using CMat4 = PODMat<scalar, 4, 4>;

using CVec2f = CVec2<float>;
using CVec3f = CVec3<float>;
using CVec4f = CVec4<float>;

using CMat3f = CMat3<float>;
using CMat4f = CMat4<float>;

#ifdef __CUDACC__
template <typename vec>
__device__ void atomic_add_vec(vec *dst, const vec &src) {
    for (int i = 0; i < src.size(); ++i) {
        atomicAdd(&((*dst)[i]), src[i]);
    }
}
#endif

/// @brief Generate a random 3-vector with each component in the range [0, 1]
inline RADFOAM_HD Vec3f rand3(RNGState &rngstate) {
    return Vec3f(rand(rngstate), rand(rngstate), rand(rngstate));
}

/// @brief Generate a random 3-vector from a unit normal distribution
inline RADFOAM_HD Vec3f randn3(RNGState &rngstate) {
    return Vec3f(randn(rngstate), randn(rngstate), randn(rngstate));
}

/// @brief Generate a random unit 3-vector from a uniform distribution on S^2
inline RADFOAM_HD Vec3f rand_unit3(RNGState &rngstate) {
    float theta = 2 * M_PIf * rand(rngstate);
    float z = 2 * rand(rngstate) - 1;
    float r = sqrtf(1 - z * z);
    return Vec3f(r * cosf(theta), r * sinf(theta), z);
}

/// @brief Enum representing the result of an intersection test
enum class IntersectionResult {
    Inside,
    Outside,
    Intersecting,
};

/// @brief Compare two sets of n elements for equality regardless of order
template <int n, typename T>
RADFOAM_HD bool set_equal(const T *a, const T *b) {
    bool equal = true;
#pragma unroll
    for (int i = 0; i < n; i++) {
        bool found = false;
#pragma unroll
        for (int j = 0; j < n; j++) {
            found |= (a[i] == b[j]);
        }
        equal &= found;
    }
    return equal;
}

/// @brief Check if a set of n elements contains a given element
template <int n, typename T>
RADFOAM_HD bool set_contains(const T *a, const T &b) {
    bool found = false;
#pragma unroll
    for (int i = 0; i < n; i++) {
        found |= (a[i] == b);
    }
    return found;
}

/// @brief Sort an array of n elements in place
template <int n, typename T>
RADFOAM_HD void sort_in_place(T *a) {
#pragma unroll
    for (int i = 0; i < n; i++) {
#pragma unroll
        for (int j = i + 1; j < n; j++) {
            if (a[j] < a[i]) {
                swap(a[i], a[j]);
            }
        }
    }
}

/// @brief Compare two sets of n elements lexicographically
template <int n, typename T>
RADFOAM_HD bool set_less(const T *a, const T *b) {
    T a_sorted[n];
    T b_sorted[n];
    memcpy(a_sorted, a, n * sizeof(T));
    memcpy(b_sorted, b, n * sizeof(T));
    sort_in_place<n>(a_sorted);
    sort_in_place<n>(b_sorted);
    for (int i = 0; i < n; i++) {
        if (a_sorted[i] < b_sorted[i]) {
            return true;
        } else if (a_sorted[i] > b_sorted[i]) {
            return false;
        }
    }
    return false;
}

/// @brief Check the parity of a permutation
template <int n, typename T>
RADFOAM_HD bool permutation_parity(const T *perm) {
    int parity = 0;
#pragma unroll
    for (int i = 0; i < n; i++) {
#pragma unroll
        for (int j = i + 1; j < n; j++) {
            parity += (perm[j] < perm[i]);
        }
    }
    return parity % 2 == 0;
}

/// @brief Perform a binary search on a sorted sequence
/// @return An iterator to an element in the sequence that is equal to value, or
/// the end iterator if no such element exists
template <typename Iter>
RADFOAM_HD Iter binary_search(Iter begin,
                              Iter end,
                              const decltype(*begin) &value) {
    Iter low = begin;
    Iter high = end;
    while (low < high) {
        Iter mid = low + (high - low) / 2;
        if (*mid < value) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    if (low < end && *low == value) {
        return low;
    } else {
        return end;
    }
}

/// @brief Compute the Morton code of a 3D point
inline RADFOAM_HD uint32_t morton_code(const Vec3i &coords) {
    uint32_t code = 0;
#pragma unroll
    for (uint32_t i = 0; i < 10; i++) {
        code |= ((coords[0] >> i) & 1) << (3 * i);
        code |= ((coords[1] >> i) & 1) << (3 * i + 1);
        code |= ((coords[2] >> i) & 1) << (3 * i + 2);
    }
    return code;
}

/// @brief Compute the inverse Morton code of a 3D point
inline RADFOAM_HD Vec3i inverse_morton_code(uint32_t code) {
    Vec3i coords(0, 0, 0);
#pragma unroll
    for (uint32_t i = 0; i < 10; i++) {
        coords[0] |= ((code >> (3 * i)) & 1) << i;
        coords[1] |= ((code >> (3 * i + 1)) & 1) << i;
        coords[2] |= ((code >> (3 * i + 2)) & 1) << i;
    }
    return coords;
}

/// @brief An axis-aligned bounding box
template <typename scalar>
struct AABB {
    Vec3<scalar> min;
    Vec3<scalar> max;

    AABB() = default;

    RADFOAM_HD AABB(Vec3<scalar> min, Vec3<scalar> max) : min(min), max(max) {}

    template <typename other_scalar>
    RADFOAM_HD AABB(const AABB<other_scalar> &other)
        : min(other.min.template cast<scalar>()),
          max(other.max.template cast<scalar>()) {}

    /// @brief Computing the common bounding box of two AABBs
    RADFOAM_HD AABB merge(const AABB &other) const {
        return AABB(min.cwiseMin(other.min), max.cwiseMax(other.max));
    }

    /// @brief Query the SDF of the AABB at a given point
    RADFOAM_HD scalar sdf(const Vec3<scalar> &p) const {
        // Calculate the distance to the nearest point inside the AABB
        Vec3<scalar> d =
            (min - p).cwiseMax(Vec3<scalar>::Zero()).cwiseMax(p - max);

        // Calculate the signed distance
        scalar out_dist = d.norm();
        scalar in_dist = std::min(max.x() - p.x(), p.x() - min.x());
        in_dist = std::min(in_dist, std::min(max.y() - p.y(), p.y() - min.y()));
        in_dist = std::min(in_dist, std::min(max.z() - p.z(), p.z() - min.z()));

        return out_dist > scalar(1e-6) ? out_dist : -in_dist;
    }

    /// @brief Check if the AABB intersects a sphere
    /// @param center The center of the sphere
    /// @param radius The radius of the sphere
    ///
    /// @returns IntersectionResult::Outside indicates that the AABB is
    /// entirely outside the sphere, IntersectionResult::Inside indicates that
    /// the AABB is entirely inside the sphere, and
    /// IntersectionResult::Intersecting indicates that the AABB intersects the
    /// sphere.
    RADFOAM_HD IntersectionResult intersects_sphere(const Vec3<scalar> &center,
                                                    scalar radius) const {
        scalar sdf_val = sdf(center);
        if (sdf_val > radius) {
            return IntersectionResult::Outside;
        } else if (sdf_val < -radius) {
            return IntersectionResult::Inside;
        } else {
            return IntersectionResult::Intersecting;
        }
    }
};

/// @brief A triangle represented by indices of its vertices in a point set
struct IndexedTriangle {
    uint32_t vertices[3];

    RADFOAM_HD IndexedTriangle() {}

    RADFOAM_HD IndexedTriangle(uint32_t i0, uint32_t i1, uint32_t i2) {
        vertices[0] = i0;
        vertices[1] = i1;
        vertices[2] = i2;
    }

    __device__ __forceinline__ uint32_t hash() const {
        return mix(vertices[0]) ^ mix(vertices[1]) ^ mix(vertices[2]);
    }

    RADFOAM_HD friend bool operator==(const IndexedTriangle &a,
                                      const IndexedTriangle &b) {
        return set_equal<3>(a.vertices, b.vertices);
    }

    RADFOAM_HD friend bool operator!=(const IndexedTriangle &a,
                                      const IndexedTriangle &b) {
        return !set_equal<3>(a.vertices, b.vertices);
    }

    RADFOAM_HD friend bool operator<(const IndexedTriangle &a,
                                     const IndexedTriangle &b) {
        return set_less<3>(a.vertices, b.vertices);
    }

    RADFOAM_HD bool is_valid() {
        return (vertices[0] < UINT32_MAX) && (vertices[1] < UINT32_MAX) &&
               (vertices[2] < UINT32_MAX);
    }

    /// @brief Compute the normal vector of the triangle given the point set
    template <typename scalar>
    RADFOAM_HD Vec3<scalar> normal(const Vec3<scalar> &v0,
                                   const Vec3<scalar> &v1,
                                   const Vec3<scalar> &v2) const {
        return (v1 - v0).cross(v2 - v1).normalized();
    }

    /// @brief Compute barycentric coordinates given a new point
    template <typename scalar>
    RADFOAM_HD Vec3f barycentric_coords(const Vec3<scalar> *points,
                                        const Vec3f &new_point) {
        Vec3<scalar> p0 = points[vertices[0]];
        Vec3<scalar> p1 = points[vertices[1]];
        Vec3<scalar> p2 = points[vertices[2]];

        Vec3<scalar> AB = p1 - p0;
        Vec3<scalar> AC = p2 - p0;
        Vec3<scalar> BC = p2 - p1;
        Vec3<scalar> AP = new_point - p0;
        Vec3<scalar> BP = new_point - p1;

        float areaABC = AB.cross(AC).norm();
        float areaABP = AB.cross(AP).norm();
        float areaACP = AC.cross(AP).norm();
        float areaBCP = BC.cross(BP).norm();

        float beta = areaACP / (areaABC + 1e-8f);
        float gamma = areaABP / (areaABC + 1e-8f);
        float alpha = areaBCP / (areaABC + 1e-8f);

        return Vec3f(alpha, beta, gamma);
    }

    /// @brief Compute interpolated attributes given a new point
    template <typename scalar, int dim>
    RADFOAM_HD Vec<scalar, dim + 1>
    interpolated_attributes(const Vec<scalar, dim + 1> *attributes,
                            const Vec3f &barycentrics) {
        Vec<scalar, dim + 1> attr0 = attributes[vertices[0]];
        Vec<scalar, dim + 1> attr1 = attributes[vertices[1]];
        Vec<scalar, dim + 1> attr2 = attributes[vertices[2]];

        Vec<scalar, dim + 1> attr_inter = attr0 * barycentrics[0] +
                                          attr1 * barycentrics[1] +
                                          attr2 * barycentrics[2];

        return attr_inter;
    }
};

/// @brief A tetrahedron represented by indices of its vertices in a point set
struct IndexedTet {
    uint32_t vertices[4];

    RADFOAM_HD IndexedTet() {}

    RADFOAM_HD IndexedTet(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) {
        vertices[0] = i0;
        vertices[1] = i1;
        vertices[2] = i2;
        vertices[3] = i3;
    }

    __device__ __forceinline__ uint32_t hash() const {
        return mix(vertices[0]) ^ mix(vertices[1]) ^ mix(vertices[2]) ^
               mix(vertices[3]);
    }

    RADFOAM_HD friend bool operator==(const IndexedTet &a,
                                      const IndexedTet &b) {
        return set_equal<4>(a.vertices, b.vertices);
    }

    RADFOAM_HD friend bool operator<(const IndexedTet &a, const IndexedTet &b) {
        return set_less<4>(a.vertices, b.vertices);
    }

    RADFOAM_HD bool is_valid() {
        return (vertices[0] < UINT32_MAX) && (vertices[1] < UINT32_MAX) &&
               (vertices[2] < UINT32_MAX) && (vertices[3] < UINT32_MAX);
    }

    /// @brief Fetch the face of the tetrahedron opposite to a given vertex
    /// @param i The index of the vertex opposite to the face
    RADFOAM_HD IndexedTriangle face(uint32_t i) const {
        switch (i % 4) {
        case 0:
            return IndexedTriangle(vertices[1], vertices[3], vertices[2]);
        case 1:
            return IndexedTriangle(vertices[0], vertices[2], vertices[3]);
        case 2:
            return IndexedTriangle(vertices[0], vertices[3], vertices[1]);
        default:
            return IndexedTriangle(vertices[0], vertices[1], vertices[2]);
        }
    }

    /// @brief Compute the volume of the tetrahedron given the vertices
    template <typename scalar>
    RADFOAM_HD scalar volume(const Vec3<scalar> &v0,
                             const Vec3<scalar> &v1,
                             const Vec3<scalar> &v2,
                             const Vec3<scalar> &v3) const {
        return (v2 - v0).cross(v1 - v0).dot(v3 - v0) / 6;
    }
};

inline RADFOAM_HD void
tet_permutation(const IndexedTet &old, const IndexedTet &new_, int32_t *perm) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
        perm[i] = -1;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            if (new_.vertices[j] == old.vertices[i]) {
                perm[i] = j;
            }
        }
    }
}

/// @brief An edge represented by indices of its vertices in a point set
struct IndexedEdge {
    uint32_t vertices[2];

    RADFOAM_HD IndexedEdge() {}

    RADFOAM_HD IndexedEdge(uint32_t i0, uint32_t i1) {
        vertices[0] = i0;
        vertices[1] = i1;
    }

    __device__ __forceinline__ uint32_t hash() const {
        return mix(vertices[0]) ^ mix(vertices[1]);
    }

    RADFOAM_HD friend bool operator==(const IndexedEdge &a,
                                      const IndexedEdge &b) {
        return set_equal<2>(a.vertices, b.vertices);
    }

    RADFOAM_HD friend bool operator<(const IndexedEdge &a,
                                     const IndexedEdge &b) {
        return set_less<2>(a.vertices, b.vertices);
    }

    RADFOAM_HD bool is_valid() {
        return (vertices[0] < UINT32_MAX) && (vertices[1] < UINT32_MAX);
    }
};

} // namespace radfoam