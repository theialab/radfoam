#pragma once

#include "../utils/geometry.h"
#include "camera.h"

namespace radfoam {

__constant__ float C0 = 0.28209479177387814f;
__constant__ float C1 = 0.4886025119029199f;
__constant__ float C2[5] = {1.0925484305920792f,
                            -1.0925484305920792f,
                            0.31539156525252005f,
                            -1.0925484305920792f,
                            0.5462742152960396f};
__constant__ float C3[7] = {-0.5900435899266435f,
                            2.890611442640554f,
                            -0.4570457994644658f,
                            0.3731763325901154f,
                            -0.4570457994644658f,
                            1.445305721320277f,
                            -0.5900435899266435f};
__constant__ float C4[9] = {2.5033429417967046f,
                            -1.7701307697799304f,
                            0.9461746957575601f,
                            -0.6690465435572892f,
                            0.10578554691520431f,
                            -0.6690465435572892f,
                            0.47308734787878004f,
                            -1.7701307697799304f,
                            0.6258357354491761f};

constexpr int sh_dimension(int degree) { return (degree + 1) * (degree + 1); }

template <int degree>
__device__ Vecf<sh_dimension(degree)> sh_coefficients(const Vec3f &dir) {
    float x = dir[0];
    float y = dir[1];
    float z = dir[2];

    Vecf<sh_dimension(degree)> sh = Vecf<sh_dimension(degree)>::Zero();

    sh[0] = C0;

    if (degree > 0) {
        sh[1] = -C1 * y;
        sh[2] = C1 * z;
        sh[3] = -C1 * x;
    }
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    if (degree > 1) {

        sh[4] = C2[0] * xy;
        sh[5] = C2[1] * yz;
        sh[6] = C2[2] * (2.0f * zz - xx - yy);
        sh[7] = C2[3] * xz;
        sh[8] = C2[4] * (xx - yy);
    }
    if (degree > 2) {
        sh[9] = C3[0] * y * (3.0f * xx - yy);
        sh[10] = C3[1] * xy * z;
        sh[11] = C3[2] * y * (4.0f * zz - xx - yy);
        sh[12] = C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
        sh[13] = C3[4] * x * (4.0f * zz - xx - yy);
        sh[14] = C3[5] * z * (xx - yy);
        sh[15] = C3[6] * x * (xx - 3.0f * yy);
    }

    return sh;
}

template <typename scalar, int degree>
__device__ Vec3f load_sh_as_rgb(const Vecf<sh_dimension(degree)> &coeffs,
                                const scalar *sh_rgb_vals) {
    Vec3f rgb = Vec3f(0.5f, 0.5f, 0.5f);

#pragma unroll
    for (uint32_t i = 0; i < 3 * sh_dimension(degree); ++i) {
        rgb[i % 3] += coeffs[i / 3] * (float)sh_rgb_vals[i];
    }

    return rgb.cwiseMax(0.0f);
}

template <typename scalar, int degree>
__device__ void write_rgb_grad_to_sh(const Vecf<sh_dimension(degree)> &coeffs,
                                     Vec3f grad_rgb,
                                     scalar *sh_rgb_grad) {
    for (uint32_t i = 0; i < 3 * sh_dimension(degree); ++i) {
        atomicAdd(sh_rgb_grad + i, (scalar)(coeffs[i / 3] * grad_rgb[i % 3]));
    }
}

template <typename attr_scalar, int sh_dim>
__device__ Vec3<attr_scalar>
forward_sh(uint32_t deg, Vec<attr_scalar, sh_dim> sh_vec, Vec3f dirs) {
    float x = dirs[0];
    float y = dirs[1];
    float z = dirs[2];

    constexpr int sh_vars = int(sh_dim / 3);
    Eigen::Map<Eigen::Matrix<attr_scalar, 3, sh_vars>> sh_mat(sh_vec.data());

    Vec3<attr_scalar> result = C0 * sh_mat.col(0);
    if (deg > 0) {
        result = result - C1 * y * sh_mat.col(1) + C1 * z * sh_mat.col(2) -
                 C1 * x * sh_mat.col(3);
    }

    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    if (deg > 1) {
        result = result + C2[0] * xy * sh_mat.col(4) +
                 C2[1] * yz * sh_mat.col(5) +
                 C2[2] * (2.0f * zz - xx - yy) * sh_mat.col(6) +
                 C2[3] * xz * sh_mat.col(7) + C2[4] * (xx - yy) * sh_mat.col(8);
    }
    if (deg > 2) {
        result =
            result + C3[0] * y * (3.0f * xx - yy) * sh_mat.col(9) +
            C3[1] * xy * z * sh_mat.col(10) +
            C3[2] * y * (4.0f * zz - xx - yy) * sh_mat.col(11) +
            C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh_mat.col(12) +
            C3[4] * x * (4.0f * zz - xx - yy) * sh_mat.col(13) +
            C3[5] * z * (xx - yy) * sh_mat.col(14) +
            C3[6] * x * (xx - 3.0f * yy) * sh_mat.col(15);
    }

    result = result.array() + 0.5f;
    return result;
}

template <typename attr_scalar, int sh_dim>
__device__ Vec<attr_scalar, sh_dim>
backward_sh(uint32_t deg, Vec3<attr_scalar> pd_color, Vec3f dirs) {
    float x = dirs[0];
    float y = dirs[1];
    float z = dirs[2];

    constexpr int sh_vars = int(sh_dim / 3);
    Eigen::Matrix<attr_scalar, 3, sh_vars> pd_sh;

    pd_sh.col(0) = C0 * pd_color;
    if (deg > 0) {
        pd_sh.col(1) = -C1 * y * pd_color;
        pd_sh.col(2) = C1 * z * pd_color;
        pd_sh.col(3) = -C1 * x * pd_color;

        if (deg > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;

            pd_sh.col(4) = C2[0] * xy * pd_color;
            pd_sh.col(5) = C2[1] * yz * pd_color;
            pd_sh.col(6) = C2[2] * (2.0f * zz - xx - yy) * pd_color;
            pd_sh.col(7) = C2[3] * xz * pd_color;
            pd_sh.col(8) = C2[4] * (xx - yy) * pd_color;

            if (deg > 2) {
                pd_sh.col(9) = C3[0] * y * (3.0f * xx - yy) * pd_color;
                pd_sh.col(10) = C3[1] * xy * z * pd_color;
                pd_sh.col(11) = C3[2] * y * (4.0f * zz - xx - yy) * pd_color;
                pd_sh.col(12) =
                    C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * pd_color;
                pd_sh.col(13) = C3[4] * x * (4.0f * zz - xx - yy) * pd_color;
                pd_sh.col(14) = C3[5] * z * (xx - yy) * pd_color;
                pd_sh.col(15) = C3[6] * x * (xx - 3.0f * yy) * pd_color;
            }
        }
    }

    Eigen::Map<Vec<attr_scalar, sh_dim>> pd_sh_vector(pd_sh.data());
    return pd_sh_vector;
}

} // namespace radfoam
