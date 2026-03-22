// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "initializer_array.h"
#include "vec.h"
#include "crt.h"

namespace wp {

template <typename T> struct quat_t;

template <unsigned Rows, unsigned Cols, typename Type> struct mat_t {
    inline CUDA_CALLABLE mat_t()
        : data()
    {
    }

    inline CUDA_CALLABLE mat_t(Type s)
    {
        for (unsigned i = 0; i < Rows; ++i)
            for (unsigned j = 0; j < Cols; ++j)
                data[i][j] = s;
    }

    template <typename OtherType> inline explicit CUDA_CALLABLE mat_t(const mat_t<Rows, Cols, OtherType>& other)
    {
        for (unsigned i = 0; i < Rows; ++i)
            for (unsigned j = 0; j < Cols; ++j)
                data[i][j] = other.data[i][j];
    }

    inline CUDA_CALLABLE mat_t(vec_t<2, Type> c0, vec_t<2, Type> c1)
    {
        data[0][0] = c0[0];
        data[1][0] = c0[1];

        data[0][1] = c1[0];
        data[1][1] = c1[1];
    }

    inline CUDA_CALLABLE mat_t(vec_t<3, Type> c0, vec_t<3, Type> c1, vec_t<3, Type> c2)
    {
        data[0][0] = c0[0];
        data[1][0] = c0[1];
        data[2][0] = c0[2];

        data[0][1] = c1[0];
        data[1][1] = c1[1];
        data[2][1] = c1[2];

        data[0][2] = c2[0];
        data[1][2] = c2[1];
        data[2][2] = c2[2];
    }

    inline CUDA_CALLABLE mat_t(vec_t<4, Type> c0, vec_t<4, Type> c1, vec_t<4, Type> c2, vec_t<4, Type> c3)
    {
        data[0][0] = c0[0];
        data[1][0] = c0[1];
        data[2][0] = c0[2];
        data[3][0] = c0[3];

        data[0][1] = c1[0];
        data[1][1] = c1[1];
        data[2][1] = c1[2];
        data[3][1] = c1[3];

        data[0][2] = c2[0];
        data[1][2] = c2[1];
        data[2][2] = c2[2];
        data[3][2] = c2[3];

        data[0][3] = c3[0];
        data[1][3] = c3[1];
        data[2][3] = c3[2];
        data[3][3] = c3[3];
    }

    inline CUDA_CALLABLE mat_t(Type m00, Type m01, Type m10, Type m11)
    {
        data[0][0] = m00;
        data[1][0] = m10;
        data[0][1] = m01;
        data[1][1] = m11;
    }

    inline CUDA_CALLABLE mat_t(Type m00, Type m01, Type m02, Type m10, Type m11, Type m12, Type m20, Type m21, Type m22)
    {
        data[0][0] = m00;
        data[1][0] = m10;
        data[2][0] = m20;

        data[0][1] = m01;
        data[1][1] = m11;
        data[2][1] = m21;

        data[0][2] = m02;
        data[1][2] = m12;
        data[2][2] = m22;
    }

    inline CUDA_CALLABLE mat_t(
        Type m00,
        Type m01,
        Type m02,
        Type m03,
        Type m10,
        Type m11,
        Type m12,
        Type m13,
        Type m20,
        Type m21,
        Type m22,
        Type m23,
        Type m30,
        Type m31,
        Type m32,
        Type m33
    )
    {
        data[0][0] = m00;
        data[1][0] = m10;
        data[2][0] = m20;
        data[3][0] = m30;

        data[0][1] = m01;
        data[1][1] = m11;
        data[2][1] = m21;
        data[3][1] = m31;

        data[0][2] = m02;
        data[1][2] = m12;
        data[2][2] = m22;
        data[3][2] = m32;

        data[0][3] = m03;
        data[1][3] = m13;
        data[2][3] = m23;
        data[3][3] = m33;
    }

    inline CUDA_CALLABLE mat_t(const initializer_array<Rows * Cols, Type>& l)
    {
        for (unsigned i = 0; i < Rows; ++i) {
            for (unsigned j = 0; j < Cols; ++j) {
                data[i][j] = l[i * Cols + j];
            }
        }
    }

    inline CUDA_CALLABLE mat_t(const initializer_array<Cols, vec_t<Rows, Type>>& l)
    {
        for (unsigned j = 0; j < Cols; ++j) {
            for (unsigned i = 0; i < Rows; ++i) {
                data[i][j] = l[j][i];
            }
        }
    }

    CUDA_CALLABLE vec_t<Cols, Type> get_row(int index) const
    {
        return reinterpret_cast<const vec_t<Cols, Type>&>(data[index]);
    }

    CUDA_CALLABLE void set_row(int index, const vec_t<Cols, Type>& v)
    {
        reinterpret_cast<vec_t<Cols, Type>&>(data[index]) = v;
    }

    CUDA_CALLABLE vec_t<Rows, Type> get_col(int index) const
    {
        vec_t<Rows, Type> ret;
        for (unsigned i = 0; i < Rows; ++i) {
            ret[i] = data[i][index];
        }
        return ret;
    }

    CUDA_CALLABLE void set_col(int index, const vec_t<Rows, Type>& v)
    {
        for (unsigned i = 0; i < Rows; ++i) {
            data[i][index] = v[i];
        }
    }

    // row major storage assumed to be compatible with PyTorch
    Type data[Rows < 1 ? 1 : Rows][Cols < 1 ? 1 : Cols];
};

// Type trait to detect if a type is a mat_t
template <typename T> struct is_matrix {
    static constexpr bool value = false;
};

template <unsigned Rows, unsigned Cols, typename Type> struct is_matrix<mat_t<Rows, Cols, Type>> {
    static constexpr bool value = true;
};

}  // namespace wp
