/*
 * Copyright (c) 2019 nicemath contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy 
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#pragma once

#include <cmath>
#include <stdint.h>

/**
 * \mainpage Reference Manual
 * 
 * \section Introduction
 *
 * nicemath is a compact single-header C++ library that provides data types and
 * routines for basic linear algebra operations often encountered in computer
 * graphics and game development.
 *
 * To use the library, simply place
 * <a href = "https://github.com/nicebyte/nicemath/blob/master/nicemath.h">nicemath.h</a> 
 * anywhere within your C++ project's include path. 
 *
 * Please see the <a href="namespacenm.html">list of `nm` namespace members</a> 
 * for detailed documentation.
 * \section Intended Usage
 *
 * This is not a generic linear algebra library. It is mostly intended to
 * help users deal with 3D and 2D affine transforms, quaternions and basic
 * vector operations. 
 * If you require e.g. support for arbitrarily large, sparse matrices,
 * you should look elsewhere.
 *
 * \section Example
 *
 * ```
 * #include "nicemath.h"
 * 
 * void vec_demo(const nm::float4 &a, const nm::float4 &b) {
 *   // basic operations.
 *   const nm::float4 sum = a + b,
 *                    dif = a - b,
 *                    prd = a * b, // elementwise product
 *                    div = a / b; // elementwise division
 *   
 *   // dot product
 *   const float dot = nm::dot(a, b);
 *
 *   // cross product
 *   const nm::float3 cross = nm::cross(a.xyz(), b.xyz());
 *
 *   // arbitrary vector swizzles are supported
 *   const nm::float3 a_wyx = a.wyx(),
 *                    b_zyy = b.zyy();
 *
 *   // vector expressions may be used in c++ constexprs.
 *   constexpr float scale = 2.0f;
 *   constexpr nm::float4 scaled_vector =
 *       scale * nm::float4 { 1.0f, .0f, .0f, .0f};
 * }
 *
 * void mat_demo(nm::float4x4 a, nm::float4x4 b) {
 *   // basic operations
 *   const nm::float4x4 prd = a * b;
 *   
 *   // matrix-vector mul
 *   const nm::float4 v = a * nm::float4 { 1.f, .0f, .0f, .0f };
 *
 *   // inverse
 *   const nm::float4x4 inv = nm::inverse(a);
 *
 *   // determinant
 *   const float det = nm::det(a);
 *
 *   // matrix expressions can be used in c++ contstexprs as well
 *   constexpr nm::float4x4 id = nm::float4x4::identity();
 *   constexpr nm::float4x4 scaled5x = 5.f * id;
 *   constexpr nm::float4x4 m = scaled5x * nm::float4x4::from_rows(nm::float4 { 2.f, 3.f, .0f, .0f },
 *                                                                 nm::float4 { 9.f, 1.f, 2.f, .0f },
 *                                                                 nm::float4 { 8.f, 4.f, .7f, .0f },
 *                                                                 nm::float4 { 6.f, 7.f, 1.f, 1.f });
 * }
 * ```
 */

/**
 * Namespace for all nicemath types and routines.
 */
namespace nm {

constexpr float PI = 3.1415926f;

/**
 * Generic N-dimensional vector. Do not use this class directly.
 */
template <class S, unsigned N>
struct vec;

namespace detail {

constexpr int cestrlen(const char *p) {
  int r = 0;
  while(*(p++)) ++r;
  return r;
}
#define l2i(c) ((c-'w'+3)%4)
#define l2m(c) (uint16_t)(1u << l2i(c))
constexpr uint16_t swzl_mask(const char *p) {
  uint16_t mask = 0x0;
  for (const char *q = p; *q; q++)
    mask |= l2m(*q);
  return mask;
}
#define NMSWZL(swzl) \
  constexpr vec<S, cestrlen( #swzl )> swzl() const { \
    constexpr uint16_t msk = swzl_mask(#swzl); \
    static_assert(!(msk & l2m('z')) || ((msk & l2m('z')) && N >= 3), \
                  "swizzled vector has no z component"); \
    static_assert(!(msk & l2m('w')) || ((msk & l2m('w')) && N >= 4), \
                  "swizzled vector has no w component"); \
    vec<S, cestrlen(#swzl)> result {}; \
    constexpr int r_N = decltype(result)::Dimensionality; \
    for (int i = 0; i < r_N; ++i) { \
      result.data[i] = data[l2i(#swzl[i])]; \
    } \
    return result; \
  }

template <unsigned N, class S>
struct vbase {
  static constexpr unsigned Dimensionality = N;
  using Scalar = S;

  S data[N];

  constexpr vbase() = default;

  template <class... Args>
  constexpr vbase(Args... args) : data{ args... } {
    static_assert(sizeof...(args) == N, "Wrong number of initializers.");
  }

  NMSWZL(xx)
  NMSWZL(xy)
  NMSWZL(xz)
  NMSWZL(xw)
  NMSWZL(yx)
  NMSWZL(yy)
  NMSWZL(yz)
  NMSWZL(yw)
  NMSWZL(zx)
  NMSWZL(zy)
  NMSWZL(zz)
  NMSWZL(zw)
  NMSWZL(wx)
  NMSWZL(wy)
  NMSWZL(wz)
  NMSWZL(ww)
  NMSWZL(xxx)
  NMSWZL(xxy)
  NMSWZL(xxz)
  NMSWZL(xxw)
  NMSWZL(yxx)
  NMSWZL(yxy)
  NMSWZL(yxz)
  NMSWZL(yxw)
  NMSWZL(zxx)
  NMSWZL(zxy)
  NMSWZL(zxz)
  NMSWZL(zxw)
  NMSWZL(wxx)
  NMSWZL(wxy)
  NMSWZL(wxz)
  NMSWZL(wxw)
  NMSWZL(xyx)
  NMSWZL(xyy)
  NMSWZL(xyz)
  NMSWZL(xyw)
  NMSWZL(yyx)
  NMSWZL(yyy)
  NMSWZL(yyz)
  NMSWZL(yyw)
  NMSWZL(zyx)
  NMSWZL(zyy)
  NMSWZL(zyz)
  NMSWZL(zyw)
  NMSWZL(wyx)
  NMSWZL(wyy)
  NMSWZL(wyz)
  NMSWZL(wyw)
  NMSWZL(xzx)
  NMSWZL(xzy)
  NMSWZL(xzz)
  NMSWZL(xzw)
  NMSWZL(yzx)
  NMSWZL(yzy)
  NMSWZL(yzz)
  NMSWZL(yzw)
  NMSWZL(zzx)
  NMSWZL(zzy)
  NMSWZL(zzz)
  NMSWZL(zzw)
  NMSWZL(wzx)
  NMSWZL(wzy)
  NMSWZL(wzz)
  NMSWZL(wzw)
  NMSWZL(xwx)
  NMSWZL(xwy)
  NMSWZL(xwz)
  NMSWZL(xww)
  NMSWZL(ywx)
  NMSWZL(ywy)
  NMSWZL(ywz)
  NMSWZL(yww)
  NMSWZL(zwx)
  NMSWZL(zwy)
  NMSWZL(zwz)
  NMSWZL(zww)
  NMSWZL(wwx)
  NMSWZL(wwy)
  NMSWZL(wwz)
  NMSWZL(www)
  NMSWZL(xxxx)
  NMSWZL(xxxy)
  NMSWZL(xxxz)
  NMSWZL(xxxw)
  NMSWZL(yxxx)
  NMSWZL(yxxy)
  NMSWZL(yxxz)
  NMSWZL(yxxw)
  NMSWZL(zxxx)
  NMSWZL(zxxy)
  NMSWZL(zxxz)
  NMSWZL(zxxw)
  NMSWZL(wxxx)
  NMSWZL(wxxy)
  NMSWZL(wxxz)
  NMSWZL(wxxw)
  NMSWZL(xyxx)
  NMSWZL(xyxy)
  NMSWZL(xyxz)
  NMSWZL(xyxw)
  NMSWZL(yyxx)
  NMSWZL(yyxy)
  NMSWZL(yyxz)
  NMSWZL(yyxw)
  NMSWZL(zyxx)
  NMSWZL(zyxy)
  NMSWZL(zyxz)
  NMSWZL(zyxw)
  NMSWZL(wyxx)
  NMSWZL(wyxy)
  NMSWZL(wyxz)
  NMSWZL(wyxw)
  NMSWZL(xzxx)
  NMSWZL(xzxy)
  NMSWZL(xzxz)
  NMSWZL(xzxw)
  NMSWZL(yzxx)
  NMSWZL(yzxy)
  NMSWZL(yzxz)
  NMSWZL(yzxw)
  NMSWZL(zzxx)
  NMSWZL(zzxy)
  NMSWZL(zzxz)
  NMSWZL(zzxw)
  NMSWZL(wzxx)
  NMSWZL(wzxy)
  NMSWZL(wzxz)
  NMSWZL(wzxw)
  NMSWZL(xwxx)
  NMSWZL(xwxy)
  NMSWZL(xwxz)
  NMSWZL(xwxw)
  NMSWZL(ywxx)
  NMSWZL(ywxy)
  NMSWZL(ywxz)
  NMSWZL(ywxw)
  NMSWZL(zwxx)
  NMSWZL(zwxy)
  NMSWZL(zwxz)
  NMSWZL(zwxw)
  NMSWZL(wwxx)
  NMSWZL(wwxy)
  NMSWZL(wwxz)
  NMSWZL(wwxw)
  NMSWZL(xxyx)
  NMSWZL(xxyy)
  NMSWZL(xxyz)
  NMSWZL(xxyw)
  NMSWZL(yxyx)
  NMSWZL(yxyy)
  NMSWZL(yxyz)
  NMSWZL(yxyw)
  NMSWZL(zxyx)
  NMSWZL(zxyy)
  NMSWZL(zxyz)
  NMSWZL(zxyw)
  NMSWZL(wxyx)
  NMSWZL(wxyy)
  NMSWZL(wxyz)
  NMSWZL(wxyw)
  NMSWZL(xyyx)
  NMSWZL(xyyy)
  NMSWZL(xyyz)
  NMSWZL(xyyw)
  NMSWZL(yyyx)
  NMSWZL(yyyy)
  NMSWZL(yyyz)
  NMSWZL(yyyw)
  NMSWZL(zyyx)
  NMSWZL(zyyy)
  NMSWZL(zyyz)
  NMSWZL(zyyw)
  NMSWZL(wyyx)
  NMSWZL(wyyy)
  NMSWZL(wyyz)
  NMSWZL(wyyw)
  NMSWZL(xzyx)
  NMSWZL(xzyy)
  NMSWZL(xzyz)
  NMSWZL(xzyw)
  NMSWZL(yzyx)
  NMSWZL(yzyy)
  NMSWZL(yzyz)
  NMSWZL(yzyw)
  NMSWZL(zzyx)
  NMSWZL(zzyy)
  NMSWZL(zzyz)
  NMSWZL(zzyw)
  NMSWZL(wzyx)
  NMSWZL(wzyy)
  NMSWZL(wzyz)
  NMSWZL(wzyw)
  NMSWZL(xwyx)
  NMSWZL(xwyy)
  NMSWZL(xwyz)
  NMSWZL(xwyw)
  NMSWZL(ywyx)
  NMSWZL(ywyy)
  NMSWZL(ywyz)
  NMSWZL(ywyw)
  NMSWZL(zwyx)
  NMSWZL(zwyy)
  NMSWZL(zwyz)
  NMSWZL(zwyw)
  NMSWZL(wwyx)
  NMSWZL(wwyy)
  NMSWZL(wwyz)
  NMSWZL(wwyw)
  NMSWZL(xxzx)
  NMSWZL(xxzy)
  NMSWZL(xxzz)
  NMSWZL(xxzw)
  NMSWZL(yxzx)
  NMSWZL(yxzy)
  NMSWZL(yxzz)
  NMSWZL(yxzw)
  NMSWZL(zxzx)
  NMSWZL(zxzy)
  NMSWZL(zxzz)
  NMSWZL(zxzw)
  NMSWZL(wxzx)
  NMSWZL(wxzy)
  NMSWZL(wxzz)
  NMSWZL(wxzw)
  NMSWZL(xyzx)
  NMSWZL(xyzy)
  NMSWZL(xyzz)
  NMSWZL(xyzw)
  NMSWZL(yyzx)
  NMSWZL(yyzy)
  NMSWZL(yyzz)
  NMSWZL(yyzw)
  NMSWZL(zyzx)
  NMSWZL(zyzy)
  NMSWZL(zyzz)
  NMSWZL(zyzw)
  NMSWZL(wyzx)
  NMSWZL(wyzy)
  NMSWZL(wyzz)
  NMSWZL(wyzw)
  NMSWZL(xzzx)
  NMSWZL(xzzy)
  NMSWZL(xzzz)
  NMSWZL(xzzw)
  NMSWZL(yzzx)
  NMSWZL(yzzy)
  NMSWZL(yzzz)
  NMSWZL(yzzw)
  NMSWZL(zzzx)
  NMSWZL(zzzy)
  NMSWZL(zzzz)
  NMSWZL(zzzw)
  NMSWZL(wzzx)
  NMSWZL(wzzy)
  NMSWZL(wzzz)
  NMSWZL(wzzw)
  NMSWZL(xwzx)
  NMSWZL(xwzy)
  NMSWZL(xwzz)
  NMSWZL(xwzw)
  NMSWZL(ywzx)
  NMSWZL(ywzy)
  NMSWZL(ywzz)
  NMSWZL(ywzw)
  NMSWZL(zwzx)
  NMSWZL(zwzy)
  NMSWZL(zwzz)
  NMSWZL(zwzw)
  NMSWZL(wwzx)
  NMSWZL(wwzy)
  NMSWZL(wwzz)
  NMSWZL(wwzw)
  NMSWZL(xxwx)
  NMSWZL(xxwy)
  NMSWZL(xxwz)
  NMSWZL(xxww)
  NMSWZL(yxwx)
  NMSWZL(yxwy)
  NMSWZL(yxwz)
  NMSWZL(yxww)
  NMSWZL(zxwx)
  NMSWZL(zxwy)
  NMSWZL(zxwz)
  NMSWZL(zxww)
  NMSWZL(wxwx)
  NMSWZL(wxwy)
  NMSWZL(wxwz)
  NMSWZL(wxww)
  NMSWZL(xywx)
  NMSWZL(xywy)
  NMSWZL(xywz)
  NMSWZL(xyww)
  NMSWZL(yywx)
  NMSWZL(yywy)
  NMSWZL(yywz)
  NMSWZL(yyww)
  NMSWZL(zywx)
  NMSWZL(zywy)
  NMSWZL(zywz)
  NMSWZL(zyww)
  NMSWZL(wywx)
  NMSWZL(wywy)
  NMSWZL(wywz)
  NMSWZL(wyww)
  NMSWZL(xzwx)
  NMSWZL(xzwy)
  NMSWZL(xzwz)
  NMSWZL(xzww)
  NMSWZL(yzwx)
  NMSWZL(yzwy)
  NMSWZL(yzwz)
  NMSWZL(yzww)
  NMSWZL(zzwx)
  NMSWZL(zzwy)
  NMSWZL(zzwz)
  NMSWZL(zzww)
  NMSWZL(wzwx)
  NMSWZL(wzwy)
  NMSWZL(wzwz)
  NMSWZL(wzww)
  NMSWZL(xwwx)
  NMSWZL(xwwy)
  NMSWZL(xwwz)
  NMSWZL(xwww)
  NMSWZL(ywwx)
  NMSWZL(ywwy)
  NMSWZL(ywwz)
  NMSWZL(ywww)
  NMSWZL(zwwx)
  NMSWZL(zwwy)
  NMSWZL(zwwz)
  NMSWZL(zwww)
  NMSWZL(wwwx)
  NMSWZL(wwwy)
  NMSWZL(wwwz)
  NMSWZL(wwww)

  /**
   * Access the i-th scalar value in the vector.
   */
  S& operator[](const int i) { return data[i]; }

  /**
   * Access the i-th scalar value in the vector as a constant expression.
   */
  constexpr const S& operator[](const int i) const { return data[i]; }
};

}

/**
 * Specialization of \ref vec for two-dimensional vectors.
 */
template <class S>
struct vec<S, 2> : public detail::vbase<2, S> {
  using BaseT = detail::vbase<2, S>;

  vec() = default;
  explicit constexpr vec(const S v) : BaseT{v, v} {}
  constexpr vec(const S x, const S y) : BaseT{x, y} {}

  constexpr S x() const { return this->data[0]; }
  constexpr S y() const { return this->data[1]; }
};

/**
 * A two-dimensional vector of 32-bit floating point values.
 */
using float2 = vec<float, 2>;

/**
 * Specialization of \ref vec for two-dimensional vectors.
 */
template <class S>
struct vec<S, 3> : public detail::vbase<3, S> {
  using BaseT = detail::vbase<3, S>;

  vec() = default;
  explicit constexpr vec(const S v) : BaseT{v, v, v} {}
  constexpr vec(const S x, const S y, const S z) : BaseT{x, y, z} {}
  constexpr vec(const vec<S, 2> &xy, const S z) : BaseT{xy[0], xy[1], z} {}
  constexpr vec(const S x, const vec<S, 2> &yz) : BaseT{x, yz[0], yz[1]} {} 

  constexpr S x() const { return this->data[0]; }
  constexpr S y() const { return this->data[1]; }
  constexpr S z() const { return this->data[2]; }
};

/**
 * A three-dimensional vector of 32-bit floating point values.
 */
using float3 = vec<float, 3>;

/**
 * Specialization of \ref vec for four-dimensional vectors.
 */
template <class S>
struct vec<S, 4> : public detail::vbase<4, S> {
  using BaseT = detail::vbase<4, S>;

  vec() = default;
  constexpr vec(const BaseT &b) : BaseT(b) {}
  explicit constexpr vec(const S v) : BaseT{v, v, v, v} {}
  constexpr vec(const S x, const S y, const S z, const S w) : BaseT{x, y, z, w} {}
  constexpr vec(const S x, const S y, const vec<S, 2> &zw) : BaseT{x, y, zw[0], zw[1]} {}
  constexpr vec(const S x, const vec<S, 3> &yzw) : BaseT{x, yzw[0], yzw[1], yzw[2]} {}
  constexpr vec(const S x, const vec<S, 2> &yz, const S w) : BaseT{x, yz[0], yz[1], w} {}
  constexpr vec(const vec<S, 2> &xy, const S z, const S w) : BaseT{xy[0], xy[1], z, w} {}
  constexpr vec(const vec<S, 2> &xy, const vec<S, 2> &zw) : BaseT{xy[0], xy[1], zw[0], zw[1]} {}
  constexpr vec(const vec<S, 3> &xyz, const S w) : BaseT{xyz[0], xyz[1], xyz[2], w} {}

  constexpr S x() const { return this->data[0]; }
  constexpr S y() const { return this->data[1]; }
  constexpr S z() const { return this->data[2]; }
  constexpr S w() const { return this->data[3]; }
};

/**
 * A four-dimensional vector of 32-bit floating point values.
 */
using float4 = vec<float, 4>;

/**
 * A square matrix. Elements are laid out in memory column-by-column in a
 * contiguous manner. For example, for a 2x2 matrix, the memory layout would
 * be: [m00 m01 m10 m11] where [m00 m01] is the first column, and [m10 m11] is
 * the second column.
 */
template <class S, unsigned N>
struct mat {
  /** Underlying column type. */
  using ColumnT = vec<S, N>;

  /** Number of rows/columns. */
  static constexpr int Size = N;

  ColumnT column[Size];

  constexpr mat(){};

  /**
   * Create a new matrix from the given column vectors.
   *
   * Example usage:
   * `float2x2::from_columns(float2 {.0f, .0f}, float2 {.1f, 1.f})`
   */
  template<class... Args>
  static constexpr auto from_columns(const Args&... cols) {
    static_assert(sizeof...(cols) == N, "Wrong number of columns given.");
    return mat { cols... };
  }

  /**
   * Create a new matrix from the given row vectors.
   *
   * Example usage:
   * `float2x2::from_rows(float2 {.0f, .0f}, float2 {.1f, 1.f})`
   */
  template<class... Args>
  static constexpr mat from_rows(const Args&... rows) {
    static_assert(sizeof...(rows) == Size, "Wrong number of rows given.");
    mat result;
    for (unsigned i = 0u; i < N; ++i) {
      result.column[i] = ColumnT { rows.data[i]... };
    }
    return result;
  }

  /**
   * Create a new identity matrix (i.e. with all coefficients set to 0, except
   * the main diagonal coefficients, which are set to 1).
   */
  static constexpr mat identity() {
    mat result;
    for (unsigned i = 0u; i < N; ++i)
      for (unsigned j = 0u; j < N; ++j)
        result.column[i].data[j] = (S)(i == j ? 1.0f : 0.0f);
    return result;
  }

  /**
   * Access the i-th column of the matrix.
   */
  constexpr ColumnT& operator[](const unsigned i) { return column[i]; }

  /**
   * Access the i-th column of the matrix (read-only).
   */
  constexpr const ColumnT& operator[](const int i) const { return column[i]; }

private:
  template<class... Args>
  constexpr mat(const Args&... cols) : column { cols... } {
    static_assert(sizeof...(cols) == N, "Wrong number of columns given.");
  }
};


/**
 * 2x2 matrix.
 */
template <class S>
using mat2x2 = mat<S, 2>;

/**
 * A 2x2 matrix of 32-bit floating point coefficients.
 */
using float2x2 = mat2x2<float>;

/**
 * 3x3 matrix.
 */
template <class S>
using mat3x3 = mat<S, 3>;

/**
 * A 3x3 matrix of 32-bit floating point coefficients.
 */
using float3x3 = mat3x3<float>;

/**
 * 4x4 matrix.
 */
template <class S>
using mat4x4 = mat<S, 4>;

/**
 * A 4x4 matrix of 32-bit floating point coefficients.
 */
using float4x4 = mat4x4<float>;

/**
 * A quaternion.
 */
template <class S>
struct quat : public vec<S, 4> {
  quat() = default;

  /**
   * Constructs a unit quaternion representing a rotation by the given amount
   * around an axis.
   */
  constexpr quat(const S theta, const vec<S, 3> &axis) : 
      vec<S, 4>(std::sin(theta / (S)2.0) * normalize(axis),
                cos(theta / (S) 2.0)) {}

  /**
   * Explicitly construct a quaternion from components.
   */
   constexpr quat(const S x, const S y, const S z, const S w) :
      vec<S, 4>(x, y, z, w) {}
};

/**
 * A quaternion with 32-bit floating point coefficients.
 */
using quatf = quat<float>;

/**
 * @return The conjugate of a given quaternion.
 */
template <class S>
inline constexpr quat<S> conjugate(const quat<S> &q) {
  return quat<S> { -q.data[0], -q.data[1], -q.data[2], q.data[3] };
}

/**
 * @return The product of two quaternions.
 */
template <class S>
inline constexpr quat<S> operator*(const quat<S> &lhs, const quat<S> &rhs) {
  const S x1 = lhs.data[0],
          x2 = rhs.data[0],
          y1 = lhs.data[1],
          y2 = rhs.data[1],
          z1 = lhs.data[2],
          z2 = rhs.data[2],
          w1 = lhs.data[3],
          w2 = rhs.data[3];
  return quat<S> {
    x1 * w1 + y1 * z2 - z1 * y2 + x2 * w1,
    y1 * w2 - x1 * z2 + z1 * x2 + y2 * w1,
    x1 * y2 - y1 * x2 + z1 * w2 + z2 * w1,
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
  };
}

/**
 * Rotates the given vector using the given unit quaternion.
 * @param vv the vector to rotate
 * @param q  the quaternion representing the rotation.
 * @return rotated vv
 */
template <class S>
inline constexpr vec<S, 3> rotate(const vec<S, 3> &vv, const quat<S> &q) {
  const quat<S> v { v.x, v.y, v.z, (S)0.0 };
  return q * v * q.conjugate();
}

template<class S>
auto quat2mat(const quat<S> &q) {
  return mat4x4<S> {
    { 
      1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()),
      2.0 *(q.x() * q.y() - q.z() * q.w()),
      2.0 * (q.x() * q.z() + q.y() * q.w()), 
      0.0
    },
    {
      2.0 * (q.x() * q.y() + q.z() * q.w()),
      1.0 - 2.0 * (q.x() * q.x() + q.z() * q.z()),
      2.0 * (q.y() * q.z() + q.x() * q.w()), 
      0.0
    },
    {
      2.0 * (q.x() * q.z() + q.y() * q.w()),
      2.0 * (q.y() * q.z() - q.x() * q.w()),
      1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y()),
      0.0
    },
    { 0.0, 0.0, 0.0, 1.0 }
  };
}

/**
 * @return The cross product of two three-dimensional vectors.
 */
template <class S>
inline constexpr auto cross(const vec<S, 3> &lhs, const vec<S, 3> &rhs) {
  return vec<S, 3>(lhs.data[1] * rhs.data[2] - lhs.data[2] * rhs.data[1],
                   lhs.data[2] * rhs.data[0] - lhs.data[0] * rhs.data[2],
                   lhs.data[0] * rhs.data[1] - lhs.data[1] * rhs.data[0]);
}

template<>
inline constexpr auto cross(const vec<float, 3> &lhs, const vec<float, 3> &rhs) {
	const vec<double, 3> a { (double)lhs.x(), (double)lhs.y(), (double)lhs.z() },
                       b { (double)rhs.x(), (double)rhs.y(), (double)rhs.z() },
                       c = cross(a, b);
    return vec<float, 3> { (float)c.x(), (float)c.y(), (float)c.z() };
}


/**
 * @return The dot product of two vectors.
 */
template <unsigned N, class S>
inline constexpr S dot(const vec<S, N> &lhs, const vec<S, N> &rhs) {
  S result = (S)0.0f;
  for (unsigned i = 0u; i < N; ++i) result += lhs.data[i] * rhs.data[i];
  return result;
}

/**
 * @return The squared magnitude of a vector.
 */
template <unsigned N, class S>
inline constexpr S lengthsq(const vec<S, N> &v) { return dot(v, v); }

/**
 * @return The magnitude of a vector.
 */
template <unsigned N, class S>
inline constexpr S length(const vec<S, N> &v) { 
  return (S)std::sqrt(lengthsq(v));
}

/**
 * @return The projection of vector `a` onto `b`.
 */
template <class V>
inline constexpr V project(const V &a, const  V &b) {
  return b * (dot(a, b) / lengthsq(b));
}

/**
 * @return The part of `a` that is orthogonal to `b`, in other words,
 *         `a - project(a, b)`
 */
template <class V>
inline constexpr V reject(const V &a, const  V &b) { return a - project(a, b); }

/**
 * Reflect a vector against a plane.
 *
 * @param d The vector being reflected.
 * @param n The normal of the reflection plane. This is expected to be of
 *          magnitude 1.
 * @return Reflection of vector `d`.
 */
template <class V>
inline constexpr V reflect(const V &d, const V &n) {
  return d - n * (static_cast<typename V::Scalar>(2.0f) * dot(d, n));
}

/**
 * Vector normalization.
 * @param v the vector to normalize.
 * @return Vector with the same direction as `v` and a magnitude of 1.
 */
template <class V>
inline constexpr V normalize(const V &v) { return v / length(v); }

/**
 * @return The sum of two vectors.
 */
template <class S, unsigned N>
inline constexpr vec<S, N> operator+(const vec<S, N> &lhs,
                                     const vec<S, N> &rhs) {
  vec<S, N> result {};
  for (unsigned i = 0; i < N; ++i)
    result.data[i] = lhs.data[i] + rhs.data[i];
  return result;
}

/**
 * @return The difference of two vectors.
 */
template <class S, unsigned N>
inline constexpr vec<S, N> operator-(const vec<S, N> &lhs,
                                     const vec<S, N> &rhs) {
  vec<S, N> result {};
  for (unsigned i = 0; i < N; ++i)
    result.data[i] = lhs.data[i] - rhs.data[i];
  return result;
}

/**
 * @return A vector in which each element is the result of multiplying the
 *         corresponding elements of lhs and rhs.
 */
template <class S, unsigned N>
inline constexpr vec<S, N> operator*(const vec<S, N> &lhs, const vec<S, N> &rhs) {
  vec<S, N> result {};
  for (unsigned i = 0; i < N; ++i)
    result.data[i] = lhs.data[i] * rhs.data[i];
  return result;
}

/**
 * @return A vector in which each element is the result of dividing the
 *         corresponding element of lhs by the corresponding element of rhs.
 */
template <class S, unsigned N>
inline constexpr vec<S, N> operator/(const vec<S, N> &lhs, const vec<S, N> &rhs) {
  vec<S, N> result {};
  for (unsigned i = 0; i < N; ++i)
    result.data[i] = lhs.data[i] / rhs.data[i];
  return result;
}

/**
 * @return The given vector multiplied by the given scalar.
 */
template <class S, unsigned N>
inline constexpr vec<S, N> operator*(const vec<S, N> &lhs, const S rhs) {
  vec<S, N> result = lhs;
  for (unsigned i = 0; i < N; ++i) result.data[i] *= rhs;
  return result;
}

template <class S, unsigned N>
inline constexpr vec<S, N> operator*(const S lhs, const vec<S, N> &rhs) {
  return rhs * lhs;
}

/**
 * @return The given vector divided by the given scalar.
 */
template <class S, unsigned N>
inline constexpr vec<S, N> operator/(const vec<S, N> &lhs, const S rhs) {
  const S inv = (S)1.0 / rhs;
  return lhs * inv;
}

/**
 * Multiply the given vector by the given scalar.
 */
template <class S, unsigned N>
inline vec<S, N>& operator*=(vec<S, N> &v, const S s) {
  v = v * s;
  return v;
}

/**
 * Divide the given vector by the given scalar.
 */
template <class S, unsigned N>
inline vec<S, N>& operator/=(vec<S, N> &v, const S s) {
  v = v / s;
  return v;
}

/**
 * Vector negation.
 * @param v vector to negate
 * @return a vector pointing in the direction opposite of `v`, with the same 
 *         magnitude as `v`.
 */
template<class S, unsigned N>
inline vec<S, N> operator-(const vec<S, N> &v) {
  vec<S, N> result;
  for (unsigned i = 0u; i < N; ++i) result.data[i] = -v.data[i];
  return result;
}

/**
 * @return true if the given two vectors are strictly equal.
 */
template <class S, unsigned N>
inline constexpr bool operator==(const vec<S, N> &lhs, const vec<S, N> &rhs)  {
  for (unsigned i = 0; i < N; ++i)
    if(lhs.data[i] != rhs.data[i])
      return false;
  return true;
}

template <class S, unsigned N>
inline constexpr bool operator!=(const vec<S, N> &lhs, const vec<S, N> &rhs) {
  return !(lhs == rhs);
}

/**
 * @return transpose of `src`.
 */
template <class S, unsigned N>
inline constexpr mat<S, N> transpose(const mat<S, N> &src) {
  mat<S, N> result;
  for (unsigned i = 0u; i < N; ++i)
    for (unsigned j = 0u; j < N; ++j)
      result.column[i].data[j] = src.column[j].data[i];
  return result;
}

/**
 * @return Determinant of the given matrix.
 */
template<unsigned N, class S>
S det(const mat<S, N> &mat) = delete;

template <class S>
constexpr S det(const mat<S, 2> &mat) {
  return mat.column[0].data[0] * mat.column[1].data[1] 
       - mat.column[1].data[0] * mat.column[0].data[1]; 
}

template <class S>
constexpr S det(const mat<S, 3> &mat) {
  S m00 = mat.column[0].data[0];
  S m01 = mat.column[0].data[1];
  S m02 = mat.column[0].data[2];
  S m10 = mat.column[1].data[0];
  S m11 = mat.column[1].data[1];
  S m12 = mat.column[1].data[2];
  S m20 = mat.column[2].data[0];
  S m21 = mat.column[2].data[1];
  S m22 = mat.column[2].data[2];

  return m00 * (m11 * m22 - m21 * m12) 
       - m10 * (m01 * m22 - m21 * m02) 
       + m20 * (m01 * m12 - m11 * m02);
}

template <class S>
constexpr S det(const mat<S, 4> &mat) {
  S m00 = mat.column[0].data[0];
  S m01 = mat.column[0].data[1];
  S m02 = mat.column[0].data[2];
  S m03 = mat.column[0].data[3];
  S m10 = mat.column[1].data[0];
  S m11 = mat.column[1].data[1];
  S m12 = mat.column[1].data[2];
  S m13 = mat.column[1].data[3];
  S m20 = mat.column[2].data[0];
  S m21 = mat.column[2].data[1];
  S m22 = mat.column[2].data[2];
  S m23 = mat.column[2].data[3];
  S m30 = mat.column[3].data[0];
  S m31 = mat.column[3].data[1];
  S m32 = mat.column[3].data[2];
  S m33 = mat.column[3].data[3];
  return m30 * m21 * m12 * m03 - m20 * m31 * m12 * m03 -
         m30 * m11 * m22 * m03 + m10 * m31 * m22 * m03 +
         m20 * m11 * m32 * m03 - m10 * m21 * m32 * m03 -
         m30 * m21 * m02 * m13 + m20 * m31 * m02 * m13 +
         m30 * m01 * m22 * m13 - m00 * m31 * m22 * m13 -
         m20 * m01 * m32 * m13 + m00 * m21 * m32 * m13 +
         m30 * m11 * m02 * m23 - m10 * m31 * m02 * m23 -
         m30 * m01 * m12 * m23 + m00 * m31 * m12 * m23 +
         m10 * m01 * m32 * m23 - m00 * m11 * m32 * m23 -
         m20 * m11 * m02 * m33 + m10 * m21 * m02 * m33 +
         m20 * m01 * m12 * m33 - m00 * m21 * m12 * m33 -
         m10 * m01 * m22 * m33 + m00 * m11 * m22 * m33;
}

/**
 * @return Inverse of a given 2x2 matrix.
 */
template <class S>
constexpr mat<S, 2> inverse(const mat<S, 2> &m) {
  using V = typename mat<S, 2>::ColumnT;
  const S d = det(m);
  constexpr S _1 = 1.0;
  return mat<S, 2>::from_columns( 
      V {  m.column[1].data[1], -m.column[0].data[1] },
      V { -m.column[1].data[0],  m.column[0].data[0] } ) * (_1/d);
}

/**
 * @return Inverse for a given 3x3 matrix.
 */
template <class S>
constexpr mat<S, 3> inverse(const mat<S, 3> &m) {
  using V = typename mat<S, 3>::ColumnT;
  const S inv_d = (S)1.0 / det(m);
  const V &a = m.column[0],
          &b = m.column[1],
          &c = m.column[2];
  const V  r0 = cross(b, c),
           r1 = cross(c, a),
           r2 = cross(a, b);
  return mat<S, 3>::from_columns(V { r0.data[0], r1.data[0], r2.data[0] },
                                 V { r0.data[1], r1.data[1], r2.data[1] },
                                 V { r0.data[2], r1.data[2], r2.data[2] }) * inv_d;
}

/**
 * @return Inverse for a given 4x4 matrix.
 */
template <class S>
inline constexpr mat<S, 4> inverse(const mat<S, 4> &m) {
  const vec<S, 3> a { m.column[0].data[0], m.column[0].data[1], m.column[0].data[2] },
                  b { m.column[1].data[0], m.column[1].data[1], m.column[1].data[2] },
                  c { m.column[2].data[0], m.column[2].data[1], m.column[2].data[2] },
                  d { m.column[3].data[0], m.column[3].data[1], m.column[3].data[2] };
  const S x = m.column[0].data[3],
          y = m.column[1].data[3],
          z = m.column[2].data[3],
          w = m.column[3].data[3];
  vec<S, 3> s = cross(a, b),
            t = cross(c, d),
            u = a * y - b * x,
            v = c * w - d * z;
  const S inv_d = (S)1.0 / (dot(s, v) + dot(t, u));
  s *= inv_d; t *= inv_d; u *= inv_d; v *= inv_d;
  const vec<S, 3> r0 = cross(b, v) + t * y,
                  r1 = cross(v, a) - t * x,
                  r2 = cross(d, u) + s * w,
                  r3 = cross(u, c) - s * z;
  
  using V = typename mat<S, 4>::ColumnT;
  return mat<S, 4>::from_columns(
    V { r0.data[0], r1.data[0], r2.data[0], r3.data[0] },
    V { r0.data[1], r1.data[1], r2.data[1], r3.data[1] },
    V { r0.data[2], r1.data[2], r2.data[2], r3.data[2] },
    V {-dot(b, t), dot(a, t),   -dot(d, s), dot(c, s)  });
}

/**
 * @return A matrix with elements on the main diagonal set to the values from
 *         the given vector, and all the other elements set to 0.
 */
template <class S, unsigned N>
inline constexpr mat<S, N> scale(const vec<S, N> &factors) {
  auto result = mat<S, N>::identity();
  for (unsigned r = 0u; r < N; ++r) {
    result.column[r].data[r] = factors.data[r];
  }
  return result;
}

/**
 * @param theta Angle of rotation, in radians.
 * @return A 2x2 matrix representing rotation by a given angle around the origin
 *         in two dimensions.
 */
template <class S>
inline constexpr auto rotation(const S theta) {
  return mat<S, 2>::from_columns(vec<S, 2> {  std::cos(theta), std::sin(theta) },
                                  vec<S, 2> { -std::sin(theta), std::cos(theta) });
}

/**
 * @param theta Angle of rotation in radians.
 * @param axis Axis of rotation, which is assumed to be of unit magnitude.
 * @return A 3x3 matrix representing a rotation by a given angle around a given
 *         axis in three dimensions.
 */
template <class S>
inline constexpr auto rotation(const S theta, const vec<S, 3> &axis) {
  constexpr S _1 = (S)1.0;
  const     S  c  = std::cos(theta),
               s  = std::sin(theta),
               ax = axis.data[0],
               ay = axis.data[1],
               az = axis.data[2];

  return mat<S, 3>::from_columns(
    vec<S, 3> {
      c + (_1 - c) * ax * ax,
      (_1 - c) * ax * ay + s * az, 
      (_1 - c) * ax * az - s * ay

    },
    vec<S, 3> {
      (_1 - c) * ax * ay - s * az,
      c + (_1 - c) * ay * ay,
      (_1 - c) * ay * az + s * ax
    },
    vec<S, 3> {
      (_1 - c) * ax * az + s * ay,
      (_1 - c) * ay * az - s * ax,
      c + (_1 - c) * az * az
    });
}

/**
 * @param theta Angle of rotation, in radians.
 * @param axis Vector representing the axis of rotation. It is assumed to be a
 *             unit vector.
 * @return A 4x4 matrix representing a three-dimensional rotation by a given
 *         angle around a given axis.
 */
template <class S>
inline constexpr auto rotation(const S theta, const vec<S, 4> &axis) {
  constexpr S _1 = (S)1.0, 
              _0 = (S)0.0;
  const     S  c  = std::cos(theta),
               s  = std::sin(theta),
               ax = axis.data[0], 
               ay = axis.data[1],
               az = axis.data[2];

  return mat<S, 4>::from_columns(
    vec<S, 4> {
      c + (_1 - c) * ax * ax,
      (_1 - c) * ax * ay + s * az, 
      (_1 - c) * ax * az - s * ay,
      _0
    },
    vec<S, 4> {
      (_1 - c) * ax * ay - s * az,
      c + (_1 - c) * ay * ay,
      (_1 - c) * ay * az + s * ax,
      _0
    },
    vec<S, 4> {
      (_1 - c) * ax * az + s * ay,
      (_1 - c) * ay * az - s * ax,
      c + (_1 - c) * az * az,
      _0
    },
    vec<S, 4> { _0, _0, _0, _1 });
}

/**
 * @param theta Angle of rotation, in radians.
 * @return A 4x4 matrix representing a three-dimensional rotation by a given
 *         angle around the X axis.
 */
template <class S>
inline constexpr auto rotation_x(const S theta) {
  const S s = std::sin(theta),
          c = std::cos(theta),
         _0 = (S)0.0,
         _1 = (S)1.0;
  return mat<S, 4>::from_columns(
    vec<S, 4> { _1, _0, _0, _0 },
    vec<S, 4> { _0,  c,  s, _0 },
    vec<S, 4> { _0, -s,  c, _0 },
    vec<S, 4> { _0, _0, _0, _1 });
}

/**
 * @param theta Angle of rotation, in radians.
 * @return A 4x4 matrix representing a three-dimensional rotation by a given
 *         angle around the Y axis.
 */
template <class S>
inline constexpr auto rotation_y(const S theta) {
  const S s = std::sin(theta),
          c = std::cos(theta),
         _0 = (S)0.0,
         _1 = (S)1.0;
  return mat<S, 4>::from_columns(
    vec<S, 4> {  c, _0, -s, _0 },
    vec<S, 4> { _0, _1, _0, _0 },
    vec<S, 4> {  s, _0,  c, _0 },
    vec<S, 4> { _0, _0, _0, _1 });
}

/**
 * @param theta Angle of rotation, in radians.
 * @return A 4x4 matrix representing a three-dimensional rotation by a given
 *         angle around the Z axis.
 */
template <class S>
inline constexpr auto rotation_z(const S theta) {
  const S s = std::sin(theta),
          c = std::cos(theta),
         _0 = (S)0.0,
         _1 = (S)1.0;
  return mat<S, 4>::from_columns(
    vec<S, 4> {  c,  s, _0, _0 },
    vec<S, 4> { -s,  c, _0, _0 },
    vec<S, 4> { _0, _0, _1, _0 },
    vec<S, 4> { _0, _0, _0, _1 });
}

/**
 * @return A 3x3 matrix representing translation in two dimensions.
 */
template <class S>
inline constexpr auto translation(const vec<S, 2> &offset) {
  constexpr S _0 = (S)0.0, _1 = (S)1.0;
  return mat<S, 3>::from_columns(
    vec<S, 3> { _1, _0, _0 },
    vec<S, 3> { _0, _1, _0 },
    vec<S, 3> { offset, _1 });
}

/**
 * @return A 4x4 matrix representing a translation in three dimensions.
 */
template <class S>
inline constexpr auto translation(const vec<S, 3> &offset) {
  const S _0 = (S)0.0, _1 = (S)1.0;
  return mat<S, 4>::from_columns(
    vec<S, 4> { _1, _0, _0, _0 },
    vec<S, 4> { _0, _1, _0, _0 },
    vec<S, 4> { _0, _0, _1, _0 },
    vec<S, 4> { offset, _1 });
}

/**
 * @param eye The location of the camera.
 * @param target The location the camera is pointed at.
 * @param up A vector pointing towards the top of the camera.
 * @returns A matrix representing the view transform corresponding to the camera
 *          setup described by the above three parameters.
 */
template <class S>
inline constexpr mat<S, 4> look_at(const vec<S, 3> &eye,
                                   const vec<S, 3> &target,
                                   const vec<S, 3> &up) {
  constexpr S _0 = (S)0.0, _1 = (S)1.0;
  const vec<S, 3> z = -normalize(target - eye),
                  x =  normalize(cross(up, z)),
                  y =  cross(z, x);
  return mat<S, 4>::from_rows(
    vec<S, 4> { x, -dot(eye, x) },
    vec<S, 4> { y, -dot(eye, y) },
    vec<S, 4> { z, -dot(eye, z) },
    vec<S, 4> { _0, _0, _0, _1 });
}

/**
 * @param l Left boundary.
 * @param r Right boundary.
 * @param b Bottom boundary.
 * @param t Top boundary.
 * @param n Near boundary.
 * @param f Far boundary.
 * @return An orthographic projection matrix.
 */
template <class S>
inline constexpr auto ortho(const S l, const S r,
                            const S b, const S t,
                            const S n, const S f) {
  constexpr S _2 = (S)2.0, _0 = (S)0.0, _1 = (S)1.0;
  return mat<S, 4>::from_columns(
    vec<S, 4> { _2 / (r - l), _0, _0, _0 },
    vec<S, 4> { _0, _2 / (t - b), _0, _0 },
    vec<S, 4> { _0, _0, _2 / (f - n), _0, },
    vec<S, 4> { -(r + l) / (r - l), -(t + b) / (t - b), -(n + f) / (f - n), _1 });
}

/**
 * @param l Left boundary.
 * @param r Right boundary.
 * @param b Bottom boundary.
 * @param t Top boundary.
 * @param ndist Near boundary.
 * @param fdist Far boundary.
 * @return A perspective projection matrix described by the above parameters.
 */
template <class S>
inline constexpr auto perspective(const S l, const S r,
                                  const S b, const S t,
                                  const S ndist, const S fdist) {
  constexpr S _0 = (S)0.0, _1 = (S)1.0, _2 = (S)2.0;
  return mat<S, 4>::from_columns(
    vec<S, 4> { _2 * ndist / (r - l), _0, _0, _0 },
    vec<S, 4> { _0, _2 * ndist / (t - b), _0, _0 },
    vec<S, 4> { (r + l) /(r - l), (t + b)/(t - b), -(ndist + fdist)/(fdist - ndist), -_1},
    vec<S, 4> { _0, _0, -_2 * ndist * fdist / (fdist - ndist), _0 });
}

/**
 * @param fovy Vertical field-of-view angle, in radians.
 * @param aspect Aspect ratio.
 * @param ndist Near boundary.
 * @param fdist Far boundary.
 * @return A perspective projection matrix described by the above parameters.
 */ 
template <class S>
inline constexpr auto perspective(const S fovy, const S aspect,
                                  const S ndist, const S fdist) {
  constexpr S _0 = (S)0.0, _1 = (S)1.0, _2 = (S)2.0;
  const S t = _1 / std::tan(fovy / _2);
  return mat<S, 4>::from_columns(
    vec<S, 4> { t / aspect, _0, _0, _0},
    vec<S, 4> { _0, t, _0, _0 },
    vec<S, 4> { _0, _0, -(fdist + ndist)/(fdist - ndist), -_1 },
    vec<S, 4> { _0, _0, -_2 * ndist * fdist / (fdist - ndist), _0 });
}

/**
 * (Matrix) X (Vector) multiplication.
 */
template<class S, unsigned N>
constexpr vec<S, N> operator*(const mat<S, N> &lhs, const vec<S, N> &rhs) {
  vec<S, N> result { (S)0.0 };
  for (unsigned c = 0; c < N; ++c) {
    for (unsigned i = 0; i < N; ++i)
      result.data[i] += lhs.column[c].data[i] * rhs.data[c];
  }
  return result;
}

/**
 * (Matrix) X (Matrix) multiplication.
 */
template<class S, unsigned N>
constexpr mat<S, N> operator*(const mat<S, N> &lhs, const mat<S, N> &rhs) {
  mat<S, N> result;
  for (unsigned c = 0; c < N; ++c) {
    for (unsigned r = 0; r < N; ++r) {
      result.column[c].data[r] = 0.0f;
      for (unsigned i = 0; i < N; ++i) {
        result.column[c].data[r] +=
          (lhs.column[i].data[r]) * (rhs.column[c].data[i]);
      }
    }
  }
  return result;
}

/**
 * Strict equality comparison for matrices.
 */
template <class S, unsigned N>
constexpr bool operator==(const mat<S, N> &lhs, const mat<S, N> &other) {
  for (unsigned i = 0u; i < N; ++i)
    if (lhs.column[i] != other.column[i])
      return false;
  return true;
}

/**
 * Multiplies all matrix elements by a given scalar value.
 */
template <class S, unsigned N>
constexpr mat<S, N> operator*(const mat<S, N> &lhs, const S rhs) {
  mat<S, N> result {};
  for (unsigned i = 0; i < N; ++i) result.column[i] = lhs.column[i] * rhs;
  return result;
}

template <class S, unsigned N>
constexpr mat<S, N> operator*(const S lhs, const mat<S, N> &rhs) {
  return rhs * lhs;
}

/**
 * Divides all matrix elements by a given scalar value.
 */
template <class S, unsigned N>
constexpr mat<S, N> operator/(const mat<S, N> &lhs, const S rhs) {
  return lhs * ((S)1.0f / rhs);
}

template <class S, unsigned N>
constexpr mat<S, N> operator/(const S lhs, const mat<S, N> &rhs) {
  return rhs / lhs;
}

/**
 * Degree-to-radian conversion.
 * @param deg angle in degrees.
 * @return angle in radians.
 */
template <class S>
inline constexpr S deg2rad(const S deg) { return deg *  ((S)PI/(S)180.0); }

}
