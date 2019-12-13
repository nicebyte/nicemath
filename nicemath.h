#pragma once

#include <cmath>

/**
 * nicemath is a compact single-header C++ library that provides data types and
 * routines for basic linear algebra operations often encountered in computer
 * graphics and game development.
 */
namespace nm {

constexpr float PI = 3.1415926f;

namespace detail {

template <unsigned N, class S>
struct vbase {
  static constexpr unsigned Dimensions = N;
  using Scalar = S;

  S data[N];

  constexpr vbase() = default;

  template <class... Args>
  constexpr vbase(Args... args) : data{ args... } {
    static_assert(sizeof...(args) == N, "Wrong number of initializers.");
  }

  S& operator[](const int i) { return data[i]; }
  constexpr const S& operator[](const int i) const { return data[i]; }
};

}

template <class S, unsigned N>
struct vec;

template <class S>
struct vec<S, 2> : public detail::vbase<2, S> {
  using BaseT = detail::vbase<2, S>;

  vec() = default;
  explicit constexpr vec(const S v) : BaseT{v, v} {}
  constexpr vec(const S x, const S y) : BaseT{x, y} {}

  constexpr S x() const { return this->data[0]; }
  constexpr S y() const { return this->data[1]; }
};
using float2 = vec<float, 2>;

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
using float3 = vec<float, 3>;

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
using float4 = vec<float, 4>;

/**
 * A square matrix. Elements are laid out in memory column-by-column in a
 * contiguous manner. For example, for a 2x2 matrix, the memory layout would
 * be: [m00 m01 m10 m11] where [m00 m01] is the first column, and [m10 m11] is
 * the second column.
 */
template <class S, unsigned N>
struct mat {
  using ColumnT = vec<S, N>;
  static constexpr int Size = N;

  ColumnT column[Size];

  constexpr mat(){};

  /**
   * Create a new matrix from given column vectors.
   */
  template<class... Args>
  static constexpr auto from_columns(const Args&... cols) {
    static_assert(sizeof...(cols) == N, "Wrong number of columns given.");
    return mat { cols... };
  }

  /**
   * Create a new matrix from given row vectors.
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
   * Create a new identity matrix (i.e. with all elements set to 0, except the
   * main diagonal elements, which are set to 1).
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
  ColumnT& operator[](const unsigned i) { return column[i]; }

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
using float2x2 = mat2x2<float>;

/**
 * 3x3 matrix.
 */
template <class S>
using mat3x3 = mat<S, 3>;
using float3x3 = mat3x3<float>;

/**
 * 4x4 matrix.
 */
template <class S>
using mat4x4 = mat<S, 4>;
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
 */
template <class S>
inline constexpr vec<S, 3> rotate(const vec<S, 3> &vv, const quat<S> &q) {
  const quat<S> v { v.x, v.y, v.z, (S)0.0 };
  return q * v * q.conjugate();
}

// TODO: quat to mat

/**
 * @return The cross product of two three-dimensional vectors.
 */
template <class S>
inline constexpr auto cross(const vec<S, 3> &lhs, const vec<S, 3> &rhs) {
  return vec<S, 3>(lhs.data[1] * rhs.data[2] - lhs.data[2] * rhs.data[1],
                   lhs.data[2] * rhs.data[0] - lhs.data[0] * rhs.data[2],
                   lhs.data[0] * rhs.data[1] - lhs.data[1] * rhs.data[0]);
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
 * @return Projection of `a` onto `b`.
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
 * @return Vector with the same direction as v and a magnitude of 1.
 */
template <class V>
inline constexpr V normalize(const V &v) { return v / length(v); }

/**
 * @return The sum of two vectors.
 */
template <class S, unsigned N>
inline constexpr vec<S, N> operator+(const vec<S, N> &lhs,
                                     const vec<S, N> &rhs) {
  vec<S, N> result;
  for (int i = 0; i < N; ++i)
    result.data[i] = lhs.data[i] + rhs.data[i];
  return result;
}

/**
 * @return The difference of two vectors.
 */
template <class S, unsigned N>
inline constexpr vec<S, N> operator-(const vec<S, N> &lhs,
                                     const vec<S, N> &rhs) {
  vec<S, N> result;
  for (int i = 0; i < N; ++i)
    result.data[i] = lhs.data[i] - rhs.data[i];
  return result;
}

/**
 * @return A vector in which each element is the result of multiplying the
 *         corresponding elements of lhs and rhs.
 */
template <class S, unsigned N>
inline constexpr vec<S, N> operator*(const vec<S, N> &lhs, const vec<S, N> &rhs) {
  vec<S, N> result;
  for (int i = 0; i < N; ++i)
    result.data[i] = lhs.data[i] * rhs.data[i];
  return result;
}

/**
 * @return A vector in which each element is the result of dividing the
 *         corresponding element of lhs by the corresponding element of rhs.
 */
template <class S, unsigned N>
inline constexpr vec<S, N> operator/(const vec<S, N> &lhs, const vec<S, N> &rhs) {
  vec<S, N> result;
  for (int i = 0; i < N; ++i)
    result.data[i] = lhs.data[i] / rhs.data[i];
  return result;
}

/**
 * @return The given vector multiplied by the given scalar.
 */
template <class S, unsigned N>
inline constexpr vec<S, N> operator*(const vec<S, N> &lhs, const S rhs) {
  vec<S, N> result = lhs;
  for (int i = 0; i < N; ++i) result.data[i] *= rhs;
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
 * Multiply the givven vector by the given scalar.
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
  for (int i = 0; i < N; ++i)
    if(lhs.data[i] != rhs.data[i])
      return false;
  return true;
}

template <class S, unsigned N>
inline constexpr bool operator!=(const vec<S, N> &lhs, const vec<S, N> &rhs) {
  return !(lhs == rhs);
}

/**
 * @return transpose of `m`.
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
 * @return Inverse for agiven 4x4 matrix.
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
  for (int c = 0; c < N; ++c) {
    for (int r = 0; r < N; ++r) {
      result.column[c].data[r] = 0.0f;
      for (int i = 0; i < N; ++i) {
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
  mat<S, N> result = lhs;
  for (typename mat<S, N>::ColumnT &col : result.column) col *= rhs;
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

template <class S>
inline constexpr S deg2rad(const S deg) { return deg *  ((S)PI/(S)180.0); }

}
