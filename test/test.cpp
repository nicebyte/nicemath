#include "../nicemath.h"
#include "matvecio.h"
#define NT_TEST_IMPL
#include "nicetest.h"

#include <chrono>
#include <cmath>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <limits>
#include <tuple>

bool flt_eq(float a, float b) {
  NTASSERT(!std::isnan(a) && !std::isnan(b));
  constexpr float max_input_mag = 100.f;
  constexpr float abs_eps = max_input_mag * std::numeric_limits<float>::epsilon();
  if(fabs(a - b) <= abs_eps) return true;
  uint32_t a_bits, b_bits;
  memcpy(&a_bits, &a, sizeof(float));
  memcpy(&b_bits, &b, sizeof(float));
  const uint32_t ulp_dist = ((a_bits > b_bits) ? (a_bits - b_bits) : (b_bits - a_bits));
  return ulp_dist <= 4u;
}

template <class V>
bool vec_cmp(const V &a, const V &b) {
  for (unsigned i = 0; i < V::Dimensions; ++i)
    if (!flt_eq(a[i], b[i]))
      return false;
  return true;
}

template <class M>
bool mat_cmp(const M &a, const M &b) {
  for (unsigned i = 0; i < M::Size; ++i) 
    if (!vec_cmp(a[i], b[i])) return false;
  return true;
}

class timer {
public:
  explicit timer(const std::string &tag) : 
      tag_(tag),
      start_(std::chrono::system_clock::now()) {}
  ~timer() {
    auto end = std::chrono::system_clock::now();
    printf("  %s : %zu us\n",
           tag_.c_str(),
           std::chrono::duration_cast<std::chrono::microseconds>(end-start_).count());
  }
private:
  const std::string tag_;
  std::chrono::system_clock::time_point start_;

};

using namespace nm;

NTTEST(vector_ctors_and_accessors, global) {
  constexpr float2 a { 1.0f, 2.0f };
  constexpr float2 b { 3.0f };
  static_assert(a[0] == 1.0f && a[1] == 2.0f);
  static_assert(b[0] == 3.0f && b[1] == 3.0f);
  static_assert(a[0] == a.x() && a[1] == a.y());
  constexpr float3 c { 4.0f };
  constexpr float3 d { 5.0f, 6.0f, 7.0f };
  constexpr float3 e { float2 { 5.0f, 6.0f }, 7.0f};
  constexpr float3 f { 5.0f, float2 { 6.0f, 7.0f } };
  static_assert(c[0] == 4.0f && c[1] == 4.0f && c[2] == 4.0f);
  static_assert(d[0] == 5.0f && d[1] == 6.0f && d[2] == 7.0f);
  static_assert(e == d);
  static_assert(f == d);
  static_assert(d[0] == d.x() && d[1] == d.y() && d[2] == d.z());
  constexpr float4 g { 8.0f };
  constexpr float4 h { 9.0f, 10.0f, 11.0f, 12.0f };
  constexpr float4 i { 9.0f, 10.0f, float2 { 11.0f, 12.0f } };
  constexpr float4 j { 9.0f, float3 { 10.0f, 11.0f, 12.0f } };
  constexpr float4 k { 9.0f, float2 { 10.0f, 11.0f }, 12.0f };
  constexpr float4 l { float2 { 9.0f, 10.0f }, 11.0f, 12.0f };
  constexpr float4 m { float2 { 9.0f, 10.0f }, float2 { 11.0f, 12.0f } };
  constexpr float4 n { float3 { 9.0f, 10.0f, 11.0f }, 12.0f };
  static_assert(g[0] == 8.0f && g[1] == 8.0f && g[2] == 8.0f && g[3] == 8.0f);
  static_assert(h[0] == 9.0f && h[1] == 10.0f && h[2] == 11.0f && h[3] == 12.0f);
  static_assert(i == h);
  static_assert(j == h);
  static_assert(k == h);
  static_assert(l == h);
  static_assert(m == h);
  static_assert(n == h);
  static_assert(h[0] == h.x() && h[1] == h.y() && h[2] == h.z() && h[3] == h.w());
}

NTTEST(matrix_ctors_and_accessors, global) {
  constexpr auto a = float2x2::from_columns(float2 { 1.0f, 2.0f },
                                            float2 { 3.0f, 4.0f });
  static_assert(a[0][0] == 1.0f && a[0][1] == 2.0f &&
             a[1][0] == 3.0f && a[1][1] == 4.0f);
  constexpr auto b = float2x2::from_rows(float2{ 1.0f, 2.0f },
                                         float2{ 3.0f, 4.0f });
  static_assert(b[0][0] == 1.0f && b[0][1] == 3.0f && b[1][0] == 2.0f &&
             b[1][1] == 4.0f);
  constexpr auto c = float2x2::identity();
  static_assert(c[0][0] == 1.0f && c[0][1] == 0.0f &&
             c[1][0] == 0.0f && c[1][1] == 1.0f);
  constexpr auto d = float3x3::from_columns(float3 { 1.0f, 2.0f, 3.0f },
                                            float3 { 4.0f, 5.0f, 6.0f },
                                            float3 { 7.0f, 8.0f, 9.0f });
  static_assert(d[0][0] == 1.0f && d[0][1] == 2.0f && d[0][2] == 3.0f &&
             d[1][0] == 4.0f && d[1][1] == 5.0f && d[1][2] == 6.0f &&
             d[2][0] == 7.0f && d[2][1] == 8.0f && d[2][2] == 9.0f);
  constexpr auto e = float3x3::from_rows(float3 { 1.0f, 2.0f, 3.0f },
                                         float3 { 4.0f, 5.0f, 6.0f },
                                         float3 { 7.0f, 8.0f, 9.0f });
  static_assert(e[0][0] == 1.0f && e[1][0] == 2.0f && e[2][0] == 3.0f &&
             e[0][1] == 4.0f && e[1][1] == 5.0f && e[2][1] == 6.0f &&
             e[0][2] == 7.0f && e[1][2] == 8.0f && e[2][2] == 9.0f);
  constexpr auto f = float3x3::identity();
  static_assert(f[0][0] == 1.0f && f[1][0] == 0.0f && f[2][0] == 0.0f &&
             f[0][1] == 0.0f && f[1][1] == 1.0f && f[2][1] == 0.0f &&
             f[0][2] == 0.0f && f[1][2] == 0.0f && f[2][2] == 1.0f);
  constexpr auto g = float4x4::from_columns(float4 { 1.0f, 2.0f, 3.0f, 4.0f },
                                            float4 { 5.0f, 6.0f, 7.0f, 8.0f },
                                            float4 { 9.0f, 10.0f, 11.0f, 12.0f },
                                            float4 { 13.0f, 14.0f, 15.0f, 16.0f });
  static_assert(g[0][0] == 1.0f && g[0][1] == 2.0f && g[0][2] == 3.0f && g[0][3] == 4.0f &&
             g[1][0] == 5.0f && g[1][1] == 6.0f && g[1][2] == 7.0f && g[1][3] == 8.0f &&
             g[2][0] == 9.0f && g[2][1] == 10.0f && g[2][2] == 11.0f && g[2][3] == 12.0f &&
             g[3][0] == 13.0f && g[3][1] == 14.0f && g[3][2] == 15.0f && g[3][3] == 16.0f);
  constexpr auto h = float4x4::from_rows(float4 { 1.0f, 2.0f, 3.0f, 4.0f },
                                         float4 { 5.0f, 6.0f, 7.0f, 8.0f },
                                         float4 { 9.0f, 10.0f, 11.0f, 12.0f },
                                         float4 { 13.0f, 14.0f, 15.0f, 16.0f });
  static_assert(h[0][0] == 1.0f && h[1][0] == 2.0f && h[2][0] == 3.0f && h[3][0] == 4.0f &&
             h[0][1] == 5.0f && h[1][1] == 6.0f && h[2][1] == 7.0f && h[3][1] == 8.0f &&
             h[0][2] == 9.0f && h[1][2] == 10.0f && h[2][2] == 11.0f && h[3][2] == 12.0f &&
             h[0][3] == 13.0f && h[1][3] == 14.0f && h[2][3] == 15.0f && h[3][3] == 16.0f);
  constexpr auto i = float4x4::identity();
  static_assert(i[0][0] == 1.0f && i[1][0] == 0.0f && i[2][0] == 0.0f && i[3][0] == 0.0f &&
             i[0][1] == 0.0f && i[1][1] == 1.0f && i[2][1] == 0.0f && i[3][1] == 0.0f &&
             i[0][2] == 0.0f && i[1][2] == 0.0f && i[2][2] == 1.0f && i[3][2] == 0.0f &&
             i[0][3] == 0.0f && i[1][3] == 0.0f && i[2][3] == 0.0f && i[3][3] == 1.0f);
}

// god forgive me for this
template <class F, class... T>
void for_args(const F &fn, T&&... args) {
  auto t = std::make_tuple((fn(std::forward<T>(args)),0)...);
}

template <class... T>
void read_into_containers(const std::string &filename, T&... containers) {
  FILE *f = fopen(("testdata/" + filename).c_str(), "rb");
  NTASSERT(f != nullptr);
  for_args([](auto &c) { c.reserve(1000); }, containers...);
  while (!feof(f)) {
    for_args([f](auto &c) {
               c.push_back({});
               if (read(f, c.back()) == 0)
                 c.pop_back();
             },
             containers...);
  }
  fclose(f);
}

template <class T, class... Ts>
void assert_all_same_size(const T &t, const Ts&... ts) {
  const size_t size = t.size();
  for_args([size](const auto &t) { NTASSERT(size == t.size()); },
            ts...);
}

#define TIMESCOPE() timer _t(__FUNCTION__)

template <unsigned N>
void test_vec_asmd() {
  const std::string Nstr = std::to_string(N);
  using vtype = vec<float, N>;
  std::vector<vtype> v1s;
  std::vector<vtype> v2s;
  std::vector<vtype> as;
  std::vector<vtype> ss;
  std::vector<vtype> ms;
  std::vector<vtype> ds;
  read_into_containers("vec" + Nstr + "asmd", v1s, v2s, as, ss, ms, ds);
  NTASSERT(v1s.size() > 100u); // sanity check
  assert_all_same_size(v1s, v2s, as, ss, ms, ds);

  TIMESCOPE();
  for (unsigned i = 0u; i < v1s.size(); ++i) {
    NTASSERT(v1s[i] + v2s[i] == as[i] &&
             v1s[i] - v2s[i] == ss[i] &&
             v1s[i] * v2s[i] == ms[i] &&
             v1s[i] / v2s[i] == ds[i]);
  }
}

NTTEST(vector_add_subtract_multiply_divide, global) {
  test_vec_asmd<2>();
  test_vec_asmd<3>();
  test_vec_asmd<4>();
}

template <unsigned N>
void test_vec_dot() {
  const std::string Nstr = std::to_string(N);
  using vtype = vec<float, N>;
  std::vector<vtype> v1s;
  std::vector<vtype> v2s;
  std::vector<float> ds;
  read_into_containers("dot" + Nstr, v1s, v2s, ds);
  NTASSERT(v1s.size() > 100u); // sanity check
  assert_all_same_size(v1s, v2s, ds);
  NTASSERT(v1s.size() == v2s.size() &&
           v1s.size() == ds.size());
  TIMESCOPE();
  for (unsigned i = 0u; i < v1s.size(); ++i) {
    NTASSERT(dot(v1s[i], v2s[i]) == ds[i]);
  }
}

NTTEST(dot_product, global) {
  test_vec_dot<2>();
  test_vec_dot<3>();
  test_vec_dot<4>();
}

template <unsigned N>
void test_mat_mul_vec() {
  const std::string Nstr = std::to_string(N);
  using vtype = vec<float, N>;
  using mtype = mat<float, N>;
  std::vector<mtype> ms;
  std::vector<vtype> vs;
  std::vector<vtype> rs;
  read_into_containers("m" + Nstr + "v" + Nstr, ms, vs, rs);
  NTASSERT(vs.size() > 100u); // sanity check
  assert_all_same_size(vs, ms, rs);
  TIMESCOPE();
  for (unsigned i = 0u; i < vs.size(); ++i) {
    NTASSERT(ms[i] * vs[i] == rs[i]);
  }
}

NTTEST(cross_product, global) {
  std::vector<float3> as;
  std::vector<float3> bs;
  std::vector<float3> cs;
  read_into_containers("cross", as, bs, cs);
  NTASSERT(as.size() > 100);
  assert_all_same_size(as, bs, cs);
  TIMESCOPE();
  for (unsigned i = 0u; i < as.size(); ++i) {
    NTASSERT(cross(as[i], bs[i]) == cs[i]);
  }
}

NTTEST(matrix_vector_multiplication, global) {
  test_mat_mul_vec<2>();
  test_mat_mul_vec<3>();
  test_mat_mul_vec<4>();
}

template <unsigned N>
void test_vec_scale() {
  const std::string Nstr = std::to_string(N);
  using vtype = vec<float, N>;
  std::vector<vtype> vs;
  std::vector<float> ss;
  std::vector<vtype> ms;
  std::vector<vtype> ds;
  read_into_containers("vec" + Nstr + "s", vs, ss, ms, ds);
  assert_all_same_size(vs, ss, ms, ds);
  TIMESCOPE();
  for (unsigned i = 0u; i < vs.size(); ++i) {
    NTASSERT(vec_cmp(vs[i] * ss[i], ms[i]));
    NTASSERT(vec_cmp(vs[i] / ss[i], ds[i]));
  }
}

NTTEST(vector_scale, global) {
  test_vec_scale<2>();
  test_vec_scale<3>();
  test_vec_scale<4>();
}

template <unsigned N>
void test_mat_mul_mat() {
  const std::string Nstr = std::to_string(N);
  using vtype = vec<float, N>;
  using mtype = mat<float, N>;
  std::vector<mtype> as;
  std::vector<mtype> bs;
  std::vector<mtype> cs;
  read_into_containers("m" + Nstr + "m" + Nstr, as, bs, cs);
  NTASSERT(as.size() > 100u); // sanity check
  NTASSERT(as.size() == bs.size() && as.size() == cs.size());
  TIMESCOPE();
  for (unsigned i = 0u; i < as.size(); ++i) {
    NTASSERT(as[i] * bs[i] == cs[i]);
  }
}

NTTEST(matrix_matrix_multiplication, global) {
  test_mat_mul_mat<2>();
  test_mat_mul_mat<3>();
  test_mat_mul_mat<4>();
}

template <unsigned N>
void test_mat_inv() {
  const std::string Nstr = std::to_string(N);
  using vtype = vec<float, N>;
  using mtype = mat<float, N>;
  std::vector<mtype> ms;
  std::vector<mtype> is;
  const mtype id = mtype::identity();
  read_into_containers("inv" + Nstr, ms, is);
  NTASSERT(ms.size() > 100u); // sanity check
  NTASSERT(ms.size() == is.size());
  TIMESCOPE();
  for (unsigned i = 0u; i < ms.size(); ++i) {
    NTASSERT(mat_cmp(ms[i] * inverse(ms[i]), id));
  }
}

NTTEST(matrix_inverse, global) {
  test_mat_inv<2>();
  test_mat_inv<3>();
  test_mat_inv<4>();
}

template <unsigned N>
void test_mat_det() {
  const std::string Nstr = std::to_string(N);
  using vtype = vec<float, N>;
  using mtype = mat<float, N>;
  using stype = typename vtype::Scalar;
  std::vector<mtype> ms;
  std::vector<stype> ds;
  read_into_containers("det" + Nstr, ms, ds);
  NTASSERT(ms.size() > 100u); // sanity check
  NTASSERT(ms.size() == ds.size());
  TIMESCOPE();
  for (unsigned i = 0u; i < ms.size(); ++i) {
    float a = ds[i];
    float b = det(ms[i]);
    NTASSERT(flt_eq(a, b));
  }
}

NTTEST(matrix_determinant, global) {
  test_mat_det<2>();
  test_mat_det<3>();
  test_mat_det<4>();
}

NTTEST(scale_matrix, global) {
  constexpr float2 f2fs { 1.0f, 2.0f };
  constexpr float3 f3fs { 1.0f, 2.0f, 3.0f };
  constexpr float4 f4fs { 1.0f, 2.0f, 3.0f, 4.0f };
  constexpr float2x2 f2s = scale(f2fs);
  constexpr float3x3 f3s = scale(f3fs);
  constexpr float4x4 f4s = scale(f4fs);
  static_assert(f2s[0][0] == 1.0f && f3s[0][0] == 1.0f && f4s[0][0] == 1.0f);
  static_assert(f2s[1][1] == 2.0f && f3s[1][1] == 2.0f && f4s[1][1] == 2.0f);
  static_assert(f3s[2][2] == 3.0f && f4s[2][2] == 3.0f);
  static_assert(f4s[3][3] == 4.0f);
  static_assert(f3s[0][1] == f3s[0][2] == f4s[0][1] == f4s[0][2] == f4s[0][3] ==
             f3s[1][0] == f3s[1][2] == f4s[1][0] == f4s[1][2] == f4s[1][3] ==
             f3s[2][0] == f3s[2][1] == f4s[2][0] == f4s[2][1] == f4s[2][3] ==
             0.0f);
}

template <unsigned N>
void test_translation_matrix() {
  constexpr vec<float, N> offset { 98.0f };
  constexpr mat<float, N + 1> translation_mtx = translation(offset);
  constexpr vec<float, N + 1> point_translation_result =
      translation_mtx * vec<float, N + 1> { offset, 1.0f };
  constexpr vec<float, N + 1> expected_point_translation_result {
      offset * 2.0f, 1.0f };
  constexpr vec<float, N + 1> vector_translation_result =
      translation_mtx * vec<float, N + 1> { offset, 0.0f };
  constexpr vec<float, N + 1> expected_vector_translation_result = {
      offset, 0.0f };
  static_assert(point_translation_result == expected_point_translation_result);
  static_assert(vector_translation_result == expected_vector_translation_result);
}

NTTEST(translation_matrix, global) {
  test_translation_matrix<2>();
  test_translation_matrix<3>();
}

template <unsigned N>
void test_vector_magnitude() {
  const std::string Nstr = std::to_string(N);
  using vtype = vec<float, N>;
  std::vector<vtype> vs;
  std::vector<float> mss;
  std::vector<float> ms;
  read_into_containers("vec" + Nstr + "mag", vs, mss, ms);
  NTASSERT(vs.size() > 100u); // sanity check
  assert_all_same_size(vs, mss, ms);
  TIMESCOPE();
  for (unsigned i = 0u; i < vs.size(); ++i) {
    NTASSERT(lengthsq(vs[i]) == mss[i]);
    NTASSERT(length(vs[i]) == ms[i]);
  }
}

NTTEST(vector_magnitude, global) {
  test_vector_magnitude<2>();
  test_vector_magnitude<3>();
  test_vector_magnitude<4>();
}

template <unsigned N>
void test_vector_projection() {
  const std::string Nstr = std::to_string(N);
  using vtype = vec<float, N>;
  std::vector<vtype> as;
  std::vector<vtype> bs;
  std::vector<vtype> ps;
  read_into_containers("vec" + Nstr + "prj", as, bs, ps);
  NTASSERT(as.size() > 100u); // sanity check
  assert_all_same_size(as, bs, ps);
  TIMESCOPE();
  for (unsigned i = 0u; i < as.size(); ++i) {
    NTASSERT(project(as[i], bs[i]) == ps[i]);
  }
}

NTTEST(vector_projection, global) {
  test_vector_projection<2>();
  test_vector_projection<3>();
  test_vector_projection<4>();
}

template <unsigned N>
void test_matrix_transpose() {
  const std::string Nstr = std::to_string(N);
  using mtype = mat<float, N>;
  std::vector<mtype> as;
  std::vector<mtype> bs;
  read_into_containers("trn" + Nstr, as, bs);
  NTASSERT(as.size() > 100u); // sanity check
  assert_all_same_size(as, bs);
  TIMESCOPE();
  for (unsigned i = 0u; i < as.size(); ++i) {
    NTASSERT(transpose(as[i]) == bs[i]);
  }
}

NTTEST(matrix_transpose, global) {
  test_matrix_transpose<2>();
  test_matrix_transpose<3>();
  test_matrix_transpose<4>();
}

int main() {
  return nt_run_tests(global) ? 1 : 0;
}
