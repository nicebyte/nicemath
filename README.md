<img src="https://github.com/nicebyte/nicemath/blob/master/nicemath.png?raw=true" width="256"/>

![build status](https://ci.appveyor.com/api/projects/status/f3tvwbeyvattk9xd?svg=true)

## Introduction
nicemath is a compact single-header C++ library that provides data types and routines for basic linear algebra operations often encountered in computer graphics and game development.

To use the library, simply place [nicemath.h](https://raw.githubusercontent.com/nicebyte/nicemath/master/nicemath.h) anywhere within your C++ project's include path.

## Usage
This is not a generic linear algebra library. It is mostly intended to help users deal with 3D and 2D affine transforms, quaternions and basic vector operations. If you require e.g. support for arbitrarily large, sparse matrices, you should look elsewhere.

## Example

```cpp
#include "nicemath.h"
void vec_demo(const nm::float4 &a, const nm::float4 &b) {
  // basic operations.
  const nm::float4 sum = a + b,
                   dif = a - b,
                   prd = a * b, // elementwise product
                   div = a / b; // elementwise division
                   
  // dot product
  const float dot = nm::dot(a, b);
  
  // cross product
  const nm::float3 cross = nm::cross(a.xyz(), b.xyz());
  
  // arbitrary vector swizzles are supported
  const nm::float3 a_wyx = a.wyx(),
                   b_zyy = b.zyy();
                   
  // vector expressions may be used in c++ constexprs.
  constexpr float scale = 2.0f;
  constexpr nm::float4 scaled_vector =
      scale * nm::float4 { 1.0f, .0f, .0f, .0f};
}

void mat_demo(nm::float4x4 a, nm::float4x4 b) {
  // basic operations
  const nm::float4x4 prd = a * b;

  // matrix-vector mul
  const nm::float4 v = a * nm::float4 { 1.f, .0f, .0f, .0f };
  
  // inverse
  const nm::float4x4 inv = nm::inverse(a);
  
  // determinant
  const float det = nm::det(a);
  
  // matrix expressions can be used in c++ contstexprs as well
  constexpr nm::float4x4 id = nm::float4x4::identity();
  constexpr nm::float4x4 scaled5x = 5.f * id;
  constexpr nm::float4x4 m = scaled5x * nm::float4x4::from_rows(nm::float4 { 2.f, 3.f, .0f, .0f },
                                                                nm::float4 { 9.f, 1.f, 2.f, .0f },
                                                                nm::float4 { 8.f, 4.f, .7f, .0f },
                                                                nm::float4 { 6.f, 7.f, 1.f, 1.f });
}
```
