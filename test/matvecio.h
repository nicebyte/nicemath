#pragma once

#include "../nicemath.h"
#include <iostream>
#include <stdio.h>

namespace nm {

template <unsigned N, class S>
std::istream& operator>>(std::istream &in, vec<S, N> &v) {
  for (unsigned i = 0u; i < N; ++i) in >> v.data[i];
  return in;
}

template <class S, unsigned N>
std::istream& operator>>(std::istream &in, mat<S, N> &m) {
  for (unsigned i = 0u; i < N; ++i) in >> m.column[i];
  return in;
}

template <unsigned N, class S>
std::ostream& operator<<(std::ostream &out, const vec<S, N> &v) {
  for (unsigned i = 0u; i < N; ++i) out << v.data[i] << " ";
  return out;
}

template <class S, unsigned N>
std::ostream& operator<<(std::ostream &out, const mat<S, N> &m) {
  for (unsigned i = 0u; i < mat<S, N>::Size; ++i) out << m.column[i] << "\n";
  out << "\n";
  return out;
}

template <unsigned N, class S>
int read(FILE *f, vec<S, N> &v) {
  return fread(&v.data[0], sizeof(S), N, f);
}

template <class S, unsigned N>
int read(FILE *f, mat<S, N> &m) {
  int totalbytes = 0u;
  for (unsigned i = 0u; i < N; ++i)
    totalbytes += read(f, m.column[i]);
  return totalbytes;
}

int read(FILE *f, float &v) {
  return fread(&v, sizeof(float), 1, f);
}

}
