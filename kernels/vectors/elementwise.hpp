#ifndef MLIB_KERNELS_VECTORS_ELEMENTWISE_HPP
#define MLIB_KERNELS_VECTORS_ELEMENTWISE_HPP

#include <mlib/vector.hpp>
#include <cstddef>
#include <stdexcept>

namespace mlib {
namespace kernels {

template<typename T>
void add(const Vector<T>& x, const Vector<T>& y, Vector<T>& z);

template<typename T>
void sub(const Vector<T>& x, const Vector<T>& y, Vector<T>& z);

template<typename T>
void mul(const Vector<T>& x, const Vector<T>& y, Vector<T>& z);

template<typename T>
void div(const Vector<T>& x, const Vector<T>& y, Vector<T>& z);

template<typename T>
void abs(const Vector<T>& x, Vector<T>& y);

template<typename T>
void neg(const Vector<T>& x, Vector<T>& y);


inline void add_p32(const Vector<float>& x, const Vector<float>& y, Vector<float>& z) {
    add(x, y, z);
}

inline void add_p64(const Vector<double>& x, const Vector<double>& y, Vector<double>& z) {
    add(x, y, z);
}

inline void sub_p32(const Vector<float>& x, const Vector<float>& y, Vector<float>& z) {
    sub(x, y, z);
}

inline void sub_p64(const Vector<double>& x, const Vector<double>& y, Vector<double>& z) {
    sub(x, y, z);
}

inline void mul_p32(const Vector<float>& x, const Vector<float>& y, Vector<float>& z) {
    mul(x, y, z);
}

inline void mul_p64(const Vector<double>& x, const Vector<double>& y, Vector<double>& z) {
    mul(x, y, z);
}

inline void div_p32(const Vector<float>& x, const Vector<float>& y, Vector<float>& z) {
    div(x, y, z);
}

inline void div_p64(const Vector<double>& x, const Vector<double>& y, Vector<double>& z) {
    div(x, y, z);
}

inline void abs_p32(const Vector<float>& x, Vector<float>& y) {
    abs(x, y);
}

inline void abs_p64(const Vector<double>& x, Vector<double>& y) {
    abs(x, y);
}

inline void neg_p32(const Vector<float>& x, Vector<float>& y) {
    neg(x, y);
}

inline void neg_p64(const Vector<double>& x, Vector<double>& y) {
    neg(x, y);
}

} // namespace kernels
} // namespace mlib


#if defined(__AVX2__)
    #include "../../backends/avx2/vectors/elementwise.cpp"
#elif defined(__ARM_NEON)
    #include "../../backends/neon/vectors/elementwise.cpp"
#else
    #include "../../backends/scalar/vectors/elementwise.cpp"
#endif

#endif
