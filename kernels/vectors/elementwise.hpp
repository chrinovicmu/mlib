#ifndef MLIB_KERNELS_VECTORS_ELEMENTWISE_HPP
#define MLIB_KERNELS_VECTORS_ELEMENTWISE_HPP

#include <mlib/vector.hpp>
#include <cstddef>
#include <stdexcept>

namespace mlib {
namespace kernels {

template<typename T>
void add(const T* __restrict x, const T* __restrict y, T* __restrict z, size_t count);

template<typename T>
void sub(const T* __restrict x, const T* __restrict y, T* __restrict z, size_t count);

template<typename T>
void mul(const T* __restrict x, const T* __restrict y, T* __restrict z, size_t count);

template<typename T>
void div(const T* __restrict x, const T* __restrict y, T* __restrict z, size_t count);

template<typename T>
void abs(const T* __restrict x, T* __restrict y, size_t count);

template<typename T>
void neg(const T* __restrict x, T* __restrict y, size_t count);


inline void add_p32(const Vector<float>& x, const Vector<float>& y, Vector<float>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Vector sizes must match in add_p32");
    }
    if (x.empty()) return;
    add(x.aligned_data(), y.aligned_data(), z.aligned_data(), x.size());
}

inline void add_p64(const Vector<double>& x, const Vector<double>& y, Vector<double>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Vector sizes must match in add_p64");
    }
    if (x.empty()) return;
    add(x.aligned_data(), y.aligned_data(), z.aligned_data(), x.size());
}

inline void sub_p32(const Vector<float>& x, const Vector<float>& y, Vector<float>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Vector sizes must match in sub_p32");
    }
    if (x.empty()) return;
    sub(x.aligned_data(), y.aligned_data(), z.aligned_data(), x.size());
}

inline void sub_p64(const Vector<double>& x, const Vector<double>& y, Vector<double>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Vector sizes must match in sub_p64");
    }
    if (x.empty()) return;
    sub(x.aligned_data(), y.aligned_data(), z.aligned_data(), x.size());
}

inline void mul_p32(const Vector<float>& x, const Vector<float>& y, Vector<float>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Vector sizes must match in mul_p32");
    }
    if (x.empty()) return;
    mul(x.aligned_data(), y.aligned_data(), z.aligned_data(), x.size());
}

inline void mul_p64(const Vector<double>& x, const Vector<double>& y, Vector<double>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Vector sizes must match in mul_p64");
    }
    if (x.empty()) return;
    mul(x.aligned_data(), y.aligned_data(), z.aligned_data(), x.size());
}

inline void div_p32(const Vector<float>& x, const Vector<float>& y, Vector<float>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Vector sizes must match in div_p32");
    }
    if (x.empty()) return;
    div(x.aligned_data(), y.aligned_data(), z.aligned_data(), x.size());
}

inline void div_p64(const Vector<double>& x, const Vector<double>& y, Vector<double>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Vector sizes must match in div_p64");
    }
    if (x.empty()) return;
    div(x.aligned_data(), y.aligned_data(), z.aligned_data(), x.size());
}

inline void abs_p32(const Vector<float>& x, Vector<float>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector sizes must match in abs_p32");
    }
    if (x.empty()) return;
    abs(x.aligned_data(), y.aligned_data(), x.size());
}

inline void abs_p64(const Vector<double>& x, Vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector sizes must match in abs_p64");
    }
    if (x.empty()) return;
    abs(x.aligned_data(), y.aligned_data(), x.size());
}

inline void neg_p32(const Vector<float>& x, Vector<float>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector sizes must match in neg_p32");
    }
    if (x.empty()) return;
    neg(x.aligned_data(), y.aligned_data(), x.size());
}

inline void neg_p64(const Vector<double>& x, Vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector sizes must match in neg_p64");
    }
    if (x.empty()) return;
    neg(x.aligned_data(), y.aligned_data(), x.size());
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
