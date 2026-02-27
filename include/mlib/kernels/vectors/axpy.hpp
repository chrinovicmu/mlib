#ifndef MLIB_KERNELS_AXPY_HPP
#define MLIB_KERNELS_AXPY_HPP

#include <mlib/vector.hpp>
#include <cstddef>
#include <stdexcept>

namespace mlib {
namespace kernels {

template<typename T>
void axpy(T alpha, const T* __restrict x, T* __restrict y, size_t count);

inline void axpy_p32(float alpha, const Vector<float>& x, Vector<float>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector sizes must match in axpy_p32");
    }
    if (x.empty()) {
        return;
    }
    if (alpha == 0.0f) {
        return;  
    }

    axpy(alpha, x.aligned_data(), y.aligned_data(), x.size());
}

inline void axpy_p64(double alpha, const Vector<double>& x, Vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector sizes must match in axpy_p64");
    }
    if (x.empty()) {
        return;
    }
    if (alpha == 0.0) {
        return;
    }

    axpy(alpha, x.aligned_data(), y.aligned_data(), x.size());
}

} // namespace kernels
} // namespace mlib

#if defined(__AVX2__)
    #include "../../backends/avx2/vectors/axpy.cpp"
#elif defined(__ARM_NEON)
    #include "../../backends/neon/vectors/axpy.cpp"
#else
    #include "../../backends/scalar/vectors/axpy.cpp"
#endif

#endif
