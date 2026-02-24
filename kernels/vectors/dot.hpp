#ifndef MLIB_KERNELS_VECTORS_DOT_HPP
#define MLIB_KERNELS_VECTORS_DOT_HPP

#include <mlib/vector.hpp>
#include <cstddef>
#include <stdexcept>

namespace mlib {
namespace kernels {

template<typename T>
T dot(const T* __restrict x, const T* __restrict y, size_t count);

inline float dot_p32(const Vector<float>& x, const Vector<float>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector sizes must match in dot_p32");
    }
    if (x.empty()) {
        return 0.0f;
    }

    return dot(x.aligned_data(), y.aligned_data(), x.size());
}

inline double dot_p64(const Vector<double>& x, const Vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector sizes must match in dot_p64");
    }
    if (x.empty()) {
        return 0.0;
    }

    return dot(x.aligned_data(), y.aligned_data(), x.size());
}

} // namespace kernels
} // namespace mlib

#if defined(__AVX2__)
    #include "../../backends/avx2/vectors/dot.cpp"
#elif defined(__ARM_NEON)
    #include "../../backends/neon/vectors/dot.cpp"
#else
    #include "../../backends/scalar/vectors/dot.cpp"
#endif

#endif
