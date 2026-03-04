#ifndef MLIB_KERNELS_VECTORS_NORMS_HPP
#define MLIB_KERNELS_VECTORS_NORMS_HPP

#include <mlib/vector.hpp>
#include <cstddef>
#include <stdexcept>
#include <cmath>

namespace mlib {
namespace kernels {

template<typename T>
T l1_norm(const T* __restrict x, size_t count);

template<typename T>
T l2_norm(const T* __restrict x, size_t count);

inline float l1_norm_p32(const Vector<float>& x) {
    if (x.empty()) {
        return 0.0f;
    }
    return l1_norm(x.aligned_data(), x.size());
}

inline double l1_norm_p64(const Vector<double>& x) {
    if (x.empty()) {
        return 0.0;
    }
    return l1_norm(x.aligned_data(), x.size());
}

inline float l2_norm_p32(const Vector<float>& x) {
    if (x.empty()) {
        return 0.0f;
    }
    return l2_norm(x.aligned_data(), x.size());
}

inline double l2_norm_p64(const Vector<double>& x) {
    if (x.empty()) {
        return 0.0;
    }
    return l2_norm(x.aligned_data(), x.size());
}

} // namespace kernels
} // namespace mlib

#if defined(__AVX2__)
    #include "../../backends/avx2/vectors/norms.cpp"
#elif defined(__ARM_NEON)
    #include "../../backends/neon/vectors/norms.cpp"
#else
    #include "../../backends/scalar/vectors/norms.cpp"
#endif

#endif
