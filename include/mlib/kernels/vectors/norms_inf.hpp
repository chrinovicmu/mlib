#ifndef MLIB_KERNELS_VECTORS_NORMS_INF_HPP
#define MLIB_KERNELS_VECTORS_NORMS_INF_HPP

#include <mlib/vector.hpp>
#include <cstddef>
#include <stdexcept>
#include <cmath>  

namespace mlib {
namespace kernels {

template<typename T>
T norm_inf(const T* __restrict x, size_t count);

inline float norm_inf_p32(const Vector<float>& x) {
    if (x.empty()) {
        return 0.0f;
    }
    return norm_inf(x.aligned_data(), x.size());
}

inline double norm_inf_p64(const Vector<double>& x) {
    if (x.empty()) {
        return 0.0;
    }
    return norm_inf(x.aligned_data(), x.size());
}

} // namespace kernels
} // namespace mlib

#if defined(__AVX2__)
    #include "../../backends/avx2/vectors/norms_inf.cpp"
#elif defined(__ARM_NEON)
    #include "../../backends/neon/vectors/norms_inf.cpp"
#else
    #include "../../backends/scalar/vectors/norms_inf.cpp"
#endif

#endif
