#ifndef MLIB_KERNELS_SCAL_HPP
#define MLIB_KERNELS_SCAL_HPP

#include <mlib/vector.hpp>
#include <cstddef>
#include <stdexcept>

namespace mlib {
namespace kernels {

template<typename T>
void scal(T alpha, T* __restrict x, size_t count);

inline void scal_p32(float alpha, Vector<float>& x) {
    if (x.empty()) {
        return;
    }
    if (alpha == 0.0f) {
        return;
    }

    scal(alpha, x.aligned_data(), x.size());
}

inline void scal_p64(double alpha, Vector<double>& x) {
    if (x.empty()) {
        return;
    }
    if (alpha == 0.0) {
        return;
    }

    scal(alpha, x.aligned_data(), x.size());
}

} // namespace kernels
} // namespace mlib

#if defined(__AVX2__)
    #include "../../backends/avx2/vectors/scal.cpp"
#elif defined(__ARM_NEON)
    #include "../../backends/neon/vectors/scal.cpp"
#else
    #include "../../backends/scalar/vectors/scal.cpp"
#endif

#endif
