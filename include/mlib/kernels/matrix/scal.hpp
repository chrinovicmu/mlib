#ifndef MLIB_KERNELS_MATRIX_SCAL_HPP
#define MLIB_KERNELS_MATRIX_SCAL_HPP

#include <mlib/matrix.hpp>
#include <cstddef>
#include <stdexcept>

namespace mlib {
namespace kernels {

// Low-level kernel forward declaration
template<typename T>
void mscal(T alpha, Matrix<T>& A);

// Public wrappers
inline void mscal_p32(float alpha, Matrix<float>& A) {
    if (A.empty()) {
        return;
    }
    mscal(alpha, A);
}

inline void mscal_p64(double alpha, Matrix<double>& A) {
    if (A.empty()) {
        return;
    }
    mscal(alpha, A);
}

} // namespace kernels
} // namespace mlib

// Backend dispatch
#if defined(__AVX2__)
    #include "../../backends/avx2/matrix/scal.cpp"
#elif defined(__ARM_NEON)
    #include "../../backends/neon/matrix/scal.cpp"
#else
    #include "../../backends/scalar/matrix/scal.cpp"
#endif

#endif
