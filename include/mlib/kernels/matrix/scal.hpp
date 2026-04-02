#ifndef MLIB_KERNELS_MATRIX_SCAL_HPP
#define MLIB_KERNELS_MATRIX_SCAL_HPP

#include <mlib/matrix.hpp>
#include <cstddef>
#include <stdexcept>

namespace mlib {
namespace kernels {

template<typename T>
void scal(T alpha, Matrix<T>& A);

inline void mat_scal_p32(float alpha, Matrix<float>& A) {
    if (A.empty()) return;
    scal(alpha, A);
}

inline void mat_scal_p64(double alpha, Matrix<double>& A) {
    if (A.empty()) return;
    scal(alpha, A);
}

} // namespace kernels
} // namespace mlib

#if defined(__AVX2__)
    #include "../../backends/avx2/matrix/scal.cpp"
#elif defined(__ARM_NEON)
    #include "../../backends/neon/matrix/scal.cpp"
#else
    #include "../../backends/scalar/matrix/scal.cpp"
#endif

#endif
