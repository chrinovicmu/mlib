#ifndef MLIB_KERNELS_MATRIX_GEMV_HPP
#define MLIB_KERNELS_MATRIX_GEMV_HPP

#include <mlib/matrix.hpp>
#include <mlib/vector.hpp>
#include <cstddef>
#include <stdexcept>

namespace mlib {
namespace kernels {

template<typename T>
void gemv(T alpha, const Matrix<T>& A, const Vector<T>& x, T beta, Vector<T>& y);

inline void gemv_p32(float alpha,
                     const Matrix<float>& A,
                     const Vector<float>& x,
                     float beta,
                     Vector<float>& y) {
    if (A.cols() != x.size() || A.rows() != y.size()) {
        throw std::invalid_argument("Matrix-vector dimensions incompatible in gemv_p32");
    }
    gemv(alpha, A, x, beta, y);
}

inline void gemv_p64(double alpha,
                     const Matrix<double>& A,
                     const Vector<double>& x,
                     double beta,
                     Vector<double>& y) {
    if (A.cols() != x.size() || A.rows() != y.size()) {
        throw std::invalid_argument("Matrix-vector dimensions incompatible in gemv_p64");
    }
    gemv(alpha, A, x, beta, y);
}

} // namespace kernels
} // namespace mlib

#if defined(__AVX2__)
    #include "../../backends/avx2/matrix/gemv.cpp"
#elif defined(__ARM_NEON)
    #include "../../backends/neon/matrix/gemv.cpp"
#else
    #include "../../backends/scalar/matrix/gemv.cpp"
#endif

#endif
