#ifndef MLIB_KERNELS_MATRIX_SUB_HPP
#define MLIB_KERNELS_MATRIX_SUB_HPP

#include <mlib/matrix.hpp>
#include <cstddef>
#include <stdexcept>

namespace mlib {
namespace kernels {

template<typename T>
void sub(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);

inline void mat_sub_p32(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    if (A.rows() != B.rows() || A.cols() != B.cols() || A.rows() != C.rows() || A.cols() != C.cols()) {
        throw std::invalid_argument("Matrix dimensions must match in mat_sub_p32");
    }
    sub(A, B, C);
}

inline void mat_sub_p64(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C) {
    if (A.rows() != B.rows() || A.cols() != B.cols() || A.rows() != C.rows() || A.cols() != C.cols()) {
        throw std::invalid_argument("Matrix dimensions must match in mat_sub_p64");
    }
    sub(A, B, C);
}

} // namespace kernels
} // namespace mlib

#if defined(__AVX2__)
    #include "../../backends/avx2/matrix/sub.cpp"
#elif defined(__ARM_NEON)
    #include "../../backends/neon/matrix/sub.cpp"
#else
    #include "../../backends/scalar/matrix/sub.cpp"
#endif

#endif
