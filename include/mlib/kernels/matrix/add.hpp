#ifndef MLIB_KERNELS_MATRIX_ADD_HPP
#define MLIB_KERNELS_MATRIX_ADD_HPP

#include <mlib/matrix.hpp>
#include <cstddef>
#include <stdexcept>

namespace mlib {
namespace kernels {

template<typename T>
void add(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);

inline void mat_add_p32(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    if (A.rows() != B.rows() || A.cols() != B.cols() || A.rows() != C.rows() || A.cols() != C.cols()) {
        throw std::invalid_argument("Matrix dimensions must match in mat_add_p32");
    }
    add(A, B, C);
}

inline void mat_add_p64(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C) {
    if (A.rows() != B.rows() || A.cols() != B.cols() || A.rows() != C.rows() || A.cols() != C.cols()) {
        throw std::invalid_argument("Matrix dimensions must match in mat_add_p64");
    }
    add(A, B, C);
}

} // namespace kernels
} // namespace mlib

#if defined(__AVX2__)
    #include "../../backends/avx2/matrix/add.cpp"
#elif defined(__ARM_NEON)
    #include "../../backends/neon/matrix/add.cpp"
#else
    #include "../../backends/scalar/matrix/add.cpp"
#endif

#endif
