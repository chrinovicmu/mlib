#ifndef MLIB_KERNELS_MATRIX_ELEMENTWISE_HPP
#define MLIB_KERNELS_MATRIX_ELEMENTWISE_HPP

#include <mlib/matrix.hpp>
#include <cstddef>
#include <stdexcept>

namespace mlib {
namespace kernels {

template<typename T>
void mat_add(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);

template<typename T>
void mat_sub(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);

template<typename T>
void mat_hadamard(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C); // Element-wise multiplication (Hadamard product)

template<typename T>
void mat_div(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);      // Element-wise division

template<typename T>
void mat_add_scalar(T alpha, const Matrix<T>& A, Matrix<T>& C);

template<typename T>
void mat_mul_scalar(T alpha, Matrix<T>& A); // in-place scalar multiplication


template<typename T>
void mat_neg(const Matrix<T>& A, Matrix<T>& C);

template<typename T>
void mat_abs(const Matrix<T>& A, Matrix<T>& C);


inline void mat_add_p32(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    if (A.rows() != B.rows() || A.cols() != B.cols() ||
        A.rows() != C.rows() || A.cols() != C.cols()) {
        throw std::invalid_argument("Matrix dimensions must match in mat_add_p32");
    }
    mat_add(A, B, C);
}

inline void mat_add_p64(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C) {
    if (A.rows() != B.rows() || A.cols() != B.cols() ||
        A.rows() != C.rows() || A.cols() != C.cols()) {
        throw std::invalid_argument("Matrix dimensions must match in mat_add_p64");
    }
    mat_add(A, B, C);
}

inline void mat_sub_p32(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    if (A.rows() != B.rows() || A.cols() != B.cols() ||
        A.rows() != C.rows() || A.cols() != C.cols()) {
        throw std::invalid_argument("Matrix dimensions must match in mat_sub_p32");
    }
    mat_sub(A, B, C);
}

inline void mat_sub_p64(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C) {
    if (A.rows() != B.rows() || A.cols() != B.cols() ||
        A.rows() != C.rows() || A.cols() != C.cols()) {
        throw std::invalid_argument("Matrix dimensions must match in mat_sub_p64");
    }
    mat_sub(A, B, C);
}

inline void mat_hadamard_p32(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    if (A.rows() != B.rows() || A.cols() != B.cols() ||
        A.rows() != C.rows() || A.cols() != C.cols()) {
        throw std::invalid_argument("Matrix dimensions must match in mat_hadamard_p32");
    }
    mat_hadamard(A, B, C);
}

inline void mat_hadamard_p64(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C) {
    if (A.rows() != B.rows() || A.cols() != B.cols() ||
        A.rows() != C.rows() || A.cols() != C.cols()) {
        throw std::invalid_argument("Matrix dimensions must match in mat_hadamard_p64");
    }
    mat_hadamard(A, B, C);
}

inline void mat_div_p32(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    if (A.rows() != B.rows() || A.cols() != B.cols() ||
        A.rows() != C.rows() || A.cols() != C.cols()) {
        throw std::invalid_argument("Matrix dimensions must match in mat_div_p32");
    }
    mat_div(A, B, C);
}

inline void mat_div_p64(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C) {
    if (A.rows() != B.rows() || A.cols() != B.cols() ||
        A.rows() != C.rows() || A.cols() != C.cols()) {
        throw std::invalid_argument("Matrix dimensions must match in mat_div_p64");
    }
    mat_div(A, B, C);
}

// Scalar–Matrix operations
inline void mat_add_scalar_p32(float alpha, const Matrix<float>& A, Matrix<float>& C) {
    if (A.rows() != C.rows() || A.cols() != C.cols()) {
        throw std::invalid_argument("Matrix dimensions must match in mat_add_scalar_p32");
    }
    mat_add_scalar(alpha, A, C);
}

inline void mat_add_scalar_p64(double alpha, const Matrix<double>& A, Matrix<double>& C) {
    if (A.rows() != C.rows() || A.cols() != C.cols()) {
        throw std::invalid_argument("Matrix dimensions must match in mat_add_scalar_p64");
    }
    mat_add_scalar(alpha, A, C);
}

inline void mat_mul_scalar_p32(float alpha, Matrix<float>& A) {
    mat_mul_scalar(alpha, A);
}

inline void mat_mul_scalar_p64(double alpha, Matrix<double>& A) {
    mat_mul_scalar(alpha, A);
}

// Unary operations
inline void mat_neg_p32(const Matrix<float>& A, Matrix<float>& C) {
    if (A.rows() != C.rows() || A.cols() != C.cols()) {
        throw std::invalid_argument("Matrix dimensions must match in mat_neg_p32");
    }
    mat_neg(A, C);
}

inline void mat_neg_p64(const Matrix<double>& A, Matrix<double>& C) {
    if (A.rows() != C.rows() || A.cols() != C.cols()) {
        throw std::invalid_argument("Matrix dimensions must match in mat_neg_p64");
    }
    mat_neg(A, C);
}

inline void mat_abs_p32(const Matrix<float>& A, Matrix<float>& C) {
    if (A.rows() != C.rows() || A.cols() != C.cols()) {
        throw std::invalid_argument("Matrix dimensions must match in mat_abs_p32");
    }
    mat_abs(A, C);
}

inline void mat_abs_p64(const Matrix<double>& A, Matrix<double>& C) {
    if (A.rows() != C.rows() || A.cols() != C.cols()) {
        throw std::invalid_argument("Matrix dimensions must match in mat_abs_p64");
    }
    mat_abs(A, C);
}

} // namespace kernels
} // namespace mlib

#if defined(__AVX2__)
    #include "../../backends/avx2/matrix/elementwise.cpp"
#elif defined(__ARM_NEON)
    #include "../../backends/neon/matrix/elementwise.cpp"
#else
    #include "../../backends/scalar/matrix/elementwise.cpp"
#endif

#endif // MLIB_KERNELS_MATRIX_ELEMENTWISE_HPP
