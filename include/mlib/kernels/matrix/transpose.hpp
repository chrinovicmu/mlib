#ifndef MLIB_KERNELS_MATRIX_TRANSPOSE_HPP
#define MLIB_KERNELS_MATRIX_TRANSPOSE_HPP

#include <mlib/matrix.hpp>
#include <cstddef>
#include <stdexcept>

namespace mlib {
namespace kernels {

template<typename T>
void transpose(const Matrix<T>& src, Matrix<T>& dst);

inline void mat_transpose_p32(const Matrix<float>& src, Matrix<float>& dst) {
    if (src.rows() != dst.cols() || src.cols() != dst.rows()) {
        throw std::invalid_argument("Matrix dimensions incompatible for transpose (dst must be src^T)");
    }
    transpose(src, dst);
}

inline void mat_transpose_p64(const Matrix<double>& src, Matrix<double>& dst) {
    if (src.rows() != dst.cols() || src.cols() != dst.rows()) {
        throw std::invalid_argument("Matrix dimensions incompatible for transpose (dst must be src^T)");
    }
    transpose(src, dst);
}

} 
} 

#if defined(__AVX2__)
    #include "../../backends/avx2/matrix/transpose.cpp"
#elif defined(__ARM_NEON)
    #include "../../backends/neon/matrix/transpose.cpp"
#else
    #include "../../backends/scalar/matrix/transpose.cpp"
#endif

#endif
