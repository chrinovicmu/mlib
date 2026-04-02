#ifndef MLIB_KERNELS_MATRIX_COPY_HPP
#define MLIB_KERNELS_MATRIX_COPY_HPP

#include <mlib/matrix.hpp>
#include <cstddef>
#include <stdexcept>

namespace mlib {
namespace kernels {

template<typename T>
void copy(const Matrix<T>& src, Matrix<T>& dst);

inline void mat_copy_p32(const Matrix<float>& src, Matrix<float>& dst) {
    if (src.rows() != dst.rows() || src.cols() != dst.cols()) {
        throw std::invalid_argument("Matrix dimensions must match in mat_copy_p32");
    }
    copy(src, dst);
}

inline void mat_copy_p64(const Matrix<double>& src, Matrix<double>& dst) {
    if (src.rows() != dst.rows() || src.cols() != dst.cols()) {
        throw std::invalid_argument("Matrix dimensions must match in mat_copy_p64");
    }
    copy(src, dst);
}

} // namespace kernels
} // namespace mlib

#if defined(__AVX2__)
    #include "../../backends/avx2/matrix/copy.cpp"
#elif defined(__ARM_NEON)
    #include "../../backends/neon/matrix/copy.cpp"
#else
    #include "../../backends/scalar/matrix/copy.cpp"
#endif

#endif
