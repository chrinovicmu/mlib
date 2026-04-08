// kernels/backends/avx2/matrix/copy.cpp

#include <immintrin.h>

namespace mlib {
namespace kernels {

template<>
void copy<float>(const Matrix<float>& src, Matrix<float>& dst) {
    const size_t rows = src.rows();
    const size_t cols = src.cols();

    for (size_t i = 0; i < rows; ++i) {
        const float* __restrict src_row = src.row(i);
        float* __restrict dst_row = dst.row(i);

        size_t j = 0;
        for (; j + 8 <= cols; j += 8) {
            __m256 v = _mm256_load_ps(src_row + j);
            _mm256_store_ps(dst_row + j, v);
        }
        for (; j < cols; ++j) {
            dst_row[j] = src_row[j];
        }
    }
}

template<>
void copy<double>(const Matrix<double>& src, Matrix<double>& dst) {
    const size_t rows = src.rows();
    const size_t cols = src.cols();

    for (size_t i = 0; i < rows; ++i) {
        const double* __restrict src_row = src.row(i);
        double* __restrict dst_row = dst.row(i);

        size_t j = 0;
        for (; j + 4 <= cols; j += 4) {
            __m256d v = _mm256_load_pd(src_row + j);
            _mm256_store_pd(dst_row + j, v);
        }
        for (; j < cols; ++j) {
            dst_row[j] = src_row[j];
        }
    }
}

} 
}
