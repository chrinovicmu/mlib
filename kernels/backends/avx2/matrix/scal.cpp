#include <immintrin.h>

namespace mlib {
namespace kernels {

template<>
void mscal<float>(float alpha, Matrix<float>& A) {
    const size_t rows = A.rows();
    const size_t cols = A.cols();
    const __m256 valpha = _mm256_set1_ps(alpha);

    for (size_t i = 0; i < rows; ++i) {
        float* __restrict a_row = A.row(i);

        size_t j = 0;
        for (; j + 8 <= cols; j += 8) {
            __m256 va = _mm256_load_ps(a_row + j);
            __m256 r  = _mm256_mul_ps(va, valpha);
            _mm256_store_ps(a_row + j, r);
        }
        for (; j < cols; ++j) {
            a_row[j] *= alpha;
        }
    }
}

template<>
void mscal<double>(double alpha, Matrix<double>& A) {
    const size_t rows = A.rows();
    const size_t cols = A.cols();
    const __m256d valpha = _mm256_set1_pd(alpha);

    for (size_t i = 0; i < rows; ++i) {
        double* __restrict a_row = A.row(i);

        size_t j = 0;
        for (; j + 4 <= cols; j += 4) {
            __m256d va = _mm256_load_pd(a_row + j);
            __m256d r  = _mm256_mul_pd(va, valpha);
            _mm256_store_pd(a_row + j, r);
        }
        for (; j < cols; ++j) {
            a_row[j] *= alpha;
        }
    }
}

} // namespace kernels
} // namespace mlib
