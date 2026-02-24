#include <immintrin.h>

namespace mlib {
namespace kernels {

template<>
void axpy<float>(float alpha, const float* __restrict x, float* __restrict y, size_t count) {
    const __m256 valpha = _mm256_set1_ps(alpha);

    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 vx = _mm256_load_ps(x + i);
        __m256 vy = _mm256_load_ps(y + i);
        __m256 r  = _mm256_fmadd_ps(vx, valpha, vy);
        _mm256_store_ps(y + i, r);
    }

    for (; i < count; ++i) {
        y[i] += alpha * x[i];
    }
}

template<>
void axpy<double>(double alpha, const double* __restrict x, double* __restrict y, size_t count) {
    const __m256d valpha = _mm256_set1_pd(alpha);

    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256d vx = _mm256_load_pd(x + i);
        __m256d vy = _mm256_load_pd(y + i);
        __m256d r  = _mm256_fmadd_pd(vx, valpha, vy);
        _mm256_store_pd(y + i, r);
    }

    for (; i < count; ++i) {
        y[i] += alpha * x[i];
    }
}

} // namespace kernels
} // namespace mlib
