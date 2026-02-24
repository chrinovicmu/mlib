
#include <immintrin.h>

namespace mlib {
namespace kernels {

template<>
float dot<float>(const float* __restrict x, const float* __restrict y, size_t count) {
    __m256 sum = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 vx = _mm256_load_ps(x + i);
        __m256 vy = _mm256_load_ps(y + i);
        sum = _mm256_fmadd_ps(vx, vy, sum);
    }

    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float result = _mm_cvtss_f32(lo);

    for (; i < count; ++i) {
        result += x[i] * y[i];
    }

    return result;
}

template<>
double dot<double>(const double* __restrict x, const double* __restrict y, size_t count) {
    __m256d sum = _mm256_setzero_pd();

    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256d vx = _mm256_load_pd(x + i);
        __m256d vy = _mm256_load_pd(y + i);
        sum = _mm256_fmadd_pd(vx, vy, sum);
    }

    __m128d lo = _mm256_castpd256_pd128(sum);
    __m128d hi = _mm256_extractf128_pd(sum, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_hadd_pd(lo, lo);
    double result = _mm_cvtsd_f64(lo);

    for (; i < count; ++i) {
        result += x[i] * y[i];
    }

    return result;
}

} // namespace kernels
} // namespace mlib
