#include <immintrin.h>

namespace mlib {
namespace kernels {

template<>
float norm_inf<float>(const float* __restrict x, size_t count) {
    __m256 max_abs = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 vx = _mm256_load_ps(x + i);
        __m256 abs_v = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vx);
        max_abs = _mm256_max_ps(max_abs, abs_v);
    }

    __m128 lo = _mm256_castps256_ps128(max_abs);
    __m128 hi = _mm256_extractf128_ps(max_abs, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(2,3,0,1)));
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(1,0,3,2)));
    float result = _mm_cvtss_f32(lo);

    for (; i < count; ++i) {
        float abs_val = std::abs(x[i]);
        if (abs_val > result) result = abs_val;
    }

    return result;
}

template<>
double norm_inf<double>(const double* __restrict x, size_t count) {
    __m256d max_abs = _mm256_setzero_pd();

    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256d vx = _mm256_load_pd(x + i);
        __m256d abs_v = _mm256_andnot_pd(_mm256_set1_pd(-0.0), vx);
        max_abs = _mm256_max_pd(max_abs, abs_v);
    }

    __m128d lo = _mm256_castpd256_pd128(max_abs);
    __m128d hi = _mm256_extractf128_pd(max_abs, 1);
    lo = _mm_max_pd(lo, hi);
    lo = _mm_max_pd(lo, _mm_shuffle_pd(lo, lo, _MM_SHUFFLE2(1,0)));
    double result = _mm_cvtsd_f64(lo);

    for (; i < count; ++i) {
        double abs_val = std::abs(x[i]);
        if (abs_val > result) result = abs_val;
    }

    return result;
}

} // namespace kernels
} // namespace mlib
