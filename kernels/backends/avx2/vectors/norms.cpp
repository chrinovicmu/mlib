#include <immintrin.h>

namespace mlib {
namespace kernels {

template<>
float l1_norm<float>(const float* __restrict x, size_t count) {
    __m256 abs_sum = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 vx = _mm256_load_ps(x + i);
        __m256 abs_v = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vx);
        abs_sum = _mm256_add_ps(abs_sum, abs_v);
    }

    __m128 lo = _mm256_castps256_ps128(abs_sum);
    __m128 hi = _mm256_extractf128_ps(abs_sum, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float result = _mm_cvtss_f32(lo);

    for (; i < count; ++i) {
        result += std::abs(x[i]);
    }

    return result;
}

template<>
double l1_norm<double>(const double* __restrict x, size_t count) {
    __m256d abs_sum = _mm256_setzero_pd();

    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256d vx = _mm256_load_pd(x + i);
        __m256d abs_v = _mm256_andnot_pd(_mm256_set1_pd(-0.0), vx);
        abs_sum = _mm256_add_pd(abs_sum, abs_v);
    }

    __m128d lo = _mm256_castpd256_pd128(abs_sum);
    __m128d hi = _mm256_extractf128_pd(abs_sum, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_hadd_pd(lo, lo);
    double result = _mm_cvtsd_f64(lo);

    for (; i < count; ++i) {
        result += std::abs(x[i]);
    }

    return result;
}

template<>
float l2_norm<float>(const float* __restrict x, size_t count) {
    __m256 sum_sq = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 vx = _mm256_load_ps(x + i);
        __m256 sq = _mm256_mul_ps(vx, vx);
        sum_sq = _mm256_add_ps(sum_sq, sq);
    }

    __m128 lo = _mm256_castps256_ps128(sum_sq);
    __m128 hi = _mm256_extractf128_ps(sum_sq, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum_squares = _mm_cvtss_f32(lo);

    for (; i < count; ++i) {
        sum_squares += x[i] * x[i];
    }

    return std::sqrt(sum_squares);
}

template<>
double l2_norm<double>(const double* __restrict x, size_t count) {
    __m256d sum_sq = _mm256_setzero_pd();

    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256d vx = _mm256_load_pd(x + i);
        __m256d sq = _mm256_mul_pd(vx, vx);
        sum_sq = _mm256_add_pd(sum_sq, sq);
    }

    __m128d lo = _mm256_castpd256_pd128(sum_sq);
    __m128d hi = _mm256_extractf128_pd(sum_sq, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_hadd_pd(lo, lo);
    double sum_squares = _mm_cvtsd_f64(lo);

    for (; i < count; ++i) {
        sum_squares += x[i] * x[i];
    }

    return std::sqrt(sum_squares);
}

} // namespace kernels
} // namespace mlib
