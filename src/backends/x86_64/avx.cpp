
#include <immintrin.h>

#include <cmath>
#include <cstddef>  

#include "mlib/kernels/decl.hpp"

namespace mlib::kernels::avx {

struct simd_f32 {
    using element_type = float;
    using vec = __m256;
    static constexpr std::size_t width = 8;  // 256 bits / 32 bits

    static vec zero() { return _mm256_setzero_ps(); }
    static vec broadcast(float a) { return _mm256_set1_ps(a); }

    static vec loadu(const float* p) { return _mm256_loadu_ps(p); }
    static void storeu(float* p, vec v) { _mm256_storeu_ps(p, v); }

    static vec add(vec a, vec b) { return _mm256_add_ps(a, b); }
    static vec mul(vec a, vec b) { return _mm256_mul_ps(a, b); }

    static vec mul_add(vec a, vec b, vec c) {
        auto product = _mm256_mul_ps(a, b);
        return _mm256_add_ps(product, c);
    }

    static vec abs(vec v) {
        auto sign_mask = _mm256_set1_ps(-0.0f);
        return _mm256_andnot_ps(sign_mask, v);
    }

    // Horizontal sum of the 8 lanes.
    static float hsum(vec v) {
        const __m128 lo = _mm256_castps256_ps128(v);
        const __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 s = _mm_add_ps(lo, hi);

        auto upper = _mm_movehl_ps(s, s);
        s = _mm_add_ps(s, upper);

        // 0x55 is the shuffle selector 01|01|01|01 -- every lane takes
        // source lane 1. add_ss adds only lane 0, which is all we still
        // need. Lane 0 now holds the total.
        auto lane1 = _mm_shuffle_ps(s, s, 0x55);
        s = _mm_add_ss(s, lane1);

        return _mm_cvtss_f32(s);
    }
};

struct simd_f64 {
    using element_type = double;
    using vec = __m256d;
    static constexpr std::size_t width = 4;  // 256 bits / 64 bits

    static vec zero() { return _mm256_setzero_pd(); }
    static vec broadcast(double a) { return _mm256_set1_pd(a); }
    static vec loadu(const double* p) { return _mm256_loadu_pd(p); }
    static void storeu(double* p, vec v) { _mm256_storeu_pd(p, v); }
    static vec add(vec a, vec b) { return _mm256_add_pd(a, b); }
    static vec mul(vec a, vec b) { return _mm256_mul_pd(a, b); }

    static vec mul_add(vec a, vec b, vec c) {
        auto product = _mm256_mul_pd(a, b);
        return _mm256_add_pd(product, c);
    }

    static vec abs(vec v) {
        auto sign_mask = _mm256_set1_pd(-0.0);
        return _mm256_andnot_pd(sign_mask, v);
    }

    static double hsum(vec v) {
        // Halves: {v0+v2, v1+v3}.
        const __m128d lo = _mm256_castpd256_pd128(v);
        const __m128d hi = _mm256_extractf128_pd(v, 1);
        __m128d s = _mm_add_pd(lo, hi);

        // unpackhi(s, s) puts lane 1 into lane 0; add_sd sums only lane 0.
        auto lane1 = _mm_unpackhi_pd(s, s);
        s = _mm_add_sd(s, lane1);

        return _mm_cvtsd_f64(s);
    }
};

#include "../simd_kernels.inc"

}  // namespace mlib::kernels::avx
