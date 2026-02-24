#include <immintrin.h>

namespace mlib {
namespace kernels {

template<>
void scal<float>(float alpha, Vector<float>& x) {
    if (x.empty()) return;

    const size_t n = x.size();
    float* __restrict px = x.aligned_data();

    const __m256 valpha = _mm256_set1_ps(alpha);

    size_t i = 0;

    if (alpha == 0.0f) {
        __m256 vzero = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) {
            _mm256_store_ps(px + i, vzero);
        }
    } else {
        for (; i + 8 <= n; i += 8) {
            __m256 vx = _mm256_load_ps(px + i);
            __m256 r  = _mm256_mul_ps(vx, valpha);
            _mm256_store_ps(px + i, r);
        }
    }

    for (; i < n; ++i) {
        px[i] *= alpha;
    }
}

template<>
void scal<double>(double alpha, Vector<double>& x) {
    if (x.empty()) return;

    const size_t n = x.size();
    double* __restrict px = x.aligned_data();

    const __m256d valpha = _mm256_set1_pd(alpha);

    size_t i = 0;

    if (alpha == 0.0) {
        __m256d vzero = _mm256_setzero_pd();
        for (; i + 4 <= n; i += 4) {
            _mm256_store_pd(px + i, vzero);
        }
    } else {
        for (; i + 4 <= n; i += 4) {
            __m256d vx = _mm256_load_pd(px + i);
            __m256d r  = _mm256_mul_pd(vx, valpha);
            _mm256_store_pd(px + i, r);
        }
    }

    for (; i < n; ++i) {
        px[i] *= alpha;
    }
}

} // namespace kernels
} // namespace mlib
