
#include <immintrin.h>

namespace mlib {
namespace kernels {

template<>
void scal<float>(float alpha, float* __restrict x, size_t count) {
    const __m256 valpha = _mm256_set1_ps(alpha);

    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 vx = _mm256_load_ps(x + i);
        __m256 r  = _mm256_mul_ps(vx, valpha);
        _mm256_store_ps(x + i, r);
    }

    for (; i < count; ++i) {
        x[i] *= alpha;
    }
}

template<>
void scal<double>(double alpha, double* __restrict x, size_t count) {
    const __m256d valpha = _mm256_set1_pd(alpha);

    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256d vx = _mm256_load_pd(x + i);
        __m256d r  = _mm256_mul_pd(vx, valpha);
        _mm256_store_pd(x + i, r);
    }

    for (; i < count; ++i) {
        x[i] *= alpha;
    }
}

} // namespace kernels
} // namespace mlib
