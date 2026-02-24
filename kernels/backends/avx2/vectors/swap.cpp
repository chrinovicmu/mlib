// backends/avx2/vectors/swap.cpp

#include <immintrin.h>

namespace mlib {
namespace kernels {

template<>
void swap<float>(float* __restrict x, float* __restrict y, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 vx = _mm256_load_ps(x + i);
        __m256 vy = _mm256_load_ps(y + i);
        _mm256_store_ps(x + i, vy);
        _mm256_store_ps(y + i, vx);
    }

    for (; i < count; ++i) {
        float tmp = x[i];
        x[i]      = y[i];
        y[i]      = tmp;
    }
}

template<>
void swap<double>(double* __restrict x, double* __restrict y, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256d vx = _mm256_load_pd(x + i);
        __m256d vy = _mm256_load_pd(y + i);
        _mm256_store_pd(x + i, vy);
        _mm256_store_pd(y + i, vx);
    }

    for (; i < count; ++i) {
        double tmp = x[i];
        x[i]       = y[i];
        y[i]       = tmp;
    }
}

} // namespace kernels
} // namespace mlib
