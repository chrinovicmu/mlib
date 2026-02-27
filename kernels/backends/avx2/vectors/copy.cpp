
#include <immintrin.h>

namespace mlib {
namespace kernels {

template<>
void copy<float>(const float* __restrict src, float* __restrict dst, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 v = _mm256_load_ps(src + i);
        _mm256_store_ps(dst + i, v);
    }

    for (; i < count; ++i) {
        dst[i] = src[i];
    }
}

template<>
void copy<double>(const double* __restrict src, double* __restrict dst, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256d v = _mm256_load_pd(src + i);
        _mm256_store_pd(dst + i, v);
    }

    for (; i < count; ++i) {
        dst[i] = src[i];
    }
}

} // namespace kernels
} // namespace mlib
