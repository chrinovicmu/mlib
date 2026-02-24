#include <immintrin.h>

namespace mlib {
namespace kernels {

template<>
void swap<float>(Vector<float>& x, Vector<float>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector dimensions must match in swap");
    }
    if (x.empty()) {
        return;
    }

    const size_t n = x.size();
    float* __restrict px = x.aligned_data();
    float* __restrict py = y.aligned_data();

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_load_ps(px + i);
        __m256 vy = _mm256_load_ps(py + i);
        _mm256_store_ps(px + i, vy);
        _mm256_store_ps(py + i, vx);
    }

    for (; i < n; ++i) {
        float tmp   = px[i];
        px[i]       = py[i];
        py[i]       = tmp;
    }
}

template<>
void swap<double>(Vector<double>& x, Vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector dimensions must match in swap");
    }
    if (x.empty()) {
        return;
    }

    const size_t n = x.size();
    double* __restrict px = x.aligned_data();
    double* __restrict py = y.aligned_data();

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vx = _mm256_load_pd(px + i);
        __m256d vy = _mm256_load_pd(py + i);
        _mm256_store_pd(px + i, vy);
        _mm256_store_pd(py + i, vx);
    }

    for (; i < n; ++i) {
        double tmp  = px[i];
        px[i]       = py[i];
        py[i]       = tmp;
    }
}

} // namespace kernels
} // namespace mlib
