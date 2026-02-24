
#include <immintrin.h>

namespace mlib {
namespace kernels {

template<>
void copy<float>(const Vector<float>& x, Vector<float>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector dimensions must match in copy");
    }
    if (x.empty()) {
        return;
    }

    const size_t n = x.size();
    const float* __restrict px = x.aligned_data();
    float* __restrict py = y.aligned_data();

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_load_ps(px + i);
        _mm256_store_ps(py + i, vx);
    }

    for (; i < n; ++i) {
        py[i] = px[i];
    }
}

template<>
void copy<double>(const Vector<double>& x, Vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector dimensions must match in copy");
    }
    if (x.empty()) {
        return;
    }

    const size_t n = x.size();
    const double* __restrict px = x.aligned_data();
    double* __restrict py = y.aligned_data();

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vx = _mm256_load_pd(px + i);
        _mm256_store_pd(py + i, vx);
    }

    for (; i < n; ++i) {
        py[i] = px[i];
    }
}

} // namespace kernels
} // namespace mlib
