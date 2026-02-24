
#include <mlib/vector.hpp>
#include <immintrin.h>
#include <stdexcept>

namespace mlib {
namespace kernels {

// ────────────────────────────────────────────────
// float versions (256-bit = 8 elements)
// ────────────────────────────────────────────────

template<>
void add<float>(const Vector<float>& x, const Vector<float>& y, Vector<float>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Vector sizes must match in add");
    }
    const size_t n = x.size();
    if (n == 0) return;

    const float* __restrict px = x.aligned_data();
    const float* __restrict py = y.aligned_data();
    float* __restrict pz = z.aligned_data();

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_load_ps(px + i);
        __m256 vy = _mm256_load_ps(py + i);
        __m256 r  = _mm256_add_ps(vx, vy);
        _mm256_store_ps(pz + i, r);
    }
    for (; i < n; ++i) {
        pz[i] = px[i] + py[i];
    }
}

template<>
void sub<float>(const Vector<float>& x, const Vector<float>& y, Vector<float>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Vector sizes must match in sub");
    }
    const size_t n = x.size();
    if (n == 0) return;

    const float* __restrict px = x.aligned_data();
    const float* __restrict py = y.aligned_data();
    float* __restrict pz = z.aligned_data();

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_load_ps(px + i);
        __m256 vy = _mm256_load_ps(py + i);
        __m256 r  = _mm256_sub_ps(vx, vy);
        _mm256_store_ps(pz + i, r);
    }
    for (; i < n; ++i) {
        pz[i] = px[i] - py[i];
    }
}

template<>
void mul<float>(const Vector<float>& x, const Vector<float>& y, Vector<float>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Vector sizes must match in mul");
    }
    const size_t n = x.size();
    if (n == 0) return;

    const float* __restrict px = x.aligned_data();
    const float* __restrict py = y.aligned_data();
    float* __restrict pz = z.aligned_data();

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_load_ps(px + i);
        __m256 vy = _mm256_load_ps(py + i);
        __m256 r  = _mm256_mul_ps(vx, vy);
        _mm256_store_ps(pz + i, r);
    }
    for (; i < n; ++i) {
        pz[i] = px[i] * py[i];
    }
}

template<>
void div<float>(const Vector<float>& x, const Vector<float>& y, Vector<float>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Vector sizes must match in div");
    }
    const size_t n = x.size();
    if (n == 0) return;

    const float* __restrict px = x.aligned_data();
    const float* __restrict py = y.aligned_data();
    float* __restrict pz = z.aligned_data();

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_load_ps(px + i);
        __m256 vy = _mm256_load_ps(py + i);
        __m256 r  = _mm256_div_ps(vx, vy);
        _mm256_store_ps(pz + i, r);
    }
    for (; i < n; ++i) {
        pz[i] = px[i] / py[i];
    }
}

template<>
void abs<float>(const Vector<float>& x, Vector<float>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector sizes must match in abs");
    }
    const size_t n = x.size();
    if (n == 0) return;

    const float* __restrict px = x.aligned_data();
    float* __restrict py = y.aligned_data();

    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_load_ps(px + i);
        __m256 r  = _mm256_andnot_ps(sign_mask, vx);  // clear sign bit → |x|
        _mm256_store_ps(py + i, r);
    }
    for (; i < n; ++i) {
        py[i] = std::fabs(px[i]);
    }
}

template<>
void neg<float>(const Vector<float>& x, Vector<float>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector sizes must match in neg");
    }
    const size_t n = x.size();
    if (n == 0) return;

    const float* __restrict px = x.aligned_data();
    float* __restrict py = y.aligned_data();

    const __m256 sign_flip = _mm256_set1_ps(-0.0f);

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_load_ps(px + i);
        __m256 r  = _mm256_xor_ps(vx, sign_flip);  // flip sign bit
        _mm256_store_ps(py + i, r);
    }
    for (; i < n; ++i) {
        py[i] = -px[i];
    }
}

// ────────────────────────────────────────────────
// double versions (256-bit = 4 elements)
// ────────────────────────────────────────────────

template<>
void add<double>(const Vector<double>& x, const Vector<double>& y, Vector<double>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Vector sizes must match in add");
    }
    const size_t n = x.size();
    if (n == 0) return;

    const double* __restrict px = x.aligned_data();
    const double* __restrict py = y.aligned_data();
    double* __restrict pz = z.aligned_data();

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vx = _mm256_load_pd(px + i);
        __m256d vy = _mm256_load_pd(py + i);
        __m256d r  = _mm256_add_pd(vx, vy);
        _mm256_store_pd(pz + i, r);
    }
    for (; i < n; ++i) {
        pz[i] = px[i] + py[i];
    }
}

template<>
void sub<double>(const Vector<double>& x, const Vector<double>& y, Vector<double>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Vector sizes must match in sub");
    }
    const size_t n = x.size();
    if (n == 0) return;

    const double* __restrict px = x.aligned_data();
    const double* __restrict py = y.aligned_data();
    double* __restrict pz = z.aligned_data();

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vx = _mm256_load_pd(px + i);
        __m256d vy = _mm256_load_pd(py + i);
        __m256d r  = _mm256_sub_pd(vx, vy);
        _mm256_store_pd(pz + i, r);
    }
    for (; i < n; ++i) {
        pz[i] = px[i] - py[i];
    }
}

template<>
void mul<double>(const Vector<double>& x, const Vector<double>& y, Vector<double>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Vector sizes must match in mul");
    }
    const size_t n = x.size();
    if (n == 0) return;

    const double* __restrict px = x.aligned_data();
    const double* __restrict py = y.aligned_data();
    double* __restrict pz = z.aligned_data();

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vx = _mm256_load_pd(px + i);
        __m256d vy = _mm256_load_pd(py + i);
        __m256d r  = _mm256_mul_pd(vx, vy);
        _mm256_store_pd(pz + i, r);
    }
    for (; i < n; ++i) {
        pz[i] = px[i] * py[i];
    }
}

template<>
void div<double>(const Vector<double>& x, const Vector<double>& y, Vector<double>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Vector sizes must match in div");
    }
    const size_t n = x.size();
    if (n == 0) return;

    const double* __restrict px = x.aligned_data();
    const double* __restrict py = y.aligned_data();
    double* __restrict pz = z.aligned_data();

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vx = _mm256_load_pd(px + i);
        __m256d vy = _mm256_load_pd(py + i);
        __m256d r  = _mm256_div_pd(vx, vy);
        _mm256_store_pd(pz + i, r);
    }
    for (; i < n; ++i) {
        pz[i] = px[i] / py[i];
    }
}

template<>
void abs<double>(const Vector<double>& x, Vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector sizes must match in abs");
    }
    const size_t n = x.size();
    if (n == 0) return;

    const double* __restrict px = x.aligned_data();
    double* __restrict py = y.aligned_data();

    const __m256d sign_mask = _mm256_set1_pd(-0.0);

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vx = _mm256_load_pd(px + i);
        __m256d r  = _mm256_andnot_pd(sign_mask, vx);
        _mm256_store_pd(py + i, r);
    }
    for (; i < n; ++i) {
        py[i] = std::fabs(px[i]);
    }
}

template<>
void neg<double>(const Vector<double>& x, Vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector sizes must match in neg");
    }
    const size_t n = x.size();
    if (n == 0) return;

    const double* __restrict px = x.aligned_data();
    double* __restrict py = y.aligned_data();

    const __m256d sign_flip = _mm256_set1_pd(-0.0);

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vx = _mm256_load_pd(px + i);
        __m256d r  = _mm256_xor_pd(vx, sign_flip);
        _mm256_store_pd(py + i, r);
    }
    for (; i < n; ++i) {
        py[i] = -px[i];
    }
}

} // namespace kernels
} // namespace mlib
