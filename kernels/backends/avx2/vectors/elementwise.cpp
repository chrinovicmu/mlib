#include <immintrin.h>

namespace mlib {
namespace kernels {

// ─── float ────────────────────────────────────────────────
template<>
void add<float>(const float* __restrict x, const float* __restrict y, float* __restrict z, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 vx = _mm256_load_ps(x + i);
        __m256 vy = _mm256_load_ps(y + i);
        __m256 r  = _mm256_add_ps(vx, vy);
        _mm256_store_ps(z + i, r);
    }
    for (; i < count; ++i) z[i] = x[i] + y[i];
}

template<>
void sub<float>(const float* __restrict x, const float* __restrict y, float* __restrict z, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 vx = _mm256_load_ps(x + i);
        __m256 vy = _mm256_load_ps(y + i);
        __m256 r  = _mm256_sub_ps(vx, vy);
        _mm256_store_ps(z + i, r);
    }
    for (; i < count; ++i) z[i] = x[i] - y[i];
}

template<>
void mul<float>(const float* __restrict x, const float* __restrict y, float* __restrict z, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 vx = _mm256_load_ps(x + i);
        __m256 vy = _mm256_load_ps(y + i);
        __m256 r  = _mm256_mul_ps(vx, vy);
        _mm256_store_ps(z + i, r);
    }
    for (; i < count; ++i) z[i] = x[i] * y[i];
}

template<>
void div<float>(const float* __restrict x, const float* __restrict y, float* __restrict z, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 vx = _mm256_load_ps(x + i);
        __m256 vy = _mm256_load_ps(y + i);
        __m256 r  = _mm256_div_ps(vx, vy);
        _mm256_store_ps(z + i, r);
    }
    for (; i < count; ++i) z[i] = x[i] / y[i];
}

template<>
void abs<float>(const float* __restrict x, float* __restrict y, size_t count) {
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 vx = _mm256_load_ps(x + i);
        __m256 r  = _mm256_andnot_ps(sign_mask, vx);
        _mm256_store_ps(y + i, r);
    }
    for (; i < count; ++i) y[i] = std::abs(x[i]);
}

template<>
void neg<float>(const float* __restrict x, float* __restrict y, size_t count) {
    const __m256 sign_flip = _mm256_set1_ps(-0.0f);
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 vx = _mm256_load_ps(x + i);
        __m256 r  = _mm256_xor_ps(vx, sign_flip);
        _mm256_store_ps(y + i, r);
    }
    for (; i < count; ++i) y[i] = -x[i];
}

// ─── double ───────────────────────────────────────────────
template<>
void add<double>(const double* __restrict x, const double* __restrict y, double* __restrict z, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256d vx = _mm256_load_pd(x + i);
        __m256d vy = _mm256_load_pd(y + i);
        __m256d r  = _mm256_add_pd(vx, vy);
        _mm256_store_pd(z + i, r);
    }
    for (; i < count; ++i) z[i] = x[i] + y[i];
}

template<>
void sub<double>(const double* __restrict x, const double* __restrict y, double* __restrict z, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256d vx = _mm256_load_pd(x + i);
        __m256d vy = _mm256_load_pd(y + i);
        __m256d r  = _mm256_sub_pd(vx, vy);
        _mm256_store_pd(z + i, r);
    }
    for (; i < count; ++i) z[i] = x[i] - y[i];
}

template<>
void mul<double>(const double* __restrict x, const double* __restrict y, double* __restrict z, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256d vx = _mm256_load_pd(x + i);
        __m256d vy = _mm256_load_pd(y + i);
        __m256d r  = _mm256_mul_pd(vx, vy);
        _mm256_store_pd(z + i, r);
    }
    for (; i < count; ++i) z[i] = x[i] * y[i];
}

template<>
void div<double>(const double* __restrict x, const double* __restrict y, double* __restrict z, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256d vx = _mm256_load_pd(x + i);
        __m256d vy = _mm256_load_pd(y + i);
        __m256d r  = _mm256_div_pd(vx, vy);
        _mm256_store_pd(z + i, r);
    }
    for (; i < count; ++i) z[i] = x[i] / y[i];
}

template<>
void abs<double>(const double* __restrict x, double* __restrict y, size_t count) {
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256d vx = _mm256_load_pd(x + i);
        __m256d r  = _mm256_andnot_pd(sign_mask, vx);
        _mm256_store_pd(y + i, r);
    }
    for (; i < count; ++i) y[i] = std::abs(x[i]);
}

template<>
void neg<double>(const double* __restrict x, double* __restrict y, size_t count) {
    const __m256d sign_flip = _mm256_set1_pd(-0.0);
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256d vx = _mm256_load_pd(x + i);
        __m256d r  = _mm256_xor_pd(vx, sign_flip);
        _mm256_store_pd(y + i, r);
    }
    for (; i < count; ++i) y[i] = -x[i];
}

} // namespace kernels
} // namespace mlib
