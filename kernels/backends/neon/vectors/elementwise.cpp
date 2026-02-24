#include <mlib/vector.hpp>
#include <arm_neon.h>
#include <stdexcept>

namespace mlib {
namespace kernels {

// ────────────────────────────────────────────────
// float versions (128-bit = 4 elements)
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
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(px + i);
        float32x4_t vy = vld1q_f32(py + i);
        float32x4_t r  = vaddq_f32(vx, vy);
        vst1q_f32(pz + i, r);
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
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(px + i);
        float32x4_t vy = vld1q_f32(py + i);
        float32x4_t r  = vsubq_f32(vx, vy);
        vst1q_f32(pz + i, r);
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
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(px + i);
        float32x4_t vy = vld1q_f32(py + i);
        float32x4_t r  = vmulq_f32(vx, vy);
        vst1q_f32(pz + i, r);
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
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(px + i);
        float32x4_t vy = vld1q_f32(py + i);
        float32x4_t r  = vdivq_f32(vx, vy);
        vst1q_f32(pz + i, r);
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

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(px + i);
        float32x4_t r  = vabsq_f32(vx);
        vst1q_f32(py + i, r);
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

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(px + i);
        float32x4_t r  = vnegq_f32(vx);
        vst1q_f32(py + i, r);
    }
    for (; i < n; ++i) {
        py[i] = -px[i];
    }
}

// ────────────────────────────────────────────────
// double versions (128-bit = 2 elements)
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
    for (; i + 2 <= n; i += 2) {
        float64x2_t vx = vld1q_f64(px + i);
        float64x2_t vy = vld1q_f64(py + i);
        float64x2_t r  = vaddq_f64(vx, vy);
        vst1q_f64(pz + i, r);
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
    for (; i + 2 <= n; i += 2) {
        float64x2_t vx = vld1q_f64(px + i);
        float64x2_t vy = vld1q_f64(py + i);
        float64x2_t r  = vsubq_f64(vx, vy);
        vst1q_f64(pz + i, r);
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
    for (; i + 2 <= n; i += 2) {
        float64x2_t vx = vld1q_f64(px + i);
        float64x2_t vy = vld1q_f64(py + i);
        float64x2_t r  = vmulq_f64(vx, vy);
        vst1q_f64(pz + i, r);
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
    for (; i + 2 <= n; i += 2) {
        float64x2_t vx = vld1q_f64(px + i);
        float64x2_t vy = vld1q_f64(py + i);
        float64x2_t r  = vdivq_f64(vx, vy);
        vst1q_f64(pz + i, r);
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

    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t vx = vld1q_f64(px + i);
        float64x2_t r  = vabsq_f64(vx);
        vst1q_f64(py + i, r);
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

    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t vx = vld1q_f64(px + i);
        float64x2_t r  = vnegq_f64(vx);
        vst1q_f64(py + i, r);
    }
    for (; i < n; ++i) {
        py[i] = -px[i];
    }
}

} // namespace kernels
} // namespace mlib
