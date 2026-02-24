#ifndef MLIB_KERNELS_NEON_AXPY_HPP
#define MLIB_KERNELS_NEON_AXPY_HPP

#include <lin/vector.hpp>
#include <arm_neon.h>
#include <stdexcept>

namespace mlibs {
namespace kernels {
namespace neon {

template<>
inline void axpy<float>(float alpha, const Vector<float>& x, Vector<float>& y) {

    if (x.size() != y.size()) throw std::invalid_argument("size mismatch");
    if (x.empty()) return;

    const size_t n = x.size();
    const float* __restrict px = x.aligned_data();
    float* __restrict py = y.aligned_data();

    const float32x4_t valpha = vdupq_n_f32(alpha);

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(px + i);
        float32x4_t vy = vld1q_f32(py + i);
        float32x4_t r  = vfmaq_f32(vy, vx, valpha);
        vst1q_f32(py + i, r);
    }

    for (; i < n; ++i) {
        py[i] = alpha * px[i] + py[i];
    }
}

template<>
inline void axpy<double>(double alpha, const Vector<double>& x, Vector<double>& y) {

    if (x.size() != y.size()) throw std::invalid_argument("size mismatch");
    if (x.empty()) return;

    const size_t n = x.size();
    const double* __restrict px = x.aligned_data();
    double* __restrict py = y.aligned_data();

    const float64x2_t valpha = vdupq_n_f64(alpha);

    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t vx = vld1q_f64(px + i);
        float64x2_t vy = vld1q_f64(py + i);
        float64x2_t r  = vfmaq_f64(vy, vx, valpha);
        vst1q_f64(py + i, r);
    }

    for (; i < n; ++i) {
        py[i] = alpha * px[i] + py[i];
    }
}

inline void saxpy(float alpha, const Vector<float>& x, Vector<float>& y) {
    axpy(alpha, x, y);
}

inline void daxpy(double alpha, const Vector<double>& x, Vector<double>& y) {
    axpy(alpha, x, y);
}

} // namespace neon
} // namespace kernels
} // namespace lin

#endif
