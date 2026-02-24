
#include <arm_neon.h>

namespace mlib {
namespace kernels {

template<>
void axpy<float>(float alpha, const float* __restrict x, float* __restrict y, size_t count) {
    const float32x4_t valpha = vdupq_n_f32(alpha);

    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vy = vld1q_f32(y + i);
        float32x4_t r  = vfmaq_f32(vy, vx, valpha);
        vst1q_f32(y + i, r);
    }

    for (; i < count; ++i) {
        y[i] += alpha * x[i];
    }
}

template<>
void axpy<double>(double alpha, const double* __restrict x, double* __restrict y, size_t count) {
    const float64x2_t valpha = vdupq_n_f64(alpha);

    size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        float64x2_t vx = vld1q_f64(x + i);
        float64x2_t vy = vld1q_f64(y + i);
        float64x2_t r  = vfmaq_f64(vy, vx, valpha);
        vst1q_f64(y + i, r);
    }

    for (; i < count; ++i) {
        y[i] += alpha * x[i];
    }
}

} // namespace kernels
} // namespace mlib
