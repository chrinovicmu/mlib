
#include <arm_neon.h>

namespace mlib {
namespace kernels {

template<>
void scal<float>(float alpha, float* __restrict x, size_t count) {
    const float32x4_t valpha = vdupq_n_f32(alpha);

    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t r  = vmulq_f32(vx, valpha);
        vst1q_f32(x + i, r);
    }

    for (; i < count; ++i) {
        x[i] *= alpha;
    }
}

template<>
void scal<double>(double alpha, double* __restrict x, size_t count) {
    const float64x2_t valpha = vdupq_n_f64(alpha);

    size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        float64x2_t vx = vld1q_f64(x + i);
        float64x2_t r  = vmulq_f64(vx, valpha);
        vst1q_f64(x + i, r);
    }

    for (; i < count; ++i) {
        x[i] *= alpha;
    }
}

} // namespace kernels
} // namespace mlib
