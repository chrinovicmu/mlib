#include <arm_neon.h>

namespace mlib {
namespace kernels {

template<>
void scal<float>(float alpha, Vector<float>& x) {
    if (x.empty()) return;

    const size_t n = x.size();
    float* __restrict px = x.aligned_data();

    const float32x4_t valpha = vdupq_n_f32(alpha);

    size_t i = 0;

    if (alpha == 0.0f) {
        float32x4_t vzero = vdupq_n_f32(0.0f);
        for (; i + 4 <= n; i += 4) {
            vst1q_f32(px + i, vzero);
        }
    } else {
        for (; i + 4 <= n; i += 4) {
            float32x4_t vx = vld1q_f32(px + i);
            float32x4_t r  = vmulq_f32(vx, valpha);
            vst1q_f32(px + i, r);
        }
    }

    for (; i < n; ++i) {
        px[i] *= alpha;
    }
}

template<>
void scal<double>(double alpha, Vector<double>& x) {
    if (x.empty()) return;

    const size_t n = x.size();
    double* __restrict px = x.aligned_data();

    const float64x2_t valpha = vdupq_n_f64(alpha);

    size_t i = 0;

    if (alpha == 0.0) {
        float64x2_t vzero = vdupq_n_f64(0.0);
        for (; i + 2 <= n; i += 2) {
            vst1q_f64(px + i, vzero);
        }
    } else {
        for (; i + 2 <= n; i += 2) {
            float64x2_t vx = vld1q_f64(px + i);
            float64x2_t r  = vmulq_f64(vx, valpha);
            vst1q_f64(px + i, r);
        }
    }

    for (; i < n; ++i) {
        px[i] *= alpha;
    }
}

} // namespace kernels
} // namespace mlib
