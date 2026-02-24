// backends/neon/vectors/swap.cpp

#include <arm_neon.h>

namespace mlib {
namespace kernels {

template<>
void swap<float>(float* __restrict x, float* __restrict y, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vy = vld1q_f32(y + i);
        vst1q_f32(x + i, vy);
        vst1q_f32(y + i, vx);
    }

    for (; i < count; ++i) {
        float tmp = x[i];
        x[i] = y[i];
        y[i] = tmp;
    }
}

template<>
void swap<double>(double* __restrict x, double* __restrict y, size_t count) {
    size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        float64x2_t vx = vld1q_f64(x + i);
        float64x2_t vy = vld1q_f64(y + i);
        vst1q_f64(x + i, vy);
        vst1q_f64(y + i, vx);
    }

    for (; i < count; ++i) {
        double tmp = x[i];
        x[i] = y[i];
        y[i] = tmp;
    }
}

} // namespace kernels
} // namespace mlib
