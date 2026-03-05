#include <arm_neon.h>
#include <cmath>

namespace mlib {
namespace kernels {

template<>
float norm_inf<float>(const float* __restrict x, size_t count) {
    float32x4_t max_abs = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t abs_v = vabsq_f32(vx);
        max_abs = vmaxq_f32(max_abs, abs_v);
    }

    float32x2_t pair = vmax_f32(vget_low_f32(max_abs), vget_high_f32(max_abs));
    float result = vget_lane_f32(vpmax_f32(pair, pair), 0);

    for (; i < count; ++i) {
        float abs_val = std::abs(x[i]);
        if (abs_val > result) result = abs_val;
    }

    return result;
}

template<>
double norm_inf<double>(const double* __restrict x, size_t count) {
    float64x2_t max_abs = vdupq_n_f64(0.0);

    size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        float64x2_t vx = vld1q_f64(x + i);
        float64x2_t abs_v = vabsq_f64(vx);
        max_abs = vmaxq_f64(max_abs, abs_v);
    }

    double result = vget_lane_f64(vmax_f64(vget_low_f64(max_abs), vget_high_f64(max_abs)), 0);

    for (; i < count; ++i) {
        double abs_val = std::abs(x[i]);
        if (abs_val > result) result = abs_val;
    }

    return result;
}

} // namespace kernels
} // namespace mlib
