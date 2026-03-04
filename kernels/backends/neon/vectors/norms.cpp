#include <arm_neon.h>
#include <cmath>

namespace mlib {
namespace kernels {

template<>
float l1_norm<float>(const float* __restrict x, size_t count) {
    float32x4_t abs_sum = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t abs_v = vabsq_f32(vx);
        abs_sum = vaddq_f32(abs_sum, abs_v);
    }

    float32x2_t pair = vadd_f32(vget_low_f32(abs_sum), vget_high_f32(abs_sum));
    float32x2_t total = vpadd_f32(pair, pair);
    float result = vget_lane_f32(total, 0);

    for (; i < count; ++i) {
        result += std::abs(x[i]);
    }

    return result;
}

template<>
double l1_norm<double>(const double* __restrict x, size_t count) {
    float64x2_t abs_sum = vdupq_n_f64(0.0);

    size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        float64x2_t vx = vld1q_f64(x + i);
        float64x2_t abs_v = vabsq_f64(vx);
        abs_sum = vaddq_f64(abs_sum, abs_v);
    }

    double result = vget_lane_f64(vadd_f64(vget_low_f64(abs_sum), vget_high_f64(abs_sum)), 0);

    for (; i < count; ++i) {
        result += std::abs(x[i]);
    }

    return result;
}

template<>
float l2_norm<float>(const float* __restrict x, size_t count) {
    float32x4_t sum_sq = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t sq = vmulq_f32(vx, vx);
        sum_sq = vaddq_f32(sum_sq, sq);
    }

    float32x2_t pair = vadd_f32(vget_low_f32(sum_sq), vget_high_f32(sum_sq));
    float32x2_t total = vpadd_f32(pair, pair);
    float sum_squares = vget_lane_f32(total, 0);

    for (; i < count; ++i) {
        sum_squares += x[i] * x[i];
    }

    return std::sqrt(sum_squares);
}

template<>
double l2_norm<double>(const double* __restrict x, size_t count) {
    float64x2_t sum_sq = vdupq_n_f64(0.0);

    size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        float64x2_t vx = vld1q_f64(x + i);
        float64x2_t sq = vmulq_f64(vx, vx);
        sum_sq = vaddq_f64(sum_sq, sq);
    }

    double sum_squares = vget_lane_f64(vadd_f64(vget_low_f64(sum_sq), vget_high_f64(sum_sq)), 0);

    for (; i < count; ++i) {
        sum_squares += x[i] * x[i];
    }

    return std::sqrt(sum_squares);
}

} // namespace kernels
} // namespace mlib
