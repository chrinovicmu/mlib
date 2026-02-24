// backends/neon/vectors/dot.cpp

#include <arm_neon.h>

namespace mlib {
namespace kernels {

template<>
float dot<float>(const float* __restrict x, const float* __restrict y, size_t count) {
    float32x4_t sum = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vy = vld1q_f32(y + i);
        sum = vfmaq_f32(sum, vx, vy);
    }

    float32x2_t pair = vadd_f32(vget_low_f32(sum), vget_high_f32(sum)); 
    float32x2_t total = vpadd_f32(pair, pair); 
    float result = vget_lane_f32(total, 0);

    for(;i < count; ++i){
        result += x[i] * y[i]; 
    }

    return result; 

}

template<>
double dot<double>(const double* __restrict x, const double* __restrict y, size_t count) {
    float64x2_t sum = vdupq_n_f64(0.0);

    size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        float64x2_t vx = vld1q_f64(x + i);
        float64x2_t vy = vld1q_f64(y + i);
        sum = vfmaq_f64(sum ,vx, vy); 
    }

    double result = vget_lane_f64(vadd_f64(vget_low_f64(sum), vget_high_f64(sum)), 0);

    for (; i < count; ++i) {
        result += x[i] * y[i];
    }

    return result;
}

} 
} 
