// backends/neon/vectors/copy.cpp

#include <arm_neon.h>

namespace mlib {
namespace kernels {

template<>
void copy<float>(const float* __restrict src, float* __restrict dst, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        vst1q_f32(dst + i, v);
    }

    for (; i < count; ++i) {
        dst[i] = src[i];
    }
}

template<>
void copy<double>(const double* __restrict src, double* __restrict dst, size_t count) {
    size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        float64x2_t v = vld1q_f64(src + i);
        vst1q_f64(dst + i, v);
    }

    for (; i < count; ++i) {
        dst[i] = src[i];
    }
}

} // namespace kernels
} // namespace mlib
