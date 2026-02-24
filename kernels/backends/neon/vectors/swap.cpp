#include <arm_neon.h>

namespace mlib {
namespace kernels {

template<>
void swap<float>(Vector<float>& x, Vector<float>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector dimensions must match in swap");
    }
    if (x.empty()) {
        return;
    }

    const size_t n = x.size();
    float* __restrict px = x.aligned_data();
    float* __restrict py = y.aligned_data();

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(px + i);
        float32x4_t vy = vld1q_f32(py + i);
        vst1q_f32(px + i, vy);
        vst1q_f32(py + i, vx);
    }

    for (; i < n; ++i) {
        float tmp   = px[i];
        px[i]       = py[i];
        py[i]       = tmp;
    }
}

template<>
void swap<double>(Vector<double>& x, Vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector dimensions must match in swap");
    }
    if (x.empty()) {
        return;
    }

    const size_t n = x.size();
    double* __restrict px = x.aligned_data();
    double* __restrict py = y.aligned_data();

    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t vx = vld1q_f64(px + i);
        float64x2_t vy = vld1q_f64(py + i);
        vst1q_f64(px + i, vy);
        vst1q_f64(py + i, vx);
    }

    for (; i < n; ++i) {
        double tmp  = px[i];
        px[i]       = py[i];
        py[i]       = tmp;
    }
}

} // namespace kernels
} // namespace mlib
