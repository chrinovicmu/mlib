// kernels/neon/copy.cpp

#include <arm_neon.h>

namespace mlib {
namespace kernels {

template<>
void copy<float>(const Vector<float>& x, Vector<float>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector dimensions must match in copy");
    }
    if (x.empty()) {
        return;
    }

    const size_t n = x.size();
    const float* __restrict px = x.aligned_data();
    float* __restrict py = y.aligned_data();

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(px + i);
        vst1q_f32(py + i, vx);
    }

    for (; i < n; ++i) {
        py[i] = px[i];
    }
}

template<>
void copy<double>(const Vector<double>& x, Vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector dimensions must match in copy");
    }
    if (x.empty()) {
        return;
    }

    const size_t n = x.size();
    const double* __restrict px = x.aligned_data();
    double* __restrict py = y.aligned_data();

    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t vx = vld1q_f64(px + i);
        vst1q_f64(py + i, vx);
    }

    for (; i < n; ++i) {
        py[i] = px[i];
    }
}

} // namespace kernels
} // namespace mlib
