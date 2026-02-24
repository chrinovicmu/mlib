
#include <arm_neon.h>
#include <cmath>

namespace mlib {
namespace kernels {

// ─── float versions ───────────────────────────────────────────────

template<>
void add<float>(const float* __restrict x, const float* __restrict y, float* __restrict z, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vy = vld1q_f32(y + i);
        float32x4_t r  = vaddq_f32(vx, vy);
        vst1q_f32(z + i, r);
    }
    for (; i < count; ++i) {
        z[i] = x[i] + y[i];
    }
}

template<>
void sub<float>(const float* __restrict x, const float* __restrict y, float* __restrict z, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vy = vld1q_f32(y + i);
        float32x4_t r  = vsubq_f32(vx, vy);
        vst1q_f32(z + i, r);
    }
    for (; i < count; ++i) {
        z[i] = x[i] - y[i];
    }
}

template<>
void mul<float>(const float* __restrict x, const float* __restrict y, float* __restrict z, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vy = vld1q_f32(y + i);
        float32x4_t r  = vmulq_f32(vx, vy);
        vst1q_f32(z + i, r);
    }
    for (; i < count; ++i) {
        z[i] = x[i] * y[i];
    }
}

template<>
void div<float>(const float* __restrict x, const float* __restrict y, float* __restrict z, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vy = vld1q_f32(y + i);
        float32x4_t r  = vdivq_f32(vx, vy);
        vst1q_f32(z + i, r);
    }
    for (; i < count; ++i) {
        z[i] = x[i] / y[i];
    }
}

template<>
void abs<float>(const float* __restrict x, float* __restrict y, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t r  = vabsq_f32(vx);
        vst1q_f32(y + i, r);
    }
    for (; i < count; ++i) {
        y[i] = std::fabs(x[i]);
    }
}

template<>
void neg<float>(const float* __restrict x, float* __restrict y, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t r  = vnegq_f32(vx);
        vst1q_f32(y + i, r);
    }
    for (; i < count; ++i) {
        y[i] = -x[i];
    }
}

// ─── double versions ───────────────────────────────────────────────

template<>
void add<double>(const double* __restrict x, const double* __restrict y, double* __restrict z, size_t count) {
    size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        float64x2_t vx = vld1q_f64(x + i);
        float64x2_t vy = vld1q_f64(y + i);
        float64x2_t r  = vaddq_f64(vx, vy);
        vst1q_f64(z + i, r);
    }
    for (; i < count; ++i) {
        z[i] = x[i] + y[i];
    }
}

template<>
void sub<double>(const double* __restrict x, const double* __restrict y, double* __restrict z, size_t count) {
    size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        float64x2_t vx = vld1q_f64(x + i);
        float64x2_t vy = vld1q_f64(y + i);
        float64x2_t r  = vsubq_f64(vx, vy);
        vst1q_f64(z + i, r);
    }
    for (; i < count; ++i) {
        z[i] = x[i] - y[i];
    }
}

template<>
void mul<double>(const double* __restrict x, const double* __restrict y, double* __restrict z, size_t count) {
    size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        float64x2_t vx = vld1q_f64(x + i);
        float64x2_t vy = vld1q_f64(y + i);
        float64x2_t r  = vmulq_f64(vx, vy);
        vst1q_f64(z + i, r);
    }
    for (; i < count; ++i) {
        z[i] = x[i] * y[i];
    }
}

template<>
void div<double>(const double* __restrict x, const double* __restrict y, double* __restrict z, size_t count) {
    size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        float64x2_t vx = vld1q_f64(x + i);
        float64x2_t vy = vld1q_f64(y + i);
        float64x2_t r  = vdivq_f64(vx, vy);
        vst1q_f64(z + i, r);
    }
    for (; i < count; ++i) {
        z[i] = x[i] / y[i];
    }
}

template<>
void abs<double>(const double* __restrict x, double* __restrict y, size_t count) {
    size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        float64x2_t vx = vld1q_f64(x + i);
        float64x2_t r  = vabsq_f64(vx);
        vst1q_f64(y + i, r);
    }
    for (; i < count; ++i) {
        y[i] = std::fabs(x[i]);
    }
}

template<>
void neg<double>(const double* __restrict x, double* __restrict y, size_t count) {
    size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        float64x2_t vx = vld1q_f64(x + i);
        float64x2_t r  = vnegq_f64(vx);
        vst1q_f64(y + i, r);
    }
    for (; i < count; ++i) {
        y[i] = -x[i];
    }
}

} // namespace kernels
} // namespace mlib
