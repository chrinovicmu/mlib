// kernels/backends/neon/matrix/gemv.cpp

#include <arm_neon.h>

namespace mlib {
namespace kernels {

template<>
void gemv<float>(float alpha, const Matrix<float>& A, const Vector<float>& x,
                 float beta, Vector<float>& y) {
    const size_t m = A.rows();
    const size_t n = A.cols();

    const float32x4_t alpha_v = vdupq_n_f32(alpha);
    const float32x4_t beta_v  = vdupq_n_f32(beta);

    for (size_t i = 0; i < m; ++i) {
        const float* __restrict a_row = A.row(i);
        float sum = 0.0f;

        size_t j = 0;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (; j + 4 <= n; j += 4) {
            float32x4_t va = vld1q_f32(a_row + j);
            float32x4_t vx = vld1q_f32(x.data() + j);
            acc = vfmaq_f32(acc, va, vx);
        }

        // Horizontal sum
        float32x2_t pair = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        sum = vget_lane_f32(vpadd_f32(pair, pair), 0);

        for (; j < n; ++j) {
            sum += a_row[j] * x[j];
        }

        y[i] = beta * y[i] + alpha * sum;
    }
}

template<>
void gemv<double>(double alpha, const Matrix<double>& A, const Vector<double>& x,
                  double beta, Vector<double>& y) {
    const size_t m = A.rows();
    const size_t n = A.cols();

    const float64x2_t alpha_v = vdupq_n_f64(alpha);
    const float64x2_t beta_v  = vdupq_n_f64(beta);

    for (size_t i = 0; i < m; ++i) {
        const double* __restrict a_row = A.row(i);
        double sum = 0.0;

        size_t j = 0;
        float64x2_t acc = vdupq_n_f64(0.0);
        for (; j + 2 <= n; j += 2) {
            float64x2_t va = vld1q_f64(a_row + j);
            float64x2_t vx = vld1q_f64(x.data() + j);
            acc = vfmaq_f64(acc, va, vx);
        }
        sum = vget_lane_f64(vadd_f64(vget_low_f64(acc), vget_high_f64(acc)), 0);

        for (; j < n; ++j) {
            sum += a_row[j] * x[j];
        }

        y[i] = beta * y[i] + alpha * sum;
    }
}

} 
} 
