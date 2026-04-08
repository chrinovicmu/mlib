#include <arm_neon.h>

namespace mlib {
namespace kernels {

template<>
void mscal<float>(float alpha, Matrix<float>& A) {
    const size_t rows = A.rows();
    const size_t cols = A.cols();
    const float32x4_t valpha = vdupq_n_f32(alpha);

    for (size_t i = 0; i < rows; ++i) {
        float* __restrict a_row = A.row(i);

        size_t j = 0;
        for (; j + 4 <= cols; j += 4) {
            float32x4_t va = vld1q_f32(a_row + j);
            float32x4_t r  = vmulq_f32(va, valpha);
            vst1q_f32(a_row + j, r);
        }
        for (; j < cols; ++j) {
            a_row[j] *= alpha;
        }
    }
}

template<>
void mscal<double>(double alpha, Matrix<double>& A) {
    const size_t rows = A.rows();
    const size_t cols = A.cols();
    const float64x2_t valpha = vdupq_n_f64(alpha);

    for (size_t i = 0; i < rows; ++i) {
        double* __restrict a_row = A.row(i);

        size_t j = 0;
        for (; j + 2 <= cols; j += 2) {
            float64x2_t va = vld1q_f64(a_row + j);
            float64x2_t r  = vmulq_f64(va, valpha);
            vst1q_f64(a_row + j, r);
        }
        for (; j < cols; ++j) {
            a_row[j] *= alpha;
        }
    }
}

} // namespace kernels
} // namespace mlib
