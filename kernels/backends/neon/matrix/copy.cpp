#include <arm_neon.h>

namespace mlib {
namespace kernels {

template<>
void copy<float>(const Matrix<float>& src, Matrix<float>& dst) {
    const size_t rows = src.rows();
    const size_t cols = src.cols();

    for (size_t i = 0; i < rows; ++i) {
        const float* __restrict src_row = src.row(i);
        float* __restrict dst_row = dst.row(i);

        size_t j = 0;
        for (; j + 4 <= cols; j += 4) {
            float32x4_t v = vld1q_f32(src_row + j);
            vst1q_f32(dst_row + j, v);
        }
        for (; j < cols; ++j) {
            dst_row[j] = src_row[j];
        }
    }
}

template<>
void copy<double>(const Matrix<double>& src, Matrix<double>& dst) {
    const size_t rows = src.rows();
    const size_t cols = src.cols();

    for (size_t i = 0; i < rows; ++i) {
        const double* __restrict src_row = src.row(i);
        double* __restrict dst_row = dst.row(i);

        size_t j = 0;
        for (; j + 2 <= cols; j += 2) {
            float64x2_t v = vld1q_f64(src_row + j);
            vst1q_f64(dst_row + j, v);
        }
        for (; j < cols; ++j) {
            dst_row[j] = src_row[j];
        }
    }
}

} 
} 
