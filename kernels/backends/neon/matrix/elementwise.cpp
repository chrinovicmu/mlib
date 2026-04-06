#include <arm_neon.h>
#include <mlib/matrix.hpp>
#include <cmath>

namespace mlib {
namespace kernels {

template<>
void mat_add<float>(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    const size_t rows = A.rows();
    const size_t cols = A.cols();

    for (size_t i = 0; i < rows; ++i) {
        const float* __restrict a_row = A.row(i);
        const float* __restrict b_row = B.row(i);
        float* __restrict c_row = C.row(i);

        size_t j = 0;
        for (; j + 4 <= cols; j += 4) {
            float32x4_t va = vld1q_f32(a_row + j);
            float32x4_t vb = vld1q_f32(b_row + j);
            float32x4_t r  = vaddq_f32(va, vb);
            vst1q_f32(c_row + j, r);
        }
        for (; j < cols; ++j) {
            c_row[j] = a_row[j] + b_row[j];
        }
    }
}

template<>
void mat_sub<float>(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    const size_t rows = A.rows();
    const size_t cols = A.cols();

    for (size_t i = 0; i < rows; ++i) {
        const float* __restrict a_row = A.row(i);
        const float* __restrict b_row = B.row(i);
        float* __restrict c_row = C.row(i);

        size_t j = 0;
        for (; j + 4 <= cols; j += 4) {
            float32x4_t va = vld1q_f32(a_row + j);
            float32x4_t vb = vld1q_f32(b_row + j);
            float32x4_t r  = vsubq_f32(va, vb);
            vst1q_f32(c_row + j, r);
        }
        for (; j < cols; ++j) {
            c_row[j] = a_row[j] - b_row[j];
        }
    }
}

template<>
void mat_mul<float>(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {  // Hadamard product
    const size_t rows = A.rows();
    const size_t cols = A.cols();

    for (size_t i = 0; i < rows; ++i) {
        const float* __restrict a_row = A.row(i);
        const float* __restrict b_row = B.row(i);
        float* __restrict c_row = C.row(i);

        size_t j = 0;
        for (; j + 4 <= cols; j += 4) {
            float32x4_t va = vld1q_f32(a_row + j);
            float32x4_t vb = vld1q_f32(b_row + j);
            float32x4_t r  = vmulq_f32(va, vb);
            vst1q_f32(c_row + j, r);
        }
        for (; j < cols; ++j) {
            c_row[j] = a_row[j] * b_row[j];
        }
    }
}

template<>
void mat_div<float>(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    const size_t rows = A.rows();
    const size_t cols = A.cols();

    for (size_t i = 0; i < rows; ++i) {
        const float* __restrict a_row = A.row(i);
        const float* __restrict b_row = B.row(i);
        float* __restrict c_row = C.row(i);

        size_t j = 0;
        for (; j + 4 <= cols; j += 4) {
            float32x4_t va = vld1q_f32(a_row + j);
            float32x4_t vb = vld1q_f32(b_row + j);
            float32x4_t r  = vdivq_f32(va, vb);
            vst1q_f32(c_row + j, r);
        }
        for (; j < cols; ++j) {
            c_row[j] = a_row[j] / b_row[j];
        }
    }
}

template<>
void mat_add_scalar<float>(float alpha, const Matrix<float>& A, Matrix<float>& C) {
    const size_t rows = A.rows();
    const size_t cols = A.cols();
    const float32x4_t valpha = vdupq_n_f32(alpha);

    for (size_t i = 0; i < rows; ++i) {
        const float* __restrict a_row = A.row(i);
        float* __restrict c_row = C.row(i);

        size_t j = 0;
        for (; j + 4 <= cols; j += 4) {
            float32x4_t va = vld1q_f32(a_row + j);
            float32x4_t r  = vaddq_f32(va, valpha);
            vst1q_f32(c_row + j, r);
        }
        for (; j < cols; ++j) {
            c_row[j] = a_row[j] + alpha;
        }
    }
}

template<>
void mat_mul_scalar<float>(float alpha, Matrix<float>& A) {   // in-place
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
void mat_neg<float>(const Matrix<float>& A, Matrix<float>& C) {
    const size_t rows = A.rows();
    const size_t cols = A.cols();

    for (size_t i = 0; i < rows; ++i) {
        const float* __restrict a_row = A.row(i);
        float* __restrict c_row = C.row(i);

        size_t j = 0;
        for (; j + 4 <= cols; j += 4) {
            float32x4_t va = vld1q_f32(a_row + j);
            float32x4_t r  = vnegq_f32(va);
            vst1q_f32(c_row + j, r);
        }
        for (; j < cols; ++j) {
            c_row[j] = -a_row[j];
        }
    }
}

template<>
void mat_abs<float>(const Matrix<float>& A, Matrix<float>& C) {
    const size_t rows = A.rows();
    const size_t cols = A.cols();

    for (size_t i = 0; i < rows; ++i) {
        const float* __restrict a_row = A.row(i);
        float* __restrict c_row = C.row(i);

        size_t j = 0;
        for (; j + 4 <= cols; j += 4) {
            float32x4_t va = vld1q_f32(a_row + j);
            float32x4_t r  = vabsq_f32(va);
            vst1q_f32(c_row + j, r);
        }
        for (; j < cols; ++j) {
            c_row[j] = std::abs(a_row[j]);
        }
    }
}

} 
} 
