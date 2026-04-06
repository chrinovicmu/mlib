#include <immintrin.h>
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
        for (; j + 8 <= cols; j += 8) {
            __m256 va = _mm256_load_ps(a_row + j);
            __m256 vb = _mm256_load_ps(b_row + j);
            __m256 r  = _mm256_add_ps(va, vb);
            _mm256_store_ps(c_row + j, r);
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
        for (; j + 8 <= cols; j += 8) {
            __m256 va = _mm256_load_ps(a_row + j);
            __m256 vb = _mm256_load_ps(b_row + j);
            __m256 r  = _mm256_sub_ps(va, vb);
            _mm256_store_ps(c_row + j, r);
        }
        for (; j < cols; ++j) {
            c_row[j] = a_row[j] - b_row[j];
        }
    }
}

template<>
void mat_mul<float>(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {  // Hadamard
    const size_t rows = A.rows();
    const size_t cols = A.cols();

    for (size_t i = 0; i < rows; ++i) {
        const float* __restrict a_row = A.row(i);
        const float* __restrict b_row = B.row(i);
        float* __restrict c_row = C.row(i);

        size_t j = 0;
        for (; j + 8 <= cols; j += 8) {
            __m256 va = _mm256_load_ps(a_row + j);
            __m256 vb = _mm256_load_ps(b_row + j);
            __m256 r  = _mm256_mul_ps(va, vb);
            _mm256_store_ps(c_row + j, r);
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
        for (; j + 8 <= cols; j += 8) {
            __m256 va = _mm256_load_ps(a_row + j);
            __m256 vb = _mm256_load_ps(b_row + j);
            __m256 r  = _mm256_div_ps(va, vb);
            _mm256_store_ps(c_row + j, r);
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
    const __m256 valpha = _mm256_set1_ps(alpha);

    for (size_t i = 0; i < rows; ++i) {
        const float* __restrict a_row = A.row(i);
        float* __restrict c_row = C.row(i);

        size_t j = 0;
        for (; j + 8 <= cols; j += 8) {
            __m256 va = _mm256_load_ps(a_row + j);
            __m256 r  = _mm256_add_ps(va, valpha);
            _mm256_store_ps(c_row + j, r);
        }
        for (; j < cols; ++j) {
            c_row[j] = a_row[j] + alpha;
        }
    }
}

template<>
void mat_mul_scalar<float>(float alpha, Matrix<float>& A) {
    const size_t rows = A.rows();
    const size_t cols = A.cols();
    const __m256 valpha = _mm256_set1_ps(alpha);

    for (size_t i = 0; i < rows; ++i) {
        float* __restrict a_row = A.row(i);

        size_t j = 0;
        for (; j + 8 <= cols; j += 8) {
            __m256 va = _mm256_load_ps(a_row + j);
            __m256 r  = _mm256_mul_ps(va, valpha);
            _mm256_store_ps(a_row + j, r);
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
    const __m256 sign_flip = _mm256_set1_ps(-0.0f);

    for (size_t i = 0; i < rows; ++i) {
        const float* __restrict a_row = A.row(i);
        float* __restrict c_row = C.row(i);

        size_t j = 0;
        for (; j + 8 <= cols; j += 8) {
            __m256 va = _mm256_load_ps(a_row + j);
            __m256 r  = _mm256_xor_ps(va, sign_flip);
            _mm256_store_ps(c_row + j, r);
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
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    for (size_t i = 0; i < rows; ++i) {
        const float* __restrict a_row = A.row(i);
        float* __restrict c_row = C.row(i);

        size_t j = 0;
        for (; j + 8 <= cols; j += 8) {
            __m256 va = _mm256_load_ps(a_row + j);
            __m256 r  = _mm256_andnot_ps(sign_mask, va);
            _mm256_store_ps(c_row + j, r);
        }
        for (; j < cols; ++j) {
            c_row[j] = std::abs(a_row[j]);
        }
    }
}
}
}
