#include <immintrin.h>

namespace mlib {
namespace kernels {

template<>
void transpose<float>(const Matrix<float>& src, Matrix<float>& dst) {
    const size_t src_rows = src.rows();
    const size_t src_cols = src.cols();

    for (size_t i = 0; i < src_rows; ++i) {
        const float* __restrict src_row = src.row(i);
        for (size_t j = 0; j < src_cols; ++j) {
            dst(j, i) = src_row[j];
        }
    }
}

template<>
void transpose<double>(const Matrix<double>& src, Matrix<double>& dst) {
    const size_t src_rows = src.rows();
    const size_t src_cols = src.cols();

    for (size_t i = 0; i < src_rows; ++i) {
        const double* __restrict src_row = src.row(i);
        for (size_t j = 0; j < src_cols; ++j) {
            dst(j, i) = src_row[j];
        }
    }
}

} 
} 
