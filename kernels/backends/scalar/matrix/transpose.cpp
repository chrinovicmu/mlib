// kernels/backends/scalar/matrix/transpose.cpp

#include <mlib/matrix.hpp>

namespace mlib {
namespace kernels {

template<typename T>
void transpose(const Matrix<T>& src, Matrix<T>& dst) {
    const size_t src_rows = src.rows();
    const size_t src_cols = src.cols();

    for (size_t i = 0; i < src_rows; ++i) {
        const T* __restrict src_row = src.row(i);
        for (size_t j = 0; j < src_cols; ++j) {
            dst(j, i) = src_row[j];   
        }
    }
}

} 
} 
