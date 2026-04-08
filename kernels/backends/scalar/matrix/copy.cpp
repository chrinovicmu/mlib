// kernels/backends/scalar/matrix/copy.cpp

#include <mlib/matrix.hpp>

namespace mlib {
namespace kernels {

template<typename T>
void copy(const Matrix<T>& src, Matrix<T>& dst) {
    const size_t rows = src.rows();
    const size_t cols = src.cols();

    for (size_t i = 0; i < rows; ++i) {
        const T* __restrict src_row = src.row(i);
        T* __restrict dst_row = dst.row(i);

        for (size_t j = 0; j < cols; ++j) {
            dst_row[j] = src_row[j];
        }
    }
}

} // namespace kernels
} // namespace mlib
