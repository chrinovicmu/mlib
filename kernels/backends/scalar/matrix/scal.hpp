// kernels/backends/scalar/matrix/scal.cpp

#include <mlib/matrix.hpp>

namespace mlib {
namespace kernels {

template<typename T>
void mscal(T alpha, Matrix<T>& A) {
    const size_t rows = A.rows();
    const size_t cols = A.cols();

    for (size_t i = 0; i < rows; ++i) {
        T* __restrict a_row = A.row(i);
        for (size_t j = 0; j < cols; ++j) {
            a_row[j] *= alpha;
        }
    }
}

} // namespace kernels
} // namespace mlib
