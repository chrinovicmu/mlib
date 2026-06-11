#include <mlib/matrix.hpp>
#include <mlib/vector.hpp>

namespace mlib{
namespace kernels{
    
template<typename T> 
void gemv(T alpha, const Matrix<T>& A, const Vector<T>& x, T beta , Vector<T>& y)
{
    const size_t m = A.rows(); 
    const size_t n = A.cols(); 

    for (size_t i = 0; i < m; ++i){
        y[i] *= beta; 
    }

    for(size_t i = 0; i < m; ++i){
        const T* __restrict a_row = A.row(i); 
        T sum = T(0); 
        for(size_t j -0; j < n; ++j){
            sum += a_row[j] * x[j]; 
        }
        y[i] += alpha * sum; 
    }
}

} 
}
