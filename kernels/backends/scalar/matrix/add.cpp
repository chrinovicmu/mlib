#include <cstddef>
#include <mlib/matrix.hpp> 

namespace mlib{
namespace kernels{


void add(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>&C)
{
    const size_t rows = A.rows(); 
    const size_t cols = A.cols(); 

    for(size_t i = 0; i < rows; ++i){
        const T* __restict a_row = A.row(i); 
        const T* __restict b_row = B.row(i);
        T* __restict c_row = C.row(i); 

        for(size_t j = 0; j < cols; ++j){
            c_rows[j] = a_row[j] + b_roq[j]; 
        }
    }
}
}

}
