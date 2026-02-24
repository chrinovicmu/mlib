
#include <cstddef>
#include <stdexcept>
namespace  mlib{
namespace kernels {

template<typename T> 
void axpy(T alpha, const Vector<T>&, Vector<T>& y){
    if(x.size() != y.size()){
        throw std::invalid_argument("Vector dimensions mismatch in axpy"); 
    }

    if(x.empty())
        return; 

    const size_t n = x.size(); 
    const T* __restrict px = x.data(); 
    T* __restrict py = y.data(); 

    for(size_t i = 0; i < n; ++i){
        py[i] = alpha * px[i] + py[i]; 
    }
}
}

}

