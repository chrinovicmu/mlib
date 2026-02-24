#include <cstddef>
#include <immintrin.h>

namespace mlib{
namespace kernels{

template<> 
void axpy(float alpha, const Vector<float>& x, Vector<float>& y){

    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector dimensions mismatch in axpy");
    }
    if (x.empty()) 
        return;

    const size_t n = x.size();
    const float* __restrict px = x.aligned_data();
    float* __restrict py = y.aligned_data();

    __m256 valpha = _mm256_set1_ps(alpha); 
    size_t i = 0; 
    for(;i + 8 <= n; i += 8){
        __m256 vx = _mm256_load_ps(px + i); 
        __m256 vy = _mm256_load_ps(py + i); 
        __m256 res = _mm256_fmadd_ps(vx, valpha, vy);
        _mm256_store_ps(py + i, res); 
    }

    for(;i < n; ++i){
        py[i] = alpha * px[i] + py[i]; 
    }
}

template<>

void axpy<double>(double alpha, const Vector<double>& x, Vector<double>& y) {
    
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector dimensions mismatch in axpy");
    }
    if (x.empty()) return;

    const size_t n = x.size();
    const double* __restrict px = x.aligned_data();
    double* __restrict py = y.aligned_data();

    __m256d valpha = _mm256_set1_pd(alpha);

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vx = _mm256_load_pd(px + i);
        __m256d vy = _mm256_load_pd(py + i);
        __m256d result = _mm256_fmadd_pd(vx, valpha, vy);
        _mm256_store_pd(py + i, result);
    }

    for (; i < n; ++i) {
        py[i] = alpha * px[i] + py[i];
    }
}

}

}
