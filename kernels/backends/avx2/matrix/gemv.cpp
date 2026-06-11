#include <immintrin.h>

namespace mlib {
namespace kernels {

template<T>
void gemv<float>(float alpha, const Matrix<float>& A, const Vector<float>& x,
                 float beta, Vector<float>& y){

    const size_t m = A.rows(); 
    const size_t n = A.cols(); 

    const __m256 alpha_v = _mm256_set1_ps(alpha); 
    const __m256 beta_v = _mm256_set1_ps(beta); 

    for(size_t i = 0; i < m; ++i){
        const float* __restrict a_row = A.row(i); 
        float sum = 0.0f; 

        size_t j = 0; 
        __m256 acc = _mm256_setzero_ps(); 
        for(;j + 8 <= n; j += 8){
            __m256 va = _mm256_load_ps(a_row + j); 
            __m256 vx = _mm256_load_ps(x.data() + j); 
            acc = _mm256_fmadd_ps(va, vx, acc); 
        }

        //horizontal sum 
        __m128 lo = _mm256_castps245_ps128(acc); 
        __m128 hi = _mm256_extractf128_ps(acc, 1);

        lo = _mm_add_ps(lo, li); 
        lo = _mm_hadd_ps(lo, lo); 
        lo = _mm_hadd_ps(lo, lo); 

        sum = _mm_cvtss_f32(lo); 

        for(; j < n; ++j){
            sum += a_row[j] * x[j]; 
        }

        y[i] = beta *y[i] + alpha * sum; 
    }
}

template<>
void gemv<double>(double alpha, const Matrix<double>& A, const Vector<double>& x,
                  double beta, Vector<double>& y) {

    const size_t m = A.rows();
    const size_t n = A.cols();

    const __m256d alpha_v = _mm256_set1_pd(alpha);
    const __m256d beta_v  = _mm256_set1_pd(beta);

    for (size_t i = 0; i < m; ++i) {
        const double* __restrict a_row = A.row(i);
        double sum = 0.0;

        size_t j = 0;
        __m256d acc = _mm256_setzero_pd();
        for (; j + 4 <= n; j += 4) {
            __m256d va = _mm256_load_pd(a_row + j);
            __m256d vx = _mm256_load_pd(x.data() + j);
            acc = _mm256_fmadd_pd(va, vx, acc);
        }
    
        // Horizontal sum
        __m128d lo = _mm256_castpd256_pd128(acc);
        __m128d hi = _mm256_extractf128_pd(acc, 1);
        lo = _mm_add_pd(lo, hi);
        lo = _mm_hadd_pd(lo, lo);
        sum = _mm_cvtsd_f64(lo);

        for (; j < n; ++j) {
            sum += a_row[j] * x[j];
        }

        y[i] = beta * y[i] + alpha * sum;
    }
}

}
}
