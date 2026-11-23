//
// Portable scalar backend. No intrinsics, no ISA flags, buildable on any
// target with a C++17 compiler -- including ones this library has never
// been ported to.
#include "mlib/kernels/decl.hpp"

#include <cmath>    // std::abs, std::sqrt
#include <cstddef>  // std::size_t

namespace mlib::kernels::generic {
namespace {

using std::size_t;


// y := a*x + y   (BLAS AXPY)
template <typename T>
void axpy_impl(size_t n, T a, const T* x, T* y) {
    for (size_t i = 0; i < n; ++i) {
        y[i] += a * x[i];
    }
}

// x := a*x   (BLAS SCAL)
template <typename T>
void scal_impl(size_t n, T a, T* x) {
    for (size_t i = 0; i < n; ++i) {
        x[i] *= a;
    }
}

// y := x   (BLAS COPY)
template <typename T>
void copy_impl(size_t n, const T* x, T* y) {
    for (size_t i = 0; i < n; ++i) {
        y[i] = x[i];
    }
}

// x <-> y   (BLAS SWAP)
template <typename T>
void swap_impl(size_t n, T* x, T* y) {
    for (size_t i = 0; i < n; ++i) {
        const T t = x[i];
        x[i] = y[i];
        y[i] = t;
    }
}

// returns sum(x[i] * y[i])   (BLAS DOT)
template <typename T>
T dot_impl(size_t n, const T* x, const T* y) {
    T acc = T(0);
    for (size_t i = 0; i < n; ++i) {
        acc += x[i] * y[i];
    }
    return acc;
}

// returns sum(|x[i]|)   (BLAS ASUM, L1 norm)
template <typename T>
T asum_impl(size_t n, const T* x) {
    T acc = T(0);
    for (size_t i = 0; i < n; ++i) {
        acc += std::abs(x[i]);
    }
    return acc;
}

// returns sqrt(sum(x[i]^2))   (BLAS NRM2, Euclidean norm)
template <typename T>
T nrm2_impl(size_t n, const T* x) {
    T acc = T(0);
    for (size_t i = 0; i < n; ++i) {
        acc += x[i] * x[i];
    }
    return std::sqrt(acc);
}

template <typename T>
void gemv_impl(size_t m, size_t n, T alpha, const T* a, size_t lda,
               const T* x, T beta, T* y) {
    // One output element per row of A. Each y[i] is a dot product of row i
    // with the whole of x.
    for (size_t i = 0; i < m; ++i) {
        const T* row = a + i * lda;

        T acc = T(0);
        for (size_t j = 0; j < n; ++j) {
            acc += row[j] * x[j];
        }
        y[i] = (beta == T(0)) ? alpha * acc
                              : alpha * acc + beta * y[i];
    }
}

template <typename T>
void gemm_impl(size_t m, size_t n, size_t k, T alpha, const T* a, size_t lda,
               const T* b, size_t ldb, T beta, T* c, size_t ldc) {
    for (size_t i = 0; i < m; ++i) {
        T* crow = c + i * ldc;  // step by ldc, not n -- padding, as above

        if (beta == T(0)) {
            for (size_t j = 0; j < n; ++j)
                crow[j] = T(0);
        } else if (beta != T(1)) {
            for (size_t j = 0; j < n; ++j) 
                crow[j] *= beta;
        }
        // (beta == 1 falls through here having done nothing, intentionally)
    }

    for (size_t i = 0; i < m; ++i) {
        T* crow = c + i * ldc;

        for (size_t p = 0; p < k; ++p) {
            const T aik = alpha * a[i * lda + p];

            if (aik == T(0)) continue;

            const T* brow = b + p * ldb;

            for (size_t j = 0; j < n; ++j) {
                crow[j] += aik * brow[j];
            }
        }
    }
}

}  // anonymous namespace


void axpy_f32(size_t n, float a, const float* x, float* y) { axpy_impl(n, a, x, y); }
void axpy_f64(size_t n, double a, const double* x, double* y) { axpy_impl(n, a, x, y); }

void scal_f32(size_t n, float a, float* x) { scal_impl(n, a, x); }
void scal_f64(size_t n, double a, double* x) { scal_impl(n, a, x); }

void copy_f32(size_t n, const float* x, float* y) { copy_impl(n, x, y); }
void copy_f64(size_t n, const double* x, double* y) { copy_impl(n, x, y); }

void swap_f32(size_t n, float* x, float* y) { swap_impl(n, x, y); }
void swap_f64(size_t n, double* x, double* y) { swap_impl(n, x, y); }

float  dot_f32(size_t n, const float* x, const float* y) { return dot_impl(n, x, y); }
double dot_f64(size_t n, const double* x, const double* y) { return dot_impl(n, x, y); }

float  nrm2_f32(size_t n, const float* x) { return nrm2_impl(n, x); }
double nrm2_f64(size_t n, const double* x) { return nrm2_impl(n, x); }

float  asum_f32(size_t n, const float* x) { return asum_impl(n, x); }
double asum_f64(size_t n, const double* x) { return asum_impl(n, x); }

void gemv_f32(size_t m, size_t n, float alpha, const float* a, size_t lda,
              const float* x, float beta, float* y) {
    gemv_impl(m, n, alpha, a, lda, x, beta, y);
}
void gemv_f64(size_t m, size_t n, double alpha, const double* a, size_t lda,
              const double* x, double beta, double* y) {
    gemv_impl(m, n, alpha, a, lda, x, beta, y);
}

void gemm_f32(size_t m, size_t n, size_t k, float alpha, const float* a, size_t lda,
              const float* b, size_t ldb, float beta, float* c, size_t ldc) {
    gemm_impl(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
void gemm_f64(size_t m, size_t n, size_t k, double alpha, const double* a, size_t lda,
              const double* b, size_t ldb, double beta, double* c, size_t ldc) {
    gemm_impl(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

}  // namespace mlib::kernels::generic
