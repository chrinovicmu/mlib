#pragma once

#include "mlib/config.hpp"
#include "mlib/kernels/decl.hpp"
#include "mlib/tensor.hpp"

#include <stdexcept>  // std::invalid_argument, thrown by detail::require

namespace mlib::kernels {

#if defined(MLIB_TARGET_ISA_AVX512)
namespace active = avx512;
#elif defined(MLIB_TARGET_ISA_AVX2)
namespace active = avx2;
#elif defined(MLIB_TARGET_ISA_AVX)
namespace active = avx;
#elif defined(MLIB_TARGET_ISA_NEON)
namespace active = neon;
#elif defined(MLIB_TARGET_ISA_SVE)
namespace active = sve;
#elif defined(MLIB_TARGET_ISA_SME)
namespace active = sme;
#elif defined(MLIB_TARGET_ISA_RVV)
namespace active = rvv;
#elif defined(MLIB_TARGET_ISA_GENERIC)
namespace active = generic;
#else
#error "mlib: no MLIB_TARGET_ISA_* macro defined. \
Is the generated config.hpp on the include path?"
#endif

}  

namespace mlib {
    

namespace detail {

inline void require(bool cond, const char* what) {
    if (!cond) 
        throw std::invalid_argument(what);
}
    
}

inline void axpy(float a, const Vector<float>& x, Vector<float>& y) {
    detail::require(x.size() == y.size(), "mlib::axpy: size mismatch");
    if (x.empty() || a == 0.0f) 
        return;
    kernels::active::axpy_f32(x.size(), a, x.data(), y.data());
}
inline void axpy(double a, const Vector<double>& x, Vector<double>& y) {
    detail::require(x.size() == y.size(), "mlib::axpy: size mismatch");
    if (x.empty() || a == 0.0) return;
    kernels::active::axpy_f64(x.size(), a, x.data(), y.data());
}

inline void scal(float a, Vector<float>& x) {
    if (x.empty()) return;
    kernels::active::scal_f32(x.size(), a, x.data());
}
inline void scal(double a, Vector<double>& x) {
    if (x.empty()) return;
    kernels::active::scal_f64(x.size(), a, x.data());
}

inline void copy(const Vector<float>& x, Vector<float>& y) {
    detail::require(x.size() == y.size(), "mlib::copy: size mismatch");
    if (x.empty()) return;
    kernels::active::copy_f32(x.size(), x.data(), y.data());
}
inline void copy(const Vector<double>& x, Vector<double>& y) {
    detail::require(x.size() == y.size(), "mlib::copy: size mismatch");
    if (x.empty()) return;
    kernels::active::copy_f64(x.size(), x.data(), y.data());
}

inline void swap(Vector<float>& x, Vector<float>& y) {
    detail::require(x.size() == y.size(), "mlib::swap: size mismatch");
    if (x.empty()) return;
    kernels::active::swap_f32(x.size(), x.data(), y.data());
}
inline void swap(Vector<double>& x, Vector<double>& y) {
    detail::require(x.size() == y.size(), "mlib::swap: size mismatch");
    if (x.empty()) return;
    kernels::active::swap_f64(x.size(), x.data(), y.data());
}

// sum(x[i] * y[i])
inline float dot(const Vector<float>& x, const Vector<float>& y) {
    detail::require(x.size() == y.size(), "mlib::dot: size mismatch");
    if (x.empty()) 
        return 0.0f;
    return kernels::active::dot_f32(x.size(), x.data(), y.data());
}
inline double dot(const Vector<double>& x, const Vector<double>& y) {
    detail::require(x.size() == y.size(), "mlib::dot: size mismatch");
    
    if (x.empty()) 
        return 0.0;
    return kernels::active::dot_f64(x.size(), x.data(), y.data());
}

inline float nrm2(const Vector<float>& x) {
    if (x.empty()) 
        return 0.0f;
    return kernels::active::nrm2_f32(x.size(), x.data());
}
inline double nrm2(const Vector<double>& x) {
    if (x.empty()) 
        return 0.0;
    return kernels::active::nrm2_f64(x.size(), x.data());
}

// sum(|x[i]|) -- L1 norm.
inline float asum(const Vector<float>& x) {
    if (x.empty()) 
        return 0.0f;
    return kernels::active::asum_f32(x.size(), x.data());
}
inline double asum(const Vector<double>& x) {
    if (x.empty()) return 0.0;
    return kernels::active::asum_f64(x.size(), x.data());
}

// Level 2 -- matrix/vector

inline void gemv(float alpha, const Matrix<float>& A, const Vector<float>& x,
                 float beta, Vector<float>& y) {
    detail::require(A.cols() == x.size(), "mlib::gemv: A.cols != x.size");
    detail::require(A.rows() == y.size(), "mlib::gemv: A.rows != y.size");
    
    if (A.empty()) 
        return;
    kernels::active::gemv_f32(A.rows(), A.cols(), alpha, A.data(), A.stride(),
                              x.data(), beta, y.data());
}
inline void gemv(double alpha, const Matrix<double>& A, const Vector<double>& x,
                 double beta, Vector<double>& y) {
    detail::require(A.cols() == x.size(), "mlib::gemv: A.cols != x.size");
    detail::require(A.rows() == y.size(), "mlib::gemv: A.rows != y.size");
    
    if (A.empty()) 
        return;
    kernels::active::gemv_f64(A.rows(), A.cols(), alpha, A.data(), A.stride(),
                              x.data(), beta, y.data());
}

// Level 3 -- matrix/matrix

inline void gemm(float alpha, const Matrix<float>& A, const Matrix<float>& B,
                 float beta, Matrix<float>& C) {
    detail::require(A.cols() == B.rows(), "mlib::gemm: A.cols != B.rows");
    detail::require(A.rows() == C.rows(), "mlib::gemm: A.rows != C.rows");
    detail::require(B.cols() == C.cols(), "mlib::gemm: B.cols != C.cols");
    
    if (C.empty()) 
        return;
    kernels::active::gemm_f32(A.rows(), B.cols(), A.cols(), alpha,
                              A.data(), A.stride(), B.data(), B.stride(),
                              beta, C.data(), C.stride());
}
inline void gemm(double alpha, const Matrix<double>& A, const Matrix<double>& B,
                 double beta, Matrix<double>& C) {
    detail::require(A.cols() == B.rows(), "mlib::gemm: A.cols != B.rows");
    detail::require(A.rows() == C.rows(), "mlib::gemm: A.rows != C.rows");
    detail::require(B.cols() == C.cols(), "mlib::gemm: B.cols != C.cols");
    if (C.empty()) return;
    kernels::active::gemm_f64(A.rows(), B.cols(), A.cols(), alpha,
                              A.data(), A.stride(), B.data(), B.stride(),
                              beta, C.data(), C.stride());
}

}  // namespace mlib
    
}
