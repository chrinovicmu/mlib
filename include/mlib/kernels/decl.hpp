
#pragma once

#include <cstddef>  // std::size_t in every prototype below

#define MLIB_KERNEL_PROTOS(T, SUF)                                          \
    void axpy_##SUF(std::size_t n, T a, const T* x, T* y);                    \
    void scal_##SUF(std::size_t n, T a, T* x);                                \
    void copy_##SUF(std::size_t n, const T* x, T* y);                         \
    void swap_##SUF(std::size_t n, T* x, T* y);                               \
    T dot_##SUF(std::size_t n, const T* x, const T* y);                       \
    T nrm2_##SUF(std::size_t n, const T* x);                                  \
    T asum_##SUF(std::size_t n, const T* x);                                  \
    void gemv_##SUF(std::size_t m, std::size_t n, T alpha,                    \
                    const T* a, std::size_t lda,                              \
                    const T* x, T beta, T* y);                                \
    void gemm_##SUF(std::size_t m, std::size_t n, std::size_t k, T alpha,     \
                    const T* a, std::size_t lda,                              \
                    const T* b, std::size_t ldb,                              \
                    T beta, T* c, std::size_t ldc)

#define MLIB_DECLARE_ISA(NS)           \
    namespace NS {                       \
    MLIB_KERNEL_PROTOS(float, f32);    \
    MLIB_KERNEL_PROTOS(double, f64);   \
    }

namespace mlib::kernels {

MLIB_DECLARE_ISA(generic)  // portable scalar; always compiled and linked
MLIB_DECLARE_ISA(avx)      // x86-64, 128/256-bit, no FMA
MLIB_DECLARE_ISA(avx2)     // x86-64, 256-bit + FMA
MLIB_DECLARE_ISA(avx512)   // x86-64, 512-bit + opmask tails
MLIB_DECLARE_ISA(neon)     // AArch64, 128-bit, architecturally mandatory
MLIB_DECLARE_ISA(sve)      // AArch64, scalable vectors, optional
MLIB_DECLARE_ISA(sme)      // AArch64, streaming matrix extension, optional
MLIB_DECLARE_ISA(rvv)      // RISC-V V extension, scalable, optional

}  // namespace mlib::kernels

#undef MLIB_DECLARE_ISA

