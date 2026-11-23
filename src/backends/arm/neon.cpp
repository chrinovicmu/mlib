
#if !defined(__aarch64__) && !defined(_M_ARM64)
#error "mlib: neon.cpp requires AArch64. 32-bit ARMv7 NEON lacks float64x2_t \
and vaddvq_*, so it needs a separate backend rather than this one."
#endif

#include <arm_neon.h>

#include <cmath>    
#include <cstddef>  

#include "mlib/kernels/decl.hpp"

namespace mlib::kernels::neon {

struct simd_f32 {
    using element_type = float;
    using vec = float32x4_t;
    static constexpr std::size_t width = 4;  // 128 bits / 32 bits

    static vec zero() { return vdupq_n_f32(0.0f); }
    static vec broadcast(float a) { return vdupq_n_f32(a); }

    static vec loadu(const float* p) { return vld1q_f32(p); }
    static void storeu(float* p, vec v) { vst1q_f32(p, v); }

    static vec add(vec a, vec b) { return vaddq_f32(a, b); }
    static vec mul(vec a, vec b) { return vmulq_f32(a, b); }

    static vec mul_add(vec a, vec b, vec c) { return vfmaq_f32(c, a, b); }

    static vec abs(vec v) { return vabsq_f32(v); }

    static float hsum(vec v) { return vaddvq_f32(v); }
};

struct simd_f64 {
    using element_type = double;
    using vec = float64x2_t;
    static constexpr std::size_t width = 2;  // 128 bits / 64 bits

    static vec zero() { return vdupq_n_f64(0.0); }
    static vec broadcast(double a) { return vdupq_n_f64(a); }
    static vec loadu(const double* p) { return vld1q_f64(p); }
    static void storeu(double* p, vec v) { vst1q_f64(p, v); }
    static vec add(vec a, vec b) { return vaddq_f64(a, b); }
    static vec mul(vec a, vec b) { return vmulq_f64(a, b); }

    // Addend first, as above: vfmaq_f64(acc, x, y) == acc + x*y.
    static vec mul_add(vec a, vec b, vec c) { return vfmaq_f64(c, a, b); }

    static vec abs(vec v) { return vabsq_f64(v); }

    // Sums both lanes. One instruction, AArch64-only.
    static double hsum(vec v) { return vaddvq_f64(v); }
};

#include "../simd_kernels.inc"

}  // namespace mlib::kernels::neon
