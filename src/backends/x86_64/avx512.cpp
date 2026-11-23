#if !defined(__x86_64__) || defined(_WIN32)
#error "mlib: AVX-512 assembly requires the x86-64 System V ABI"
#endif

#include <cmath>
#include <cstddef>
#include "mlib/kernels/decl.hpp"


namespace mlib::kernels::avx512 {
    
using f32_vec = float __attribute__((vector_size(64)));
using f64_vec = double __attribute__((vector_size(64)));

extern "C" {
f32_vec mlib_avx512_zero_f32() noexcept;
f32_vec mlib_avx512_broadcast_f32(float value) noexcept;
f32_vec mlib_avx512_loadu_f32(const float* values) noexcept;
void mlib_avx512_storeu_f32(float* values, f32_vec value) noexcept;
f32_vec mlib_avx512_add_f32(f32_vec a, f32_vec b) noexcept;
f32_vec mlib_avx512_mul_f32(f32_vec a, f32_vec b) noexcept;
f32_vec mlib_avx512_mul_add_f32(f32_vec a, f32_vec b, f32_vec c) noexcept;
f32_vec mlib_avx512_abs_f32(f32_vec value) noexcept;
float mlib_avx512_hsum_f32(f32_vec value) noexcept;

f64_vec mlib_avx512_zero_f64() noexcept;
f64_vec mlib_avx512_broadcast_f64(double value) noexcept;
f64_vec mlib_avx512_loadu_f64(const double* values) noexcept;
void mlib_avx512_storeu_f64(double* values, f64_vec value) noexcept;
f64_vec mlib_avx512_add_f64(f64_vec a, f64_vec b) noexcept;
f64_vec mlib_avx512_mul_f64(f64_vec a, f64_vec b) noexcept;
f64_vec mlib_avx512_mul_add_f64(f64_vec a, f64_vec b, f64_vec c) noexcept;
f64_vec mlib_avx512_abs_f64(f64_vec value) noexcept;
double mlib_avx512_hsum_f64(f64_vec value) noexcept;
}

struct simd_f32 {
    using element_type = float;
    using vec = f32_vec;
    static constexpr std::size_t width = 16;

    static vec zero() { return mlib_avx512_zero_f32(); }
    static vec broadcast(float value) { return mlib_avx512_broadcast_f32(value); }
    static vec loadu(const float* values) { return mlib_avx512_loadu_f32(values); }
    static void storeu(float* values, vec value) { mlib_avx512_storeu_f32(values, value); }
    static vec add(vec a, vec b) { return mlib_avx512_add_f32(a, b); }
    static vec mul(vec a, vec b) { return mlib_avx512_mul_f32(a, b); }
    static vec mul_add(vec a, vec b, vec c) { return mlib_avx512_mul_add_f32(a, b, c); }
    static vec abs(vec value) { return mlib_avx512_abs_f32(value); }
    static float hsum(vec value) { return mlib_avx512_hsum_f32(value); }
};

struct simd_f64 {
    using element_type = double;
    using vec = f64_vec;
    static constexpr std::size_t width = 8;

    static vec zero() { return mlib_avx512_zero_f64(); }
    static vec broadcast(double value) { return mlib_avx512_broadcast_f64(value); }
    static vec loadu(const double* values) { return mlib_avx512_loadu_f64(values); }
    static void storeu(double* values, vec value) { mlib_avx512_storeu_f64(values, value); }
    static vec add(vec a, vec b) { return mlib_avx512_add_f64(a, b); }
    static vec mul(vec a, vec b) { return mlib_avx512_mul_f64(a, b); }
    static vec mul_add(vec a, vec b, vec c) { return mlib_avx512_mul_add_f64(a, b, c); }
    static vec abs(vec value) { return mlib_avx512_abs_f64(value); }
    static double hsum(vec value) { return mlib_avx512_hsum_f64(value); }
};


#include "../simd_kernels.inc"

} 
