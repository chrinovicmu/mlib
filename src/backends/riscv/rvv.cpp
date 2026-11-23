#if !defined(__riscv) || __riscv_xlen != 64 || !defined(__riscv_vector)
#error "mlib: RVV requires RV64 with the V extension (-march=rv64gcv)"
#endif
#if !defined(__riscv_float_abi_double)
#error "mlib: RVV assembly requires an LP64D toolchain (-mabi=lp64d)"
#endif

#include <riscv_vector.h> 
#include <cmath>
#include <cstddef>
#include "mlib/kernels/decl.hpp"

namespace mlib::kernels::rvv {

extern "C" {
std::size_t mlib_rvv_width_f32() noexcept;
void mlib_rvv_zero_f32(vfloat32m1_t* out) noexcept;
void mlib_rvv_broadcast_f32(vfloat32m1_t* out, float value) noexcept;
void mlib_rvv_loadu_f32(vfloat32m1_t* out, const float* values) noexcept;
void mlib_rvv_storeu_f32(float* values, const vfloat32m1_t* value) noexcept;
void mlib_rvv_add_f32(vfloat32m1_t* out, const vfloat32m1_t* a, const vfloat32m1_t* b) noexcept;
void mlib_rvv_mul_f32(vfloat32m1_t* out, const vfloat32m1_t* a, const vfloat32m1_t* b) noexcept;
void mlib_rvv_mul_add_f32(vfloat32m1_t* out, const vfloat32m1_t* a,
                        const vfloat32m1_t* b, const vfloat32m1_t* c) noexcept;
void mlib_rvv_abs_f32(vfloat32m1_t* out, const vfloat32m1_t* value) noexcept;
float mlib_rvv_hsum_f32(const vfloat32m1_t* value) noexcept;

std::size_t mlib_rvv_width_f64() noexcept;
void mlib_rvv_zero_f64(vfloat64m1_t* out) noexcept;
void mlib_rvv_broadcast_f64(vfloat64m1_t* out, double value) noexcept;
void mlib_rvv_loadu_f64(vfloat64m1_t* out, const double* values) noexcept;
void mlib_rvv_storeu_f64(double* values, const vfloat64m1_t* value) noexcept;
void mlib_rvv_add_f64(vfloat64m1_t* out, const vfloat64m1_t* a, const vfloat64m1_t* b) noexcept;
void mlib_rvv_mul_f64(vfloat64m1_t* out, const vfloat64m1_t* a, const vfloat64m1_t* b) noexcept;
void mlib_rvv_mul_add_f64(vfloat64m1_t* out, const vfloat64m1_t* a,
                        const vfloat64m1_t* b, const vfloat64m1_t* c) noexcept;
void mlib_rvv_abs_f64(vfloat64m1_t* out, const vfloat64m1_t* value) noexcept;
double mlib_rvv_hsum_f64(const vfloat64m1_t* value) noexcept;
}

struct simd_f32 {
    using element_type = float;
    using vec = vfloat32m1_t;
    static std::size_t width() { return mlib_rvv_width_f32(); }

    static vec zero() {
        vec result;
        mlib_rvv_zero_f32(&result);
        return result;
    }
    static vec broadcast(float value) {
        vec result;
        mlib_rvv_broadcast_f32(&result, value);
        return result;
    }
    static vec loadu(const float* values) {
        vec result;
        mlib_rvv_loadu_f32(&result, values);
        return result;
    }
    static void storeu(float* values, vec value) {
        mlib_rvv_storeu_f32(values, &value);
    }
    static vec add(vec a, vec b) {
        vec result;
        mlib_rvv_add_f32(&result, &a, &b);
        return result;
    }
    static vec mul(vec a, vec b) {
        vec result;
        mlib_rvv_mul_f32(&result, &a, &b);
        return result;
    }
    static vec mul_add(vec a, vec b, vec c) {
        vec result;
        mlib_rvv_mul_add_f32(&result, &a, &b, &c);
        return result;
    }
    static vec abs(vec value) {
        vec result;
        mlib_rvv_abs_f32(&result, &value);
        return result;
    }
    static float hsum(vec value) { return mlib_rvv_hsum_f32(&value); }
};

struct simd_f64 {
    using element_type = double;
    using vec = vfloat64m1_t;
    static std::size_t width() { return mlib_rvv_width_f64(); }

    static vec zero() {
        vec result;
        mlib_rvv_zero_f64(&result);
        return result;
    }
    static vec broadcast(double value) {
        vec result;
        mlib_rvv_broadcast_f64(&result, value);
        return result;
    }
    static vec loadu(const double* values) {
        vec result;
        mlib_rvv_loadu_f64(&result, values);
        return result;
    }
    static void storeu(double* values, vec value) {
        mlib_rvv_storeu_f64(values, &value);
    }
    static vec add(vec a, vec b) {
        vec result;
        mlib_rvv_add_f64(&result, &a, &b);
        return result;
    }
    static vec mul(vec a, vec b) {
        vec result;
        mlib_rvv_mul_f64(&result, &a, &b);
        return result;
    }
    static vec mul_add(vec a, vec b, vec c) {
        vec result;
        mlib_rvv_mul_add_f64(&result, &a, &b, &c);
        return result;
    }
    static vec abs(vec value) {
        vec result;
        mlib_rvv_abs_f64(&result, &value);
        return result;
    }
    static double hsum(vec value) { return mlib_rvv_hsum_f64(&value); }
};

#include "../simd_kernels.inc"

} // namespace mlib::kernels::rvv
