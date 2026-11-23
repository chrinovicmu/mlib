#if !defined(__riscv) || __riscv_xlen != 64 || !defined(__riscv_vector)
#error "rvv.S requires RV64 with the V extension"
#endif
#if !defined(__riscv_float_abi_double)
#error "rvv.S requires the LP64D ABI"
#endif

    .text

    .p2align 2
    .globl mlib_rvv_width_f32
    .type mlib_rvv_width_f32, @function
mlib_rvv_width_f32:
    .cfi_startproc
    csrr a0, vlenb
    srli a0, a0, 2
    ret
    .cfi_endproc
    .size mlib_rvv_width_f32, .-mlib_rvv_width_f32

    .p2align 2
    .globl mlib_rvv_width_f64
    .type mlib_rvv_width_f64, @function
mlib_rvv_width_f64:
    .cfi_startproc
    csrr a0, vlenb
    srli a0, a0, 3
    ret
    .cfi_endproc
    .size mlib_rvv_width_f64, .-mlib_rvv_width_f64

// simd_f32::zero(): write a vector of zero floats.
    .p2align 2
    .globl mlib_rvv_zero_f32
    .type mlib_rvv_zero_f32, @function
mlib_rvv_zero_f32:
    .cfi_startproc
    vsetvli t0, zero, e32, m1, ta, ma
    vmv.v.i v0, 0
    vse32.v v0, (a0)
    ret
    .cfi_endproc
    .size mlib_rvv_zero_f32, .-mlib_rvv_zero_f32

// simd_f64::zero(): write a vector of zero doubles.
// a0 = result-storage pointer.
    .p2align 2
    .globl mlib_rvv_zero_f64
    .type mlib_rvv_zero_f64, @function
mlib_rvv_zero_f64:
    .cfi_startproc
    vsetvli t0, zero, e64, m1, ta, ma
    vmv.v.i v0, 0
    vse64.v v0, (a0)
    ret
    .cfi_endproc
    .size mlib_rvv_zero_f64, .-mlib_rvv_zero_f64

// simd_f32::broadcast(value): repeat one float in every lane.
    .p2align 2
    .globl mlib_rvv_broadcast_f32
    .type mlib_rvv_broadcast_f32, @function
mlib_rvv_broadcast_f32:
    .cfi_startproc
    vsetvli t0, zero, e32, m1, ta, ma
    vfmv.v.f v0, fa0
    vse32.v v0, (a0)
    ret
    .cfi_endproc
    .size mlib_rvv_broadcast_f32, .-mlib_rvv_broadcast_f32

// simd_f64::broadcast(value): repeat one double in every lane.
    .p2align 2
    .globl mlib_rvv_broadcast_f64
    .type mlib_rvv_broadcast_f64, @function
mlib_rvv_broadcast_f64:
    .cfi_startproc
    vsetvli t0, zero, e64, m1, ta, ma
    vfmv.v.f v0, fa0
    vse64.v v0, (a0)
    ret
    .cfi_endproc
    .size mlib_rvv_broadcast_f64, .-mlib_rvv_broadcast_f64

// simd_f32::loadu(values): read a float vector into temporary storage.
    .p2align 2
    .globl mlib_rvv_loadu_f32
    .type mlib_rvv_loadu_f32, @function
mlib_rvv_loadu_f32:
    .cfi_startproc
    vsetvli t0, zero, e32, m1, ta, ma
    vle32.v v0, (a1)
    vse32.v v0, (a0)
    ret
    .cfi_endproc
    .size mlib_rvv_loadu_f32, .-mlib_rvv_loadu_f32

// simd_f64::loadu(values): read a double vector into temporary storage.
    .p2align 2
    .globl mlib_rvv_loadu_f64
    .type mlib_rvv_loadu_f64, @function
mlib_rvv_loadu_f64:
    .cfi_startproc
    vsetvli t0, zero, e64, m1, ta, ma
    vle64.v v0, (a1)
    vse64.v v0, (a0)
    ret
    .cfi_endproc
    .size mlib_rvv_loadu_f64, .-mlib_rvv_loadu_f64

// simd_f32::storeu(values, value): write a float vector to the user array.
    .p2align 2
    .globl mlib_rvv_storeu_f32
    .type mlib_rvv_storeu_f32, @function
mlib_rvv_storeu_f32:
    .cfi_startproc
    vsetvli t0, zero, e32, m1, ta, ma
    vle32.v v0, (a1)
    vse32.v v0, (a0)
    ret
    .cfi_endproc
    .size mlib_rvv_storeu_f32, .-mlib_rvv_storeu_f32

// simd_f64::storeu(values, value): write a double vector to the user array.
    .p2align 2
    .globl mlib_rvv_storeu_f64
    .type mlib_rvv_storeu_f64, @function
mlib_rvv_storeu_f64:
    .cfi_startproc
    vsetvli t0, zero, e64, m1, ta, ma
    vle64.v v0, (a1)
    vse64.v v0, (a0)
    ret
    .cfi_endproc
    .size mlib_rvv_storeu_f64, .-mlib_rvv_storeu_f64

// simd_f32::add(a, b): add corresponding float lanes.
    .p2align 2
    .globl mlib_rvv_add_f32
    .type mlib_rvv_add_f32, @function
mlib_rvv_add_f32:
    .cfi_startproc
    vsetvli t0, zero, e32, m1, ta, ma
    vle32.v v0, (a1)
    vle32.v v1, (a2)
    vfadd.vv v0, v0, v1
    vse32.v v0, (a0)
    ret
    .cfi_endproc
    .size mlib_rvv_add_f32, .-mlib_rvv_add_f32

// simd_f64::add(a, b): add corresponding double lanes.
    .p2align 2
    .globl mlib_rvv_add_f64
    .type mlib_rvv_add_f64, @function
mlib_rvv_add_f64:
    .cfi_startproc
    vsetvli t0, zero, e64, m1, ta, ma
    vle64.v v0, (a1)
    vle64.v v1, (a2)
    vfadd.vv v0, v0, v1
    vse64.v v0, (a0)
    ret
    .cfi_endproc
    .size mlib_rvv_add_f64, .-mlib_rvv_add_f64

// simd_f32::mul(a, b): multiply corresponding float lanes.
    .p2align 2
    .globl mlib_rvv_mul_f32
    .type mlib_rvv_mul_f32, @function
mlib_rvv_mul_f32:
    .cfi_startproc
    vsetvli t0, zero, e32, m1, ta, ma
    vle32.v v0, (a1)
    vle32.v v1, (a2)
    vfmul.vv v0, v0, v1
    vse32.v v0, (a0)
    ret
    .cfi_endproc
    .size mlib_rvv_mul_f32, .-mlib_rvv_mul_f32

// simd_f64::mul(a, b): multiply corresponding double lanes.
    .p2align 2
    .globl mlib_rvv_mul_f64
    .type mlib_rvv_mul_f64, @function
mlib_rvv_mul_f64:
    .cfi_startproc
    vsetvli t0, zero, e64, m1, ta, ma
    vle64.v v0, (a1)
    vle64.v v1, (a2)
    vfmul.vv v0, v0, v1
    vse64.v v0, (a0)
    ret
    .cfi_endproc
    .size mlib_rvv_mul_f64, .-mlib_rvv_mul_f64

// simd_f32::mul_add(a, b, c): compute a*b + c.
    .p2align 2
    .globl mlib_rvv_mul_add_f32
    .type mlib_rvv_mul_add_f32, @function
mlib_rvv_mul_add_f32:
    .cfi_startproc
    vsetvli t0, zero, e32, m1, ta, ma
    vle32.v v0, (a1)
    vle32.v v1, (a2)
    vle32.v v2, (a3)
    vfmacc.vv v2, v0, v1
    vse32.v v2, (a0)
    ret
    .cfi_endproc
    .size mlib_rvv_mul_add_f32, .-mlib_rvv_mul_add_f32

// simd_f64::mul_add(a, b, c): compute a*b + c.
    .p2align 2
    .globl mlib_rvv_mul_add_f64
    .type mlib_rvv_mul_add_f64, @function
mlib_rvv_mul_add_f64:
    .cfi_startproc
    vsetvli t0, zero, e64, m1, ta, ma
    vle64.v v0, (a1)
    vle64.v v1, (a2)
    vle64.v v2, (a3)
    vfmacc.vv v2, v0, v1
    vse64.v v2, (a0)
    ret
    .cfi_endproc
    .size mlib_rvv_mul_add_f64, .-mlib_rvv_mul_add_f64

// simd_f32::abs(value): clear each float sign bit.
    .p2align 2
    .globl mlib_rvv_abs_f32
    .type mlib_rvv_abs_f32, @function
mlib_rvv_abs_f32:
    .cfi_startproc
    vsetvli t0, zero, e32, m1, ta, ma
    vle32.v v0, (a1)
    vfsgnjx.vv v0, v0, v0
    vse32.v v0, (a0)
    ret
    .cfi_endproc
    .size mlib_rvv_abs_f32, .-mlib_rvv_abs_f32

// simd_f64::abs(value): clear each double sign bit.
    .p2align 2
    .globl mlib_rvv_abs_f64
    .type mlib_rvv_abs_f64, @function
mlib_rvv_abs_f64:
    .cfi_startproc
    vsetvli t0, zero, e64, m1, ta, ma
    vle64.v v0, (a1)
    vfsgnjx.vv v0, v0, v0
    vse64.v v0, (a0)
    ret
    .cfi_endproc
    .size mlib_rvv_abs_f64, .-mlib_rvv_abs_f64

// simd_f32::hsum(value): sum every float lane into one float.
    .p2align 2
    .globl mlib_rvv_hsum_f32
    .type mlib_rvv_hsum_f32, @function
mlib_rvv_hsum_f32:
    .cfi_startproc
    vsetvli t0, zero, e32, m1, ta, ma
    vle32.v v0, (a0)
    fmv.w.x ft0, zero
    vfmv.s.f v1, ft0
    vfredosum.vs v1, v0, v1
    vfmv.f.s fa0, v1
    ret
    .cfi_endproc
    .size mlib_rvv_hsum_f32, .-mlib_rvv_hsum_f32

// simd_f64::hsum(value): sum every double lane into one double.
    .p2align 2
    .globl mlib_rvv_hsum_f64
    .type mlib_rvv_hsum_f64, @function
mlib_rvv_hsum_f64:
    .cfi_startproc
    vsetvli t0, zero, e64, m1, ta, ma
    vle64.v v0, (a0)
    fmv.d.x ft0, zero
    vfmv.s.f v1, ft0
    vfredosum.vs v1, v0, v1
    vfmv.f.s fa0, v1
    ret
    .cfi_endproc
    .size mlib_rvv_hsum_f64, .-mlib_rvv_hsum_f64

    .section .note.GNU-stack,"",@progbits
