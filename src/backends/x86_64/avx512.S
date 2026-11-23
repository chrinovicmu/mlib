#if !defined(__x86_64__) || defined(_WIN32)
#error "avx512.S requires the x86-64 System V ABI"
#endif

    .intel_syntax noprefix
    .text

// simd_f32::zero(): return sixteen zero floats.
    .p2align 4
    .globl mlib_avx512_zero_f32
mlib_avx512_zero_f32:
    .cfi_startproc
    vxorps zmm0, zmm0, zmm0           // XOR a register with itself: every bit becomes zero.
    ret                              
    .cfi_endproc
.Lend_zero_f32:

// simd_f64::zero(): return eight zero doubles.
    .p2align 4
    .globl mlib_avx512_zero_f64
mlib_avx512_zero_f64:
    .cfi_startproc
    vxorpd zmm0, zmm0, zmm0           // All eight double lanes become +0.0.
    ret                             
    .cfi_endproc
.Lend_zero_f64:

// simd_f32::broadcast(value): repeat one float sixteen times.
    .p2align 4
    .globl mlib_avx512_broadcast_f32
mlib_avx512_broadcast_f32:
    .cfi_startproc
    vbroadcastss zmm0, xmm0           // Copy the low float into every lane of zmm0.
    ret                              
    .cfi_endproc
.Lend_broadcast_f32:

// simd_f64::broadcast(value): repeat one double eight times.
    .p2align 4
    .globl mlib_avx512_broadcast_f64
mlib_avx512_broadcast_f64:
    .cfi_startproc
    vbroadcastsd zmm0, xmm0           // Copy the low double into every lane of zmm0.
    ret                              
    .cfi_endproc
.Lend_broadcast_f64:

// simd_f32::loadu(values): load sixteen floats from memory.
    .p2align 4
    .globl mlib_avx512_loadu_f32
mlib_avx512_loadu_f32:
    .cfi_startproc
    vmovups zmm0, [rdi]               // Read 64 bytes at rdi into zmm0; unaligned is allowed.
    ret                              
    .cfi_endproc
.Lend_loadu_f32:

// simd_f64::loadu(values): load eight doubles from memory.
    .p2align 4
    .globl mlib_avx512_loadu_f64
mlib_avx512_loadu_f64:
    .cfi_startproc
    vmovupd zmm0, [rdi]               // Read eight consecutive doubles into zmm0.
    ret                              
    .cfi_endproc
.Lend_loadu_f64:

// simd_f32::storeu(values, value): store sixteen floats.
    .p2align 4
    .globl mlib_avx512_storeu_f32
mlib_avx512_storeu_f32:
    .cfi_startproc
    vmovups [rdi], zmm0               // Write the 64 bytes in zmm0 to memory at rdi.
    ret                             
    .cfi_endproc
.Lend_storeu_f32:

// simd_f64::storeu(values, value): store eight doubles.
    .p2align 4
    .globl mlib_avx512_storeu_f64
mlib_avx512_storeu_f64:
    .cfi_startproc
    vmovupd [rdi], zmm0               // Write eight doubles to memory at rdi.
    ret                              
    .cfi_endproc
.Lend_storeu_f64:

// simd_f32::add(a, b): add corresponding float lanes.
    .p2align 4
    .globl mlib_avx512_add_f32
mlib_avx512_add_f32:
    .cfi_startproc
    vaddps zmm0, zmm0, zmm1           // For each lane i: result[i] = a[i] + b[i].
    ret                              
    .cfi_endproc
.Lend_add_f32:

// simd_f64::add(a, b): add corresponding double lanes.
    .p2align 4
    .globl mlib_avx512_add_f64
mlib_avx512_add_f64:
    .cfi_startproc
    vaddpd zmm0, zmm0, zmm1           // Eight double additions, one per lane.
    ret                              
    .cfi_endproc
.Lend_add_f64:

// simd_f32::mul(a, b): multiply corresponding float lanes.
    .p2align 4
    .globl mlib_avx512_mul_f32
mlib_avx512_mul_f32:
    .cfi_startproc
    vmulps zmm0, zmm0, zmm1           // For each lane i: result[i] = a[i] * b[i].
    ret                              
    .cfi_endproc
.Lend_mul_f32:

// simd_f64::mul(a, b): multiply corresponding double lanes.
    .p2align 4
    .globl mlib_avx512_mul_f64
mlib_avx512_mul_f64:
    .cfi_startproc
    vmulpd zmm0, zmm0, zmm1           // Eight double multiplications, one per lane.
    ret                              
    .cfi_endproc
.Lend_mul_f64:

// simd_f32::mul_add(a, b, c): compute a*b + c.
    .p2align 4
    .globl mlib_avx512_mul_add_f32
mlib_avx512_mul_add_f32:
    .cfi_startproc
    vfmadd213ps zmm0, zmm1, zmm2      // Sixteen fused a[i]*b[i] + c[i] results; one rounding each.
    ret                              
    .cfi_endproc
.Lend_mul_add_f32:

// simd_f64::mul_add(a, b, c): compute a*b + c.
    .p2align 4
    .globl mlib_avx512_mul_add_f64
mlib_avx512_mul_add_f64:
    .cfi_startproc
    vfmadd213pd zmm0, zmm1, zmm2      // Eight fused a[i]*b[i] + c[i] results.
    ret                              
    .cfi_endproc
.Lend_mul_add_f64:

// simd_f32::abs(value): clear the sign bit in each float.
    .p2align 4
    .globl mlib_avx512_abs_f32
mlib_avx512_abs_f32:
    .cfi_startproc
    vpternlogd zmm1, zmm1, zmm1, 255  // Truth table 0xff: every output bit is 1.
    vpsrld zmm1, zmm1, 1              // Shift each 32-bit lane right: mask = 0x7fffffff.
    vandps zmm0, zmm0, zmm1           // Keep magnitude bits and clear every sign bit.
    ret                              
    .cfi_endproc
.Lend_abs_f32:

// simd_f64::abs(value): clear the sign bit in each double.
// Input/result: zmm0. Scratch register: zmm1.
    .p2align 4
    .globl mlib_avx512_abs_f64
mlib_avx512_abs_f64:
    .cfi_startproc
    vpternlogd zmm1, zmm1, zmm1, 255  // Fill the entire mask register with 1 bits.
    vpsrlq zmm1, zmm1, 1              // Shift each 64-bit lane: mask = 0x7fffffffffffffff.
    vandpd zmm0, zmm0, zmm1           // Clear the sign bit of each double.
    ret                              
    .cfi_endproc
.Lend_abs_f64:

// simd_f32::hsum(value): add all sixteen float lanes.
    .p2align 4
    .globl mlib_avx512_hsum_f32
mlib_avx512_hsum_f32:
    .cfi_startproc
    vextractf64x4 ymm1, zmm0, 1       // Copy the upper 256 bits (8 floats) into ymm1; no conversion.
    vaddps ymm0, ymm0, ymm1           // Add upper 8 floats to lower 8: 16 lanes -> 8 sums.
    vextractf128 xmm1, ymm0, 1        // Copy the upper 128 bits (4 of those sums) into xmm1.
    vaddps xmm0, xmm0, xmm1           // Add upper 4 to lower 4: 8 sums -> 4 sums.
    vmovhlps xmm1, xmm0, xmm0         // Move lanes 2 and 3 into the low two lanes of xmm1.
    vaddps xmm0, xmm0, xmm1           // Low lanes now hold s0+s2 and s1+s3.
    vshufps xmm1, xmm0, xmm0, 0x55    // Copy lane 1 into every lane of xmm1.
    vaddss xmm0, xmm0, xmm1           // Add the two remaining sums in lane 0; return one float.
    ret                              
    .cfi_endproc
.Lend_hsum_f32:

// simd_f64::hsum(value): add all eight double lanes.
// Input: zmm0. Result: lowest double in xmm0. Scratch: zmm1/ymm1/xmm1.
    .p2align 4
    .globl mlib_avx512_hsum_f64
mlib_avx512_hsum_f64:
    .cfi_startproc
    vextractf64x4 ymm1, zmm0, 1       // Copy the upper 4 doubles into ymm1.
    vaddpd ymm0, ymm0, ymm1           // Add upper 4 to lower 4: 8 lanes -> 4 sums.
    vextractf128 xmm1, ymm0, 1        // Copy the upper 2 sums into xmm1.
    vaddpd xmm0, xmm0, xmm1           // Add upper 2 to lower 2: 4 sums -> 2 sums.
    vunpckhpd xmm1, xmm0, xmm0        // Copy the high double into both lanes of xmm1.
    vaddsd xmm0, xmm0, xmm1           // Add the last two sums in lane 0; return one double.
    ret                              
    .cfi_endproc
.Lend_hsum_f64:

#ifdef __APPLE__
    .globl _mlib_avx512_zero_f32
    .set _mlib_avx512_zero_f32, mlib_avx512_zero_f32
    .globl _mlib_avx512_zero_f64
    .set _mlib_avx512_zero_f64, mlib_avx512_zero_f64
    .globl _mlib_avx512_broadcast_f32
    .set _mlib_avx512_broadcast_f32, mlib_avx512_broadcast_f32
    .globl _mlib_avx512_broadcast_f64
    .set _mlib_avx512_broadcast_f64, mlib_avx512_broadcast_f64
    .globl _mlib_avx512_loadu_f32
    .set _mlib_avx512_loadu_f32, mlib_avx512_loadu_f32
    .globl _mlib_avx512_loadu_f64
    .set _mlib_avx512_loadu_f64, mlib_avx512_loadu_f64
    .globl _mlib_avx512_storeu_f32
    .set _mlib_avx512_storeu_f32, mlib_avx512_storeu_f32
    .globl _mlib_avx512_storeu_f64
    .set _mlib_avx512_storeu_f64, mlib_avx512_storeu_f64
    .globl _mlib_avx512_add_f32
    .set _mlib_avx512_add_f32, mlib_avx512_add_f32
    .globl _mlib_avx512_add_f64
    .set _mlib_avx512_add_f64, mlib_avx512_add_f64
    .globl _mlib_avx512_mul_f32
    .set _mlib_avx512_mul_f32, mlib_avx512_mul_f32
    .globl _mlib_avx512_mul_f64
    .set _mlib_avx512_mul_f64, mlib_avx512_mul_f64
    .globl _mlib_avx512_mul_add_f32
    .set _mlib_avx512_mul_add_f32, mlib_avx512_mul_add_f32
    .globl _mlib_avx512_mul_add_f64
    .set _mlib_avx512_mul_add_f64, mlib_avx512_mul_add_f64
    .globl _mlib_avx512_abs_f32
    .set _mlib_avx512_abs_f32, mlib_avx512_abs_f32
    .globl _mlib_avx512_abs_f64
    .set _mlib_avx512_abs_f64, mlib_avx512_abs_f64
    .globl _mlib_avx512_hsum_f32
    .set _mlib_avx512_hsum_f32, mlib_avx512_hsum_f32
    .globl _mlib_avx512_hsum_f64
    .set _mlib_avx512_hsum_f64, mlib_avx512_hsum_f64
#else
// ELF records each symbol as a function and records its byte length.
    .type mlib_avx512_zero_f32, @function
    .size mlib_avx512_zero_f32, .Lend_zero_f32-mlib_avx512_zero_f32
    .type mlib_avx512_zero_f64, @function
    .size mlib_avx512_zero_f64, .Lend_zero_f64-mlib_avx512_zero_f64
    .type mlib_avx512_broadcast_f32, @function
    .size mlib_avx512_broadcast_f32, .Lend_broadcast_f32-mlib_avx512_broadcast_f32
    .type mlib_avx512_broadcast_f64, @function
    .size mlib_avx512_broadcast_f64, .Lend_broadcast_f64-mlib_avx512_broadcast_f64
    .type mlib_avx512_loadu_f32, @function
    .size mlib_avx512_loadu_f32, .Lend_loadu_f32-mlib_avx512_loadu_f32
    .type mlib_avx512_loadu_f64, @function
    .size mlib_avx512_loadu_f64, .Lend_loadu_f64-mlib_avx512_loadu_f64
    .type mlib_avx512_storeu_f32, @function
    .size mlib_avx512_storeu_f32, .Lend_storeu_f32-mlib_avx512_storeu_f32
    .type mlib_avx512_storeu_f64, @function
    .size mlib_avx512_storeu_f64, .Lend_storeu_f64-mlib_avx512_storeu_f64
    .type mlib_avx512_add_f32, @function
    .size mlib_avx512_add_f32, .Lend_add_f32-mlib_avx512_add_f32
    .type mlib_avx512_add_f64, @function
    .size mlib_avx512_add_f64, .Lend_add_f64-mlib_avx512_add_f64
    .type mlib_avx512_mul_f32, @function
    .size mlib_avx512_mul_f32, .Lend_mul_f32-mlib_avx512_mul_f32
    .type mlib_avx512_mul_f64, @function
    .size mlib_avx512_mul_f64, .Lend_mul_f64-mlib_avx512_mul_f64
    .type mlib_avx512_mul_add_f32, @function
    .size mlib_avx512_mul_add_f32, .Lend_mul_add_f32-mlib_avx512_mul_add_f32
    .type mlib_avx512_mul_add_f64, @function
    .size mlib_avx512_mul_add_f64, .Lend_mul_add_f64-mlib_avx512_mul_add_f64
    .type mlib_avx512_abs_f32, @function
    .size mlib_avx512_abs_f32, .Lend_abs_f32-mlib_avx512_abs_f32
    .type mlib_avx512_abs_f64, @function
    .size mlib_avx512_abs_f64, .Lend_abs_f64-mlib_avx512_abs_f64
    .type mlib_avx512_hsum_f32, @function
    .size mlib_avx512_hsum_f32, .Lend_hsum_f32-mlib_avx512_hsum_f32
    .type mlib_avx512_hsum_f64, @function
    .size mlib_avx512_hsum_f64, .Lend_hsum_f64-mlib_avx512_hsum_f64
// Tell ELF linkers that this object does not need an executable stack.
    .section .note.GNU-stack,"",@progbits
#endif
     
