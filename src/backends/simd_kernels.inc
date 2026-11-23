namespace {

// axpy: y := a*x + y
template <class S>
void axpy_impl(std::size_t n, typename S::element_type a,
               const typename S::element_type* x, typename S::element_type* y) {
    constexpr std::size_t W = S::width;

    const typename S::vec av = S::broadcast(a);

    std::size_t i = 0;
    for (; i + W <= n; i += W) {
        auto xv = S::loadu(x + i);            // Load a vector from x
        auto yv = S::loadu(y + i);            // Load a vector from y
        auto result = S::mul_add(av, xv, yv); // Calculate av * xv + yv
        S::storeu(y + i, result);             // Write the result back into y
    }
    for (; i < n; ++i) {
        y[i] += a * x[i];
    }
}

// scal: x := a*x, in place.
template <class S>
void scal_impl(std::size_t n, typename S::element_type a,
               typename S::element_type* x) {
    constexpr std::size_t W = S::width;
    const typename S::vec av = S::broadcast(a);

    std::size_t i = 0;
    for (; i + W <= n; i += W) {
        auto xv = S::loadu(x + i);      // Load a vector from x
        auto result = S::mul(av, xv);   // Scale the loaded values
        S::storeu(x + i, result);       // Write the result back into x
    }
    for (; i < n; ++i) {
        x[i] *= a;
    }
}

// copy: y := x
template <class S>
void copy_impl(std::size_t n, const typename S::element_type* x,
               typename S::element_type* y) {
    constexpr std::size_t W = S::width;

    std::size_t i = 0;
    for (; i + W <= n; i += W) {
        auto xv = S::loadu(x + i); // Load a vector from x
        S::storeu(y + i, xv);      // Copy the loaded values into y
    }
    for (; i < n; ++i) {
        y[i] = x[i];
    }
}

// swap: x <-> y
//
template <class S>
void swap_impl(std::size_t n, typename S::element_type* x,
               typename S::element_type* y) {
    constexpr std::size_t W = S::width;

    std::size_t i = 0;
    for (; i + W <= n; i += W) {
        const typename S::vec xv = S::loadu(x + i);
        const typename S::vec yv = S::loadu(y + i);
        S::storeu(x + i, yv);
        S::storeu(y + i, xv);
    }
    for (; i < n; ++i) {
        const typename S::element_type t = x[i];
        x[i] = y[i];
        y[i] = t;
    }
}

// A vectorised sum(x[i]*y[i]) over a raw range.
template <class S>
typename S::element_type dot_range(std::size_t n,
                                   const typename S::element_type* x,
                                   const typename S::element_type* y) {
    constexpr std::size_t W = S::width;

    typename S::vec a0 = S::zero();
    typename S::vec a1 = S::zero();
    typename S::vec a2 = S::zero();
    typename S::vec a3 = S::zero();

    std::size_t i = 0;

    // Main loop: four independent FMA chains, 4*W elements per iteration.
    for (; i + 4 * W <= n; i += 4 * W) {
        auto x0 = S::loadu(x + i + 0 * W);
        auto y0 = S::loadu(y + i + 0 * W);
        a0 = S::mul_add(x0, y0, a0);

        auto x1 = S::loadu(x + i + 1 * W);
        auto y1 = S::loadu(y + i + 1 * W);
        a1 = S::mul_add(x1, y1, a1);

        auto x2 = S::loadu(x + i + 2 * W);
        auto y2 = S::loadu(y + i + 2 * W);
        a2 = S::mul_add(x2, y2, a2);

        auto x3 = S::loadu(x + i + 3 * W);
        auto y3 = S::loadu(y + i + 3 * W);
        a3 = S::mul_add(x3, y3, a3);
    }

    // Drain loop: whole vectors that did not fill a group of four. Folded
    // into a0 only -- with at most three iterations left there is nothing
    // to gain from spreading them.
    for (; i + W <= n; i += W) {
        auto xv = S::loadu(x + i);
        auto yv = S::loadu(y + i);
        a0 = S::mul_add(xv, yv, a0);
    }

    // Pairwise tree, not a0+a1+a2+a3 left to right. Same instruction
    // count, shorter dependency chain, and marginally better rounding
    // because the partial magnitudes stay closer together.
    auto sum01 = S::add(a0, a1);
    auto sum23 = S::add(a2, a3);
    auto sumv = S::add(sum01, sum23);
    typename S::element_type s = S::hsum(sumv);

    // Scalar tail, accumulated into the already-reduced sum.
    for (; i < n; ++i) {
        s += x[i] * y[i];
    }
    return s;
}

// dot: the public reduction. Thin wrapper so that dot and gemv provably
// share one implementation.
template <class S>
typename S::element_type dot_impl(std::size_t n,
                                  const typename S::element_type* x,
                                  const typename S::element_type* y) {
    return dot_range<S>(n, x, y);
}

// nrm2: sqrt(sum(x[i]^2))
template <class S>
typename S::element_type nrm2_impl(std::size_t n, const typename S::element_type* x) {
    constexpr std::size_t W = S::width;

    typename S::vec a0 = S::zero();
    typename S::vec a1 = S::zero();
    typename S::vec a2 = S::zero();
    typename S::vec a3 = S::zero();

    std::size_t i = 0;
    for (; i + 4 * W <= n; i += 4 * W) {
        // mul_add(v, v, acc) squares and accumulates in whatever form
        // the target provides.
        const typename S::vec v0 = S::loadu(x + i + 0 * W);
        const typename S::vec v1 = S::loadu(x + i + 1 * W);
        const typename S::vec v2 = S::loadu(x + i + 2 * W);
        const typename S::vec v3 = S::loadu(x + i + 3 * W);
        a0 = S::mul_add(v0, v0, a0);
        a1 = S::mul_add(v1, v1, a1);
        a2 = S::mul_add(v2, v2, a2);
        a3 = S::mul_add(v3, v3, a3);
    }
    for (; i + W <= n; i += W) {
        const typename S::vec v = S::loadu(x + i);
        a0 = S::mul_add(v, v, a0);
    }

    auto sum01 = S::add(a0, a1);
    auto sum23 = S::add(a2, a3);
    auto sumv = S::add(sum01, sum23);
    typename S::element_type s = S::hsum(sumv);
    for (; i < n; ++i) {
        s += x[i] * x[i];
    }

    // std::sqrt, not the rsqrt approximation instructions. vrsqrtps is
    // ~12-bit accurate and would need Newton refinement to be usable;
    // sqrt here is correctly rounded and this is one instruction outside
    // the loop, so it is not on any hot path.
    return std::sqrt(s);
}

// asum: sum(|x[i]|)
// Absolute value is done by masking off the sign bit (see each ISA's
template <class S>
typename S::element_type asum_impl(std::size_t n, const typename S::element_type* x) {
    constexpr std::size_t W = S::width;

    typename S::vec a0 = S::zero();
    typename S::vec a1 = S::zero();
    typename S::vec a2 = S::zero();
    typename S::vec a3 = S::zero();

    std::size_t i = 0;
    for (; i + 4 * W <= n; i += 4 * W) {
        auto v0 = S::loadu(x + i + 0 * W);
        auto abs0 = S::abs(v0);
        a0 = S::add(a0, abs0);

        auto v1 = S::loadu(x + i + 1 * W);
        auto abs1 = S::abs(v1);
        a1 = S::add(a1, abs1);

        auto v2 = S::loadu(x + i + 2 * W);
        auto abs2 = S::abs(v2);
        a2 = S::add(a2, abs2);

        auto v3 = S::loadu(x + i + 3 * W);
        auto abs3 = S::abs(v3);
        a3 = S::add(a3, abs3);
    }
    for (; i + W <= n; i += W) {
        auto xv = S::loadu(x + i);
        auto absv = S::abs(xv);
        a0 = S::add(a0, absv);
    }

    auto sum01 = S::add(a0, a1);
    auto sum23 = S::add(a2, a3);
    auto sumv = S::add(sum01, sum23);
    typename S::element_type s = S::hsum(sumv);
    for (; i < n; ++i) {
        s += std::abs(x[i]);
    }
    return s;
}

// gemv: y := alpha*A*x + beta*y,  A is m x n row-major with leading
// dimension lda.
// A is row-major, so row i is CONTIGUOUS and a matrix-vector product is
// exactly m dot products. Vectorising along the row is therefore the
// natural direction; the alternative (vectorising down a column) would
// stride by lda on every load.
template <class S>
void gemv_impl(std::size_t m, std::size_t n, typename S::element_type alpha,
               const typename S::element_type* a, std::size_t lda,
               const typename S::element_type* x, typename S::element_type beta,
               typename S::element_type* y) {
    for (std::size_t i = 0; i < m; ++i) {
        const typename S::element_type* row = a + i * lda;

        const typename S::element_type acc = dot_range<S>(n, row, x);

        y[i] = (beta == typename S::element_type(0))
                   ? alpha * acc
                   : alpha * acc + beta * y[i];
    }
}

// gemm: C := alpha*A*B + beta*C
//
// A is m x k (lda), B is k x n (ldb), C is m x n (ldc), all row-major.
template <class S>
void gemm_impl(std::size_t m, std::size_t n, std::size_t k,
               typename S::element_type alpha,
               const typename S::element_type* a, std::size_t lda,
               const typename S::element_type* b, std::size_t ldb,
               typename S::element_type beta, typename S::element_type* c,
               std::size_t ldc) {
    using T = typename S::element_type;
    constexpr std::size_t W = S::width;

    for (std::size_t i = 0; i < m; ++i) {
        T* crow = c + i * ldc;

        if (beta == T(0)) {
            // Overwrite with zeros. Same reasoning as gemv above: 0*NaN is
            // NaN, so multiplying is not equivalent to clearing.
            const typename S::vec z = S::zero();
            std::size_t j = 0;
            for (; j + W <= n; j += W) {
                S::storeu(crow + j, z);
            }
            for (; j < n; ++j) crow[j] = T(0);

        } else if (beta != T(1)) {
            const typename S::vec bv = S::broadcast(beta);
            std::size_t j = 0;
            for (; j + W <= n; j += W) {
                auto cv = S::loadu(crow + j);
                auto result = S::mul(bv, cv);
                S::storeu(crow + j, result);
            }
            for (; j < n; ++j) crow[j] *= beta;
        }
    }

    // accumulate alpha*A*B into C 
    for (std::size_t i = 0; i < m; ++i) {
        T* crow = c + i * ldc;

        for (std::size_t p = 0; p < k; ++p) {
            const T aik = alpha * a[i * lda + p];

            if (aik == T(0)) continue;

            const typename S::vec av = S::broadcast(aik);
            const T* brow = b + p * ldb;

            std::size_t j = 0;
            for (; j + W <= n; j += W) {
                auto bv = S::loadu(brow + j);
                auto cv = S::loadu(crow + j);
                auto result = S::mul_add(av, bv, cv);
                S::storeu(crow + j, result);
            }
            for (; j < n; ++j) {
                crow[j] += aik * brow[j];
            }
        }
    }
}

}  // anonymous namespace


void axpy_f32(std::size_t n, float a, const float* x, float* y) {
    axpy_impl<simd_f32>(n, a, x, y);
}
void axpy_f64(std::size_t n, double a, const double* x, double* y) {
    axpy_impl<simd_f64>(n, a, x, y);
}

void scal_f32(std::size_t n, float a, float* x) { scal_impl<simd_f32>(n, a, x); }
void scal_f64(std::size_t n, double a, double* x) { scal_impl<simd_f64>(n, a, x); }

void copy_f32(std::size_t n, const float* x, float* y) {
    copy_impl<simd_f32>(n, x, y);
}
void copy_f64(std::size_t n, const double* x, double* y) {
    copy_impl<simd_f64>(n, x, y);
}

void swap_f32(std::size_t n, float* x, float* y) { swap_impl<simd_f32>(n, x, y); }
void swap_f64(std::size_t n, double* x, double* y) { swap_impl<simd_f64>(n, x, y); }

float dot_f32(std::size_t n, const float* x, const float* y) {
    return dot_impl<simd_f32>(n, x, y);
}
double dot_f64(std::size_t n, const double* x, const double* y) {
    return dot_impl<simd_f64>(n, x, y);
}

float nrm2_f32(std::size_t n, const float* x) { return nrm2_impl<simd_f32>(n, x); }
double nrm2_f64(std::size_t n, const double* x) { return nrm2_impl<simd_f64>(n, x); }

float asum_f32(std::size_t n, const float* x) { return asum_impl<simd_f32>(n, x); }
double asum_f64(std::size_t n, const double* x) { return asum_impl<simd_f64>(n, x); }

void gemv_f32(std::size_t m, std::size_t n, float alpha, const float* a,
              std::size_t lda, const float* x, float beta, float* y) {
    gemv_impl<simd_f32>(m, n, alpha, a, lda, x, beta, y);
}
void gemv_f64(std::size_t m, std::size_t n, double alpha, const double* a,
              std::size_t lda, const double* x, double beta, double* y) {
    gemv_impl<simd_f64>(m, n, alpha, a, lda, x, beta, y);
}

void gemm_f32(std::size_t m, std::size_t n, std::size_t k, float alpha,
              const float* a, std::size_t lda, const float* b, std::size_t ldb,
              float beta, float* c, std::size_t ldc) {
    gemm_impl<simd_f32>(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
void gemm_f64(std::size_t m, std::size_t n, std::size_t k, double alpha,
              const double* a, std::size_t lda, const double* b, std::size_t ldb,
              double beta, double* c, std::size_t ldc) {
    gemm_impl<simd_f64>(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
