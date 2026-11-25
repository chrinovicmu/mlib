// tests/test_level23.cpp
//
// Level-2 (gemv) and Level-3 (gemm) correctness.
//

#include "harness.hpp"
#include <mlib/mlib.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using namespace mlib;

namespace {

template <typename T>
void fill_pseudo_m(Matrix<T>& m, std::uint32_t seed) {
    std::uint32_t s = seed;
    for (std::size_t i = 0; i < m.rows(); ++i)
        for (std::size_t j = 0; j < m.cols(); ++j) {
            s = s * 1664525u + 1013904223u;
            m(i, j) = static_cast<T>(static_cast<double>(s >> 8) / 8388608.0 - 1.0);
        }
}

template <typename T>
void fill_pseudo_v(Vector<T>& v, std::uint32_t seed) {
    std::uint32_t s = seed;
    for (std::size_t i = 0; i < v.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        v[i] = static_cast<T>(static_cast<double>(s >> 8) / 8388608.0 - 1.0);
    }
}

template <typename T>
double tol_for(std::size_t k) {
    const double eps = static_cast<double>(std::numeric_limits<T>::epsilon());
    return eps * static_cast<double>(k + 8) * 8.0;
}

struct Shape2 { std::size_t m, n; };
struct Shape3 { std::size_t m, n, k; };

const std::vector<Shape2> kGemvShapes = {
    {1, 1}, {1, 17}, {17, 1}, {4, 4}, {8, 16}, {16, 8},
    {17, 33}, {33, 17}, {64, 64}, {65, 63}, {7, 129}};

const std::vector<Shape3> kGemmShapes = {
    {1, 1, 1},   {2, 3, 4},    {4, 4, 4},    {8, 8, 8},
    {16, 16, 16}, {17, 5, 9},  {5, 17, 9},   {9, 5, 17},
    {33, 33, 33}, {64, 64, 64}, {65, 33, 17}, {1, 64, 64},
    {64, 1, 64}, {64, 64, 1}};

}  // namespace

// gemv -

template <typename T>
static void check_gemv(T alpha, T beta) {
    for (const Shape2 s : kGemvShapes) {
        Matrix<T> A(s.m, s.n);
        Vector<T> x(s.n), y(s.m);
        fill_pseudo_m(A, 101);
        fill_pseudo_v(x, 202);
        fill_pseudo_v(y, 303);
        Vector<T> y0 = y.clone();

        gemv(alpha, A, x, beta, y);

        for (std::size_t i = 0; i < s.m; ++i) {
            long double acc = 0.0L;
            for (std::size_t j = 0; j < s.n; ++j)
                acc += static_cast<long double>(A(i, j)) *
                       static_cast<long double>(x[j]);
            const long double want =
                (beta == T(0))
                    ? static_cast<long double>(alpha) * acc
                    : static_cast<long double>(alpha) * acc +
                          static_cast<long double>(beta) *
                              static_cast<long double>(y0[i]);
            TH_CHECK_CLOSE(static_cast<double>(y[i]),
                           static_cast<double>(want), tol_for<T>(s.n));
        }
    }
}

TH_TEST(gemv_f32_alpha1_beta0)  { check_gemv<float>(1.0f, 0.0f); }
TH_TEST(gemv_f32_alpha_beta)    { check_gemv<float>(2.5f, -1.5f); }
TH_TEST(gemv_f64_alpha1_beta1)  { check_gemv<double>(1.0, 1.0); }
TH_TEST(gemv_f64_alpha_beta)    { check_gemv<double>(-0.75, 3.0); }

// beta == 0 must overwrite, not multiply. Seed y with inf/nan: if the
// kernel computes 0*y it yields nan and this fails.
TH_TEST(gemv_beta_zero_overwrites_garbage) {
    Matrix<float> A(4, 4);
    Vector<float> x(4), y(4);
    A.fill(1.0f);
    x.fill(2.0f);
    y[0] = std::numeric_limits<float>::infinity();
    y[1] = -std::numeric_limits<float>::infinity();
    y[2] = std::numeric_limits<float>::quiet_NaN();
    y[3] = 1e30f;

    gemv(1.0f, A, x, 0.0f, y);

    for (std::size_t i = 0; i < 4; ++i) {
        TH_CHECK(!std::isnan(y[i]));
        TH_CHECK(y[i] == 8.0f);   // 4 columns * 1.0 * 2.0
    }
}

TH_TEST(gemv_identity_reproduces_x) {
    const std::size_t n = 33;
    Matrix<double> I(n, n);
    Vector<double> x(n), y(n);
    for (std::size_t i = 0; i < n; ++i) I(i, i) = 1.0;
    fill_pseudo_v(x, 5);
    gemv(1.0, I, x, 0.0, y);
    for (std::size_t i = 0; i < n; ++i) TH_CHECK(y[i] == x[i]);
}

TH_TEST(gemv_shape_mismatch_throws) {
    Matrix<float> A(4, 5);
    Vector<float> bad_x(4), good_x(5), bad_y(5), good_y(4);
    TH_CHECK_THROWS(gemv(1.0f, A, bad_x, 0.0f, good_y), std::invalid_argument);
    TH_CHECK_THROWS(gemv(1.0f, A, good_x, 0.0f, bad_y), std::invalid_argument);
}

// gemm 

template <typename T>
static void check_gemm(T alpha, T beta) {
    for (const Shape3 s : kGemmShapes) {
        Matrix<T> A(s.m, s.k), B(s.k, s.n), C(s.m, s.n);
        fill_pseudo_m(A, 11);
        fill_pseudo_m(B, 22);
        fill_pseudo_m(C, 33);
        Matrix<T> C0 = C.clone();

        gemm(alpha, A, B, beta, C);

        for (std::size_t i = 0; i < s.m; ++i)
            for (std::size_t j = 0; j < s.n; ++j) {
                long double acc = 0.0L;
                for (std::size_t p = 0; p < s.k; ++p)
                    acc += static_cast<long double>(A(i, p)) *
                           static_cast<long double>(B(p, j));
                const long double want =
                    (beta == T(0))
                        ? static_cast<long double>(alpha) * acc
                        : static_cast<long double>(alpha) * acc +
                              static_cast<long double>(beta) *
                                  static_cast<long double>(C0(i, j));
                TH_CHECK_CLOSE(static_cast<double>(C(i, j)),
                               static_cast<double>(want), tol_for<T>(s.k));
            }
    }
}

TH_TEST(gemm_f32_alpha1_beta0) { check_gemm<float>(1.0f, 0.0f); }
TH_TEST(gemm_f32_alpha1_beta1) { check_gemm<float>(1.0f, 1.0f); }
TH_TEST(gemm_f32_alpha_beta)   { check_gemm<float>(2.0f, -0.5f); }
TH_TEST(gemm_f64_alpha1_beta0) { check_gemm<double>(1.0, 0.0); }
TH_TEST(gemm_f64_alpha_beta)   { check_gemm<double>(-1.25, 2.0); }

TH_TEST(gemm_beta_zero_overwrites_garbage) {
    Matrix<float> A(4, 4), B(4, 4), C(4, 4);
    A.fill(1.0f);
    B.fill(2.0f);
    C.fill(std::numeric_limits<float>::quiet_NaN());
    C(0, 0) = std::numeric_limits<float>::infinity();

    gemm(1.0f, A, B, 0.0f, C);

    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) {
            TH_CHECK(!std::isnan(C(i, j)));
            TH_CHECK(C(i, j) == 8.0f);   // k=4 * 1.0 * 2.0
        }
}

TH_TEST(gemm_identity_reproduces_operand) {
    const std::size_t n = 17;
    Matrix<double> I(n, n), A(n, n), C(n, n);
    for (std::size_t i = 0; i < n; ++i) I(i, i) = 1.0;
    fill_pseudo_m(A, 9);
    gemm(1.0, A, I, 0.0, C);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) TH_CHECK(C(i, j) == A(i, j));
}

// A row of zeros in A must give a row of zeros in C, exactly. Catches a
// kernel whose edge-block handling leaks values from an adjacent block.
TH_TEST(gemm_zero_row_stays_zero) {
    const std::size_t m = 9, n = 9, k = 9;
    Matrix<float> A(m, k), B(k, n), C(m, n);
    fill_pseudo_m(A, 1);
    fill_pseudo_m(B, 2);
    for (std::size_t p = 0; p < k; ++p) A(4, p) = 0.0f;

    gemm(1.0f, A, B, 0.0f, C);
    for (std::size_t j = 0; j < n; ++j) TH_CHECK(C(4, j) == 0.0f);
}

TH_TEST(gemm_shape_mismatch_throws) {
    Matrix<float> A(4, 5), Bbad(4, 5), Bgood(5, 6), Cbad(5, 6), Cgood(4, 6);
    TH_CHECK_THROWS(gemm(1.0f, A, Bbad, 0.0f, Cgood), std::invalid_argument);
    TH_CHECK_THROWS(gemm(1.0f, A, Bgood, 0.0f, Cbad), std::invalid_argument);
}

// Matrices allocated by this library have stride > cols whenever cols is
// not a lane multiple. Every kernel must honor lda/ldb/ldc rather than
// assuming packed rows -- a kernel that assumes stride==cols reads the
// wrong elements here and produces garbage.
TH_TEST(gemm_respects_padded_stride) {
    const std::size_t m = 3, n = 3, k = 3;
    Matrix<float> A(m, k), B(k, n), C(m, n);
    TH_CHECK(A.stride() == 16);   // 3 cols padded to 16 -- stride != cols

    // Poison the padding columns. A stride-ignoring kernel folds these in.
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = k; j < A.stride(); ++j) A.row(i)[j] = 1e30f;
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = n; j < B.stride(); ++j) B.row(i)[j] = 1e30f;

    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < k; ++j) A(i, j) = static_cast<float>(i + j);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < n; ++j) B(i, j) = static_cast<float>(i * j + 1);

    gemm(1.0f, A, B, 0.0f, C);

    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            float want = 0.0f;
            for (std::size_t p = 0; p < k; ++p)
                want += static_cast<float>(i + p) * static_cast<float>(p * j + 1);
            TH_CHECK(C(i, j) == want);
        }
}
