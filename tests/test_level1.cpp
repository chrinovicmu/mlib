// tests/test_level1.cpp
//
// Level-1 correctness: axpy, scal, copy, swap, dot, nrm2, asum.
//

#include "harness.hpp"
#include <mlib/mlib.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using namespace mlib;

namespace {

// Sizes that straddle every lane boundary a backend might use.
const std::vector<std::size_t> kSizes = {
    0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33,
    63, 64, 65, 127, 128, 129, 257, 1000};

// Deterministic fill. A fixed LCG rather than <random> so that a failure
// at a given size reproduces exactly, on any platform, with no seed to
// record.
template <typename T>
void fill_pseudo(Vector<T>& v, std::uint32_t seed) {
    std::uint32_t s = seed;
    for (std::size_t i = 0; i < v.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        // Map to roughly [-1, 1); small magnitudes keep the reference and
        // the kernel in the same exponent range so tolerance stays
        // meaningful.
        v[i] = static_cast<T>(static_cast<double>(s >> 8) / 8388608.0 - 1.0);
    }
}

// Tolerance scales with n: reassociated summation error grows with the
// number of terms. The constant is empirical headroom over the worst
// reassociation a lane-parallel kernel can produce.
template <typename T>
double tol_for(std::size_t n) {
    const double eps = static_cast<double>(std::numeric_limits<T>::epsilon());
    return eps * static_cast<double>(n + 8) * 4.0;
}

}  // namespace

//  axpy 

template <typename T>
static void check_axpy(T alpha) {
    for (std::size_t n : kSizes) {
        Vector<T> x(n), y(n);
        fill_pseudo(x, 11);
        fill_pseudo(y, 22);
        Vector<T> y0 = y.clone();

        axpy(alpha, x, y);

        for (std::size_t i = 0; i < n; ++i) {
            const long double want =
                static_cast<long double>(y0[i]) +
                static_cast<long double>(alpha) * static_cast<long double>(x[i]);
            TH_CHECK_CLOSE(static_cast<double>(y[i]),
                           static_cast<double>(want), tol_for<T>(1));
        }
    }
}

TH_TEST(axpy_f32) { check_axpy<float>(2.5f); }
TH_TEST(axpy_f64) { check_axpy<double>(2.5); }

// alpha == 0 is short-circuited in the wrapper; y must be untouched.
TH_TEST(axpy_alpha_zero_is_noop) {
    Vector<float> x(64), y(64);
    fill_pseudo(x, 1);
    fill_pseudo(y, 2);
    Vector<float> y0 = y.clone();
    axpy(0.0f, x, y);
    for (std::size_t i = 0; i < 64; ++i) TH_CHECK(y[i] == y0[i]);
}

TH_TEST(axpy_negative_alpha) { check_axpy<float>(-1.75f); }

// scal

template <typename T>
static void check_scal(T alpha) {
    for (std::size_t n : kSizes) {
        Vector<T> x(n);
        fill_pseudo(x, 33);
        Vector<T> x0 = x.clone();

        scal(alpha, x);

        for (std::size_t i = 0; i < n; ++i) {
            const long double want =
                static_cast<long double>(x0[i]) * static_cast<long double>(alpha);
            TH_CHECK_CLOSE(static_cast<double>(x[i]),
                           static_cast<double>(want), tol_for<T>(1));
        }
    }
}

TH_TEST(scal_f32) { check_scal<float>(3.25f); }
TH_TEST(scal_f64) { check_scal<double>(-0.5); }

// scal by 0 must zero the vector -- it is NOT short-circuited, unlike
// axpy's alpha==0, because the result differs from the input.
TH_TEST(scal_by_zero_zeroes) {
    Vector<float> x(37);
    fill_pseudo(x, 4);
    scal(0.0f, x);
    for (std::size_t i = 0; i < x.size(); ++i) TH_CHECK(x[i] == 0.0f);
}

//  copy / swap 

TH_TEST(copy_all_sizes) {
    for (std::size_t n : kSizes) {
        Vector<float> x(n), y(n);
        fill_pseudo(x, 55);
        y.fill(-999.0f);
        copy(x, y);
        for (std::size_t i = 0; i < n; ++i) TH_CHECK(y[i] == x[i]);
    }
}

TH_TEST(swap_all_sizes) {
    for (std::size_t n : kSizes) {
        Vector<double> x(n), y(n);
        fill_pseudo(x, 66);
        fill_pseudo(y, 77);
        Vector<double> x0 = x.clone();
        Vector<double> y0 = y.clone();
        swap(x, y);
        for (std::size_t i = 0; i < n; ++i) {
            TH_CHECK(x[i] == y0[i]);
            TH_CHECK(y[i] == x0[i]);
        }
    }
}

// dot 

template <typename T>
static void check_dot() {
    for (std::size_t n : kSizes) {
        Vector<T> x(n), y(n);
        fill_pseudo(x, 88);
        fill_pseudo(y, 99);

        long double want = 0.0L;
        for (std::size_t i = 0; i < n; ++i)
            want += static_cast<long double>(x[i]) * static_cast<long double>(y[i]);

        TH_CHECK_CLOSE(static_cast<double>(dot(x, y)),
                       static_cast<double>(want), tol_for<T>(n));
    }
}

TH_TEST(dot_f32) { check_dot<float>(); }
TH_TEST(dot_f64) { check_dot<double>(); }

TH_TEST(dot_empty_is_zero) {
    Vector<float> a, b;
    TH_CHECK(dot(a, b) == 0.0f);
}

// Orthogonal inputs: the result is exactly zero, with no reassociation
// slack to hide behind. A kernel that drops or double-counts a lane in
// the tail shows up here even when the tolerance test would pass.
TH_TEST(dot_orthogonal_is_exact_zero) {
    for (std::size_t n : {16u, 17u, 64u, 65u, 128u}) {
        Vector<float> x(n), y(n);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = (i % 2 == 0) ? 1.0f : 0.0f;
            y[i] = (i % 2 == 0) ? 0.0f : 1.0f;
        }
        TH_CHECK(dot(x, y) == 0.0f);
    }
}

// nrm2 / asum 

template <typename T>
static void check_norms() {
    for (std::size_t n : kSizes) {
        Vector<T> x(n);
        fill_pseudo(x, 123);

        long double sq = 0.0L, ab = 0.0L;
        for (std::size_t i = 0; i < n; ++i) {
            const long double v = static_cast<long double>(x[i]);
            sq += v * v;
            ab += (v < 0 ? -v : v);
        }

        TH_CHECK_CLOSE(static_cast<double>(nrm2(x)),
                       static_cast<double>(std::sqrt(sq)), tol_for<T>(n));
        TH_CHECK_CLOSE(static_cast<double>(asum(x)),
                       static_cast<double>(ab), tol_for<T>(n));
    }
}

TH_TEST(nrm2_and_asum_f32) { check_norms<float>(); }
TH_TEST(nrm2_and_asum_f64) { check_norms<double>(); }

// Exact known answer: 3-4-5 triangle, no rounding involved.
TH_TEST(nrm2_exact_case) {
    Vector<float> v{3.0f, 4.0f};
    TH_CHECK(nrm2(v) == 5.0f);
    Vector<double> d{3.0, 4.0};
    TH_CHECK(nrm2(d) == 5.0);
}

TH_TEST(asum_uses_absolute_values) {
    Vector<float> v{-1.0f, 2.0f, -3.0f, 4.0f};
    TH_CHECK(asum(v) == 10.0f);
}

TH_TEST(norms_of_empty_are_zero) {
    Vector<double> e;
    TH_CHECK(nrm2(e) == 0.0);
    TH_CHECK(asum(e) == 0.0);
}

// shape validation 
//
// Mismatched shapes must throw from the wrapper, before any kernel runs.
// Kernels themselves never validate, so this is the only line of defense.

TH_TEST(mismatched_sizes_throw) {
    Vector<float> a(8), b(9);
    TH_CHECK_THROWS(axpy(1.0f, a, b), std::invalid_argument);
    TH_CHECK_THROWS(dot(a, b), std::invalid_argument);
    TH_CHECK_THROWS(copy(a, b), std::invalid_argument);
    TH_CHECK_THROWS(swap(a, b), std::invalid_argument);
}

// padding independence 
//
// Contract (2) says padding holds indeterminate values that a kernel may
// freely clobber but must never fold into a result. Poisoning the padding
// with values that would visibly corrupt any op that read them proves no
// reduction is over-reading past size().

TH_TEST(results_ignore_padding_contents) {
    for (std::size_t n : {1u, 5u, 17u, 33u, 65u}) {
        Vector<float> x(n), y(n);
        fill_pseudo(x, 7);
        fill_pseudo(y, 8);

        // Poison every padding slot in both operands.
        for (std::size_t i = n; i < x.capacity(); ++i) x.data()[i] = 1e30f;
        for (std::size_t i = n; i < y.capacity(); ++i) y.data()[i] = 1e30f;

        long double want_dot = 0.0L, want_sq = 0.0L, want_abs = 0.0L;
        for (std::size_t i = 0; i < n; ++i) {
            const long double xv = static_cast<long double>(x[i]);
            want_dot += xv * static_cast<long double>(y[i]);
            want_sq  += xv * xv;
            want_abs += (xv < 0 ? -xv : xv);
        }

        TH_CHECK_CLOSE(static_cast<double>(dot(x, y)),
                       static_cast<double>(want_dot), tol_for<float>(n));
        TH_CHECK_CLOSE(static_cast<double>(nrm2(x)),
                       static_cast<double>(std::sqrt(want_sq)), tol_for<float>(n));
        TH_CHECK_CLOSE(static_cast<double>(asum(x)),
                       static_cast<double>(want_abs), tol_for<float>(n));
    }
}
