// Exercise the raw entry points, including paths hidden by API fast returns.
// Guard pages catch assembly overreads that sanitizers cannot instrument.
#include "harness.hpp"
#include <mlib/mlib.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

template <typename T> struct Kernels;
#define TEST_KERNELS(T, suffix)                                                \
    template <> struct Kernels<T> {                                           \
        static constexpr auto axpy = mlib::kernels::active::axpy_##suffix;     \
        static constexpr auto scal = mlib::kernels::active::scal_##suffix;     \
        static constexpr auto copy = mlib::kernels::active::copy_##suffix;     \
        static constexpr auto swap = mlib::kernels::active::swap_##suffix;     \
        static constexpr auto dot = mlib::kernels::active::dot_##suffix;       \
        static constexpr auto nrm2 = mlib::kernels::active::nrm2_##suffix;      \
        static constexpr auto asum = mlib::kernels::active::asum_##suffix;      \
        static constexpr auto gemv = mlib::kernels::active::gemv_##suffix;      \
        static constexpr auto gemm = mlib::kernels::active::gemm_##suffix;      \
    }
TEST_KERNELS(float, f32);
TEST_KERNELS(double, f64);
#undef TEST_KERNELS

// Place the logical end directly before an inaccessible page. Odd sizes
// also exercise addresses aligned to an element but not to a SIMD vector.
template <typename T>
class BoundedBuffer {
public:
    explicit BoundedBuffer(std::size_t count) : count_(count) {
#if defined(__unix__) || defined(__APPLE__)
        const long page_result = sysconf(_SC_PAGESIZE);
        if (page_result <= 0) throw std::runtime_error("sysconf page size failed");
        const auto page = static_cast<std::size_t>(page_result);
        const auto accessible = ((count + 1) * sizeof(T) + page - 1) / page * page;
        mapped_bytes_ = accessible + page;
        mapping_ = mmap(nullptr, mapped_bytes_, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping_ == MAP_FAILED) throw std::runtime_error("mmap failed");
        auto* boundary = static_cast<char*>(mapping_) + accessible;
        if (mprotect(boundary, page, PROT_NONE) != 0) {
            munmap(mapping_, mapped_bytes_);
            throw std::runtime_error("mprotect failed");
        }
        data_ = reinterpret_cast<T*>(boundary) - count;
#else
        storage_.resize(count + 2, sentinel());
        data_ = storage_.data() + 1;
#endif
        data_[-1] = sentinel();
    }
    ~BoundedBuffer() {
#if defined(__unix__) || defined(__APPLE__)
        munmap(mapping_, mapped_bytes_);
#endif
    }
    BoundedBuffer(const BoundedBuffer&) = delete;
    BoundedBuffer& operator=(const BoundedBuffer&) = delete;
    T* data() { return data_; }
    T& operator[](std::size_t i) { return data_[i]; }
    void check_boundary() {
        TH_CHECK(data_[-1] == sentinel());
#if !defined(__unix__) && !defined(__APPLE__)
        TH_CHECK(data_[count_] == sentinel());
#endif
    }
private:
    static constexpr T sentinel() { return T(123456); }
    [[maybe_unused]] std::size_t count_;
    T* data_ = nullptr;
#if defined(__unix__) || defined(__APPLE__)
    void* mapping_ = nullptr;
    std::size_t mapped_bytes_ = 0;
#else
    std::vector<T> storage_;
#endif
};

constexpr std::size_t sizes[] = {
    0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33,
    63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513};

template <typename T> T x_value(std::size_t i) {
    return T(static_cast<int>(i % 29) - 14) / T(8);
}
template <typename T> T y_value(std::size_t i) {
    return T(static_cast<int>(i % 19) - 9) / T(16);
}
template <typename T> double tolerance(std::size_t terms) {
    return double(std::numeric_limits<T>::epsilon()) * double(terms + 8) * 4;
}

template <typename T> void check_vectors() {
    using K = Kernels<T>;
    for (auto n : sizes) {
        BoundedBuffer<T> x(n), y(n);
        long double dot = 0, squared = 0, absolute = 0;
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = x_value<T>(i);
            y[i] = y_value<T>(i);
            dot += static_cast<long double>(x[i]) * y[i];
            squared += static_cast<long double>(x[i]) * x[i];
            absolute += std::abs(static_cast<long double>(x[i]));
        }
        TH_CHECK_CLOSE(double(K::dot(n, x.data(), y.data())), double(dot), tolerance<T>(n));
        TH_CHECK_CLOSE(double(K::nrm2(n, x.data())), double(std::sqrt(squared)), tolerance<T>(n));
        TH_CHECK_CLOSE(double(K::asum(n, x.data())), double(absolute), tolerance<T>(n));
        K::axpy(n, T(1.25), x.data(), y.data());
        for (std::size_t i = 0; i < n; ++i) {
            TH_CHECK_CLOSE(y[i], T(1.25) * x_value<T>(i) + y_value<T>(i), tolerance<T>(1));
        }
        K::scal(n, T(-0.5), x.data());
        for (std::size_t i = 0; i < n; ++i) TH_CHECK(x[i] == T(-0.5) * x_value<T>(i));
        K::copy(n, x.data(), y.data());
        for (std::size_t i = 0; i < n; ++i) {
            TH_CHECK(y[i] == x[i]);
            y[i] = y_value<T>(i);
        }
        K::swap(n, x.data(), y.data());
        for (std::size_t i = 0; i < n; ++i) {
            TH_CHECK(x[i] == y_value<T>(i));
            TH_CHECK(y[i] == T(-0.5) * x_value<T>(i));
        }
        x.check_boundary();
        y.check_boundary();
    }
}

template <typename T> void check_gemv() {
    using K = Kernels<T>;
    const T nan = std::numeric_limits<T>::quiet_NaN();
    const T betas[] = {T(0), -T(0), T(1), T(-0.5), nan};
    for (auto n : sizes) {
        constexpr std::size_t m = 3;
        const auto lda = n + 5;
        BoundedBuffer<T> a(m * lda), x(n), y(m);
        std::fill_n(a.data(), m * lda, nan);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j) a[i * lda + j] = x_value<T>(i + j);
        for (std::size_t j = 0; j < n; ++j) x[j] = y_value<T>(j);
        for (T beta : betas) {
            for (std::size_t i = 0; i < m; ++i) y[i] = beta == T(0) ? nan : y_value<T>(i);
            K::gemv(m, n, T(-1.25), a.data(), lda, x.data(), beta, y.data());
            for (std::size_t i = 0; i < m; ++i) {
                long double sum = 0;
                for (std::size_t j = 0; j < n; ++j)
                    sum += static_cast<long double>(x_value<T>(i + j)) * y_value<T>(j);
                long double want = -1.25L * sum;
                if (beta != T(0)) want += static_cast<long double>(beta) * y_value<T>(i);
                TH_CHECK_CLOSE(double(y[i]), double(want), tolerance<T>(n));
                for (std::size_t j = n; j < lda; ++j) TH_CHECK(std::isnan(a[i * lda + j]));
            }
        }
        a.check_boundary();
        x.check_boundary();
        y.check_boundary();
    }
}

template <typename T> void check_gemm() {
    using K = Kernels<T>;
    const T nan = std::numeric_limits<T>::quiet_NaN();
    const T betas[] = {T(0), -T(0), T(1), T(-0.5), nan};
    for (std::size_t n : {0, 1, 7, 8, 9, 15, 16, 17, 33, 65, 129}) {
        for (std::size_t k : {0, 5, 17}) {
            constexpr std::size_t m = 3;
            const auto lda = k + 2, ldb = n + 3, ldc = n + 5;
            BoundedBuffer<T> a(m * lda), b(k * ldb), c(m * ldc);
            std::fill_n(a.data(), m * lda, nan);
            std::fill_n(b.data(), k * ldb, nan);
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t p = 0; p < k; ++p) a[i * lda + p] = x_value<T>(i + p);
            for (std::size_t p = 0; p < k; ++p)
                for (std::size_t j = 0; j < n; ++j) b[p * ldb + j] = y_value<T>(p + j);
            for (T beta : betas) {
                std::fill_n(c.data(), m * ldc, T(123456));
                for (std::size_t i = 0; i < m; ++i)
                    for (std::size_t j = 0; j < n; ++j)
                        c[i * ldc + j] = beta == T(0) ? nan : y_value<T>(i + j);
                K::gemm(m, n, k, T(0.75), a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
                for (std::size_t i = 0; i < m; ++i) {
                    for (std::size_t j = 0; j < n; ++j) {
                        // Independent i-j-p reference, with wider accumulation.
                        long double sum = 0;
                        for (std::size_t p = 0; p < k; ++p)
                            sum += static_cast<long double>(x_value<T>(i + p)) * y_value<T>(p + j);
                        long double want = 0.75L * sum;
                        if (beta != T(0)) want += static_cast<long double>(beta) * y_value<T>(i + j);
                        TH_CHECK_CLOSE(double(c[i * ldc + j]), double(want), tolerance<T>(k));
                    }
                    for (std::size_t j = n; j < ldc; ++j) TH_CHECK(c[i * ldc + j] == T(123456));
                }
            }
            a.check_boundary();
            b.check_boundary();
            c.check_boundary();
        }
    }
}

template <typename T> void check_zero_calls() {
    using K = Kernels<T>;
    K::axpy(0, T(2), nullptr, nullptr);
    K::scal(0, T(2), nullptr);
    K::copy(0, nullptr, nullptr);
    K::swap(0, nullptr, nullptr);
    TH_CHECK(K::dot(0, nullptr, nullptr) == T(0));
    TH_CHECK(K::nrm2(0, nullptr) == T(0));
    TH_CHECK(K::asum(0, nullptr) == T(0));
    K::gemv(0, 17, T(2), nullptr, 17, nullptr, T(0), nullptr);
    K::gemm(0, 17, 5, T(2), nullptr, 5, nullptr, 17, T(0), nullptr, 17);
}

template <typename T> void check_nonfinite() {
    using K = Kernels<T>;
    const T nan = std::numeric_limits<T>::quiet_NaN();
    // Check a full vector plus a tail, for both f32 and f64.
    BoundedBuffer<T> x(33), y(33);
    std::fill_n(x.data(), 33, nan);
    K::scal(33, T(0), x.data());
    for (std::size_t i = 0; i < 33; ++i) TH_CHECK(std::isnan(x[i]));
    TH_CHECK(std::isnan(K::asum(33, x.data())));
    std::fill_n(x.data(), 33, -std::numeric_limits<T>::infinity());
    TH_CHECK(K::asum(33, x.data()) == std::numeric_limits<T>::infinity());
    std::fill_n(y.data(), 33, T(1));
    T a = T(0);
    // An unordered comparison must not mistake NaN alpha for zero.
    K::gemm(1, 33, 1, nan, &a, 1, y.data(), 33, T(0), x.data(), 33);
    for (std::size_t i = 0; i < 33; ++i) TH_CHECK(std::isnan(x[i]));
    // The existing zero-aik shortcut must skip NaN input rows in B.
    std::fill_n(y.data(), 33, nan);
    K::gemm(1, 33, 1, T(1), &a, 1, y.data(), 33, T(0), x.data(), 33);
    for (std::size_t i = 0; i < 33; ++i) TH_CHECK(x[i] == T(0));
    x.check_boundary();
    y.check_boundary();
}
} // namespace

TH_TEST(raw_vectors_f32) { check_vectors<float>(); }
TH_TEST(raw_vectors_f64) { check_vectors<double>(); }
TH_TEST(raw_gemv_f32) { check_gemv<float>(); }
TH_TEST(raw_gemv_f64) { check_gemv<double>(); }
TH_TEST(raw_gemm_f32) { check_gemm<float>(); }
TH_TEST(raw_gemm_f64) { check_gemm<double>(); }
TH_TEST(raw_zero_calls_f32) { check_zero_calls<float>(); }
TH_TEST(raw_zero_calls_f64) { check_zero_calls<double>(); }
TH_TEST(raw_nonfinite_f32) { check_nonfinite<float>(); }
TH_TEST(raw_nonfinite_f64) { check_nonfinite<double>(); }
