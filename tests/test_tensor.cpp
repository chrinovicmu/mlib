// tests/test_tensor.cpp

#include "harness.hpp"
#include <mlib/mlib.hpp>

#include <cstdint>

using namespace mlib;

namespace {
bool is_aligned(const void* p, std::size_t a) {
    return (reinterpret_cast<std::uintptr_t>(p) % a) == 0;
}
}  // namespace

// Contract (1): every allocation is kAlignment-aligned, at any size --
// including sizes far from a lane boundary.
TH_TEST(vector_base_alignment) {
    for (std::size_t n : {1u, 3u, 7u, 16u, 17u, 100u, 1000u}) {
        Vector<float>  vf(n);
        Vector<double> vd(n);
        TH_CHECK(is_aligned(vf.data(), kAlignment));
        TH_CHECK(is_aligned(vd.data(), kAlignment));
    }
}

// Contract (2): capacity is size rounded up to a lane multiple, and
// size() itself is unchanged by that rounding.
TH_TEST(vector_padding) {
    TH_CHECK(lanes<float>() == 16);
    TH_CHECK(lanes<double>() == 8);

    Vector<float> a(1);
    TH_CHECK(a.size() == 1);
    TH_CHECK(a.capacity() == 16);

    Vector<float> b(16);
    TH_CHECK(b.capacity() == 16);

    Vector<float> c(17);
    TH_CHECK(c.capacity() == 32);

    Vector<double> d(9);
    TH_CHECK(d.size() == 9);
    TH_CHECK(d.capacity() == 16);
}

// Contract (2): padding is zero at construction. A kernel that reads
// padding before writing it must see 0.0, not a trap value.
TH_TEST(vector_padding_zeroed) {
    Vector<float> v(5);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = 1.0f;
    for (std::size_t i = v.size(); i < v.capacity(); ++i) {
        TH_CHECK(v.data()[i] == 0.0f);
    }
}

TH_TEST(vector_empty) {
    Vector<float> v;
    TH_CHECK(v.empty());
    TH_CHECK(v.size() == 0);
    TH_CHECK(v.data() == nullptr);

    Vector<float> z(0);
    TH_CHECK(z.empty());
    TH_CHECK(z.data() == nullptr);
}

// Contract (3): move transfers ownership and leaves the source empty.
TH_TEST(vector_move_semantics) {
    Vector<float> a(10);
    a.fill(3.0f);
    const float* orig = a.data();

    Vector<float> b(std::move(a));
    TH_CHECK(b.data() == orig);
    TH_CHECK(b.size() == 10);
    TH_CHECK(a.data() == nullptr);
    TH_CHECK(a.size() == 0);

    Vector<float> c(2);
    c = std::move(b);
    TH_CHECK(c.data() == orig);
    TH_CHECK(c.size() == 10);
    TH_CHECK(b.data() == nullptr);
}

// clone() is the only deep copy, and it is a real one -- distinct storage.
TH_TEST(vector_clone) {
    Vector<float> a{1.0f, 2.0f, 3.0f};
    Vector<float> b = a.clone();
    TH_CHECK(b.size() == 3);
    TH_CHECK(b.data() != a.data());
    TH_CHECK(b[0] == 1.0f && b[1] == 2.0f && b[2] == 3.0f);
    b[0] = 99.0f;
    TH_CHECK(a[0] == 1.0f);
}

TH_TEST(vector_init_list_and_iteration) {
    Vector<double> v{1.0, 2.0, 3.0, 4.0};
    TH_CHECK(v.size() == 4);
    double sum = 0.0;
    for (double x : v) sum += x;
    TH_CHECK(sum == 10.0);
    TH_CHECK(v.end() - v.begin() == 4);
}

TH_TEST(vector_resize) {
    Vector<float> v(4);
    for (std::size_t i = 0; i < 4; ++i) v[i] = static_cast<float>(i);

    // Shrink inside existing capacity: no reallocation, data preserved.
    v.resize(2);
    TH_CHECK(v.size() == 2);
    TH_CHECK(v[0] == 0.0f && v[1] == 1.0f);

    // Grow past capacity: reallocates, preserves the live prefix.
    v.resize(100);
    TH_CHECK(v.size() == 100);
    TH_CHECK(v.capacity() == 112);   // 100 -> next multiple of 16
    TH_CHECK(v[0] == 0.0f && v[1] == 1.0f);
}

TH_TEST(vector_at_bounds_checked) {
    Vector<float> v(4);
    TH_CHECK_THROWS(v.at(4), std::out_of_range);
    TH_CHECK_THROWS(v.at(1000), std::out_of_range);
}

// Contract (4): stride is cols rounded to a lane multiple, and EVERY row
// -- not just row 0 -- lands on a kAlignment boundary as a result.
TH_TEST(matrix_stride_and_row_alignment) {
    Matrix<float> m(5, 3);
    TH_CHECK(m.rows() == 5);
    TH_CHECK(m.cols() == 3);
    TH_CHECK(m.stride() == 16);

    for (std::size_t i = 0; i < m.rows(); ++i) {
        TH_CHECK(is_aligned(m.row(i), kAlignment));
    }

    Matrix<double> d(3, 9);
    TH_CHECK(d.stride() == 16);
    for (std::size_t i = 0; i < d.rows(); ++i) {
        TH_CHECK(is_aligned(d.row(i), kAlignment));
    }
}

TH_TEST(matrix_indexing_is_row_major) {
    Matrix<float> m(3, 4);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            m(i, j) = static_cast<float>(i * 10 + j);

    TH_CHECK(m(0, 0) == 0.0f);
    TH_CHECK(m(2, 3) == 23.0f);
    // Row-major with padding: element (i,j) lives at data()[i*stride + j].
    TH_CHECK(m.data()[1 * m.stride() + 2] == 12.0f);
}

TH_TEST(matrix_empty_and_bounds) {
    Matrix<float> e(0, 5);
    TH_CHECK(e.empty());
    TH_CHECK(e.data() == nullptr);
    TH_CHECK(e.stride() == 0);

    Matrix<float> m(2, 2);
    TH_CHECK_THROWS(m.at(2, 0), std::out_of_range);
    TH_CHECK_THROWS(m.at(0, 2), std::out_of_range);
}

TH_TEST(matrix_move_and_clone) {
    Matrix<float> a(2, 3);
    a.fill(7.0f);
    const float* orig = a.data();

    Matrix<float> b(std::move(a));
    TH_CHECK(b.data() == orig);
    TH_CHECK(a.data() == nullptr);

    Matrix<float> c = b.clone();
    TH_CHECK(c.data() != b.data());
    TH_CHECK(c(1, 2) == 7.0f);
    TH_CHECK(c.stride() == b.stride());
}
