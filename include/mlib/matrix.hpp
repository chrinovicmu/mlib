#ifndef MLIB_MATRIX_HPP
#define MLIB_MATRIX_HPP

#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <cstdlib>      
#include <cstring>      
#include <algorithm>    

#include "vector.hpp"

namespace mlib{

template<typename T>
class Matrix {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "Matrix<T> supports only float or double");

public:
    using val_type      = T;
    using ptr         = T*;
    using const_ptr   = const T*;
    using size_type       = size_t;


    Matrix() noexcept = default;

    Matrix(size_type rows, size_type cols) {
        if (rows == 0 || cols == 0) return;

        rows_          = rows;
        cols_          = cols;
        capacity_cols_ = padded_size<T>(cols);

        size_type total_elements = rows_ * capacity_cols_;
        data_ = static_cast<ptr>(
            std::aligned_alloc(simd_alignment, total_elements * sizeof(T)));

        if (!data_) {
            throw std::bad_alloc();
        }
    }

    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;

    Matrix(Matrix&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          rows_(std::exchange(other.rows_, 0)),
          cols_(std::exchange(other.cols_, 0)),
          capacity_cols_(std::exchange(other.capacity_cols_, 0))
    {
    }

    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            destroy();
            data_          = std::exchange(other.data_, nullptr);
            rows_          = std::exchange(other.rows_, 0);
            cols_          = std::exchange(other.cols_, 0);
            capacity_cols_ = std::exchange(other.capacity_cols_, 0);
        }
        return *this;
    }

    ~Matrix() {
        destroy();
    }

    size_type rows() const noexcept { return rows_; }
    size_type cols() const noexcept { return cols_; }
    size_type size() const noexcept { return rows_ * cols_; }

    size_type capacity_cols() const noexcept { return capacity_cols_; }

    bool empty() const noexcept { return rows_ == 0 || cols_ == 0; }

    ptr       data()       noexcept { return data_; }
    const_ptr data() const noexcept { return data_; }

    bool is_aligned() const noexcept {
        return (empty()) ||
               (reinterpret_cast<uintptr_t>(data_) % simd_alignment == 0);
    }

    ptr row(size_type i) noexcept {
        return data_ + i * capacity_cols_;
    }

    const_ptr row(size_type i) const noexcept {
        return data_ + i * capacity_cols_;
    }

    T& operator()(size_type i, size_type j) noexcept {
        return row(i)[j];
    }

    const T& operator()(size_type i, size_type j) const noexcept {
        return row(i)[j];
    }

    void fill(T value) {

        if (empty()) 
            return;
        for (size_type i = 0; i < rows_; ++i) {
            ptr r = row(i);
            for (size_type j = 0; j < cols_; ++j) {
                r[j] = value;
            }
        }
    }

    void set_zero() { fill(T(0)); }

    void clear() noexcept {
        rows_ = cols_ = capacity_cols_ = 0;
    }

private:
    void destroy() noexcept {
        if (data_) {
            std::free(data_);
            data_ = nullptr;
        }
        rows_ = cols_ = capacity_cols_ = 0;
    }

    ptr   data_          = nullptr;
    size_type rows_          = 0;
    size_type cols_          = 0;
    size_type capacity_cols_ = 0;   
};
}
