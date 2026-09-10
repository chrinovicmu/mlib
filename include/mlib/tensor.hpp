#pragma once

#include <cstddef>          
#include <cstdint>          
#include <cstring>           
#include <initializer_list>  
#include <limits>            
#include <new>               
#include <stdexcept>         
#include <type_traits>       
#include <utility>          


namespace mlib {

inline constexpr std::size_t kAlignment = 64;

template <typename T>
inline constexpr std::size_t lanes() noexcept {
   return kAlignment / sizeof(T);
}

template <typename T>
inline constexpr std::size_t padded(std::size_t n) noexcept {
    const std::size_t L = lanes<T>();
    return (n + L - 1) / L * L;
}

namespace detail {

template <typename T>
T* alloc_aligned(std::size_t count) {
    if (count == 0) 
        return nullptr;

    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::length_error("mlib: allocation size overflow");
    }

    void* p = ::operator new(count * sizeof(T), std::align_val_t{kAlignment});
    return static_cast<T*>(p);
}

template <typename T>
void free_aligned(T* p) noexcept {
    if (p) ::operator delete(p, std::align_val_t{kAlignment});
}

template <typename T>
inline constexpr bool is_supported_v =
    std::is_same_v<T, float> || std::is_same_v<T, double>;

}  // namespace detail
//
template <typename T>
class Vector {
    static_assert(detail::is_supported_v<T>,
                  "mlib::Vector<T> supports only float and double");

public:
    using value_type     = T;
    using size_type      = std::size_t;
    using iterator       = T*;
    using const_iterator = const T*;

    Vector() noexcept = default;
    explicit Vector(size_type n)
        : data_(detail::alloc_aligned<T>(padded<T>(n))),
          size_(n),
          capacity_(padded<T>(n)) {
        if (data_) 
            std::memset(data_, 0, capacity_ * sizeof(T));
    }

    Vector(size_type n, T fill) : Vector(n) {
        for (size_type i = 0; i < size_; ++i) data_[i] = fill;
    }

    Vector(std::initializer_list<T> init) : Vector(init.size()) {
        size_type i = 0;
        for (T v : init) data_[i++] = v;
    }

    Vector(const Vector&)            = delete;
    Vector& operator=(const Vector&) = delete;

    Vector(Vector&& o) noexcept
        : data_(std::exchange(o.data_, nullptr)),
          size_(std::exchange(o.size_, 0)),
          capacity_(std::exchange(o.capacity_, 0)) {}

    Vector& operator=(Vector&& o) noexcept {
        if (this != &o) {
            detail::free_aligned(data_);
            data_     = std::exchange(o.data_, nullptr);
            size_     = std::exchange(o.size_, 0);
            capacity_ = std::exchange(o.capacity_, 0);
        }
        return *this;
    }

    ~Vector() { detail::free_aligned(data_); }

    Vector clone() const {
        Vector out(size_);
        if (size_) 
            std::memcpy(out.data_, data_, size_ * sizeof(T));
        return out;
    }

    size_type size()     const noexcept { return size_; }
    size_type capacity() const noexcept { return capacity_; }
    bool      empty()    const noexcept { return size_ == 0; }

    T*       data()       noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    T&       operator[](size_type i)       noexcept { return data_[i]; }
    const T& operator[](size_type i) const noexcept { return data_[i]; }

    T& at(size_type i) {
        if (i >= size_) throw std::out_of_range("mlib::Vector::at");
        return data_[i];
    }
    const T& at(size_type i) const {
        if (i >= size_) throw std::out_of_range("mlib::Vector::at");
        return data_[i];
    }

    iterator       begin()       noexcept { return data_; }
    iterator       end()         noexcept { return data_ + size_; }
    const_iterator begin() const noexcept { return data_; }
    const_iterator end()   const noexcept { return data_ + size_; }

    void fill(T v) noexcept {
        for (size_type i = 0; i < size_; ++i) data_[i] = v;
    }

    void set_zero() noexcept {
        if (data_) 
            std::memset(data_, 0, capacity_ * sizeof(T));
    }

    void resize(size_type n) {
        if (n == size_) 
            return;

        const size_type new_cap = padded<T>(n);

        if (new_cap <= capacity_) {
            size_ = n;
            return;
        }

        T* nd = detail::alloc_aligned<T>(new_cap);
        std::memset(nd, 0, new_cap * sizeof(T));
        if (data_)
            std::memcpy(nd, data_, size_ * sizeof(T));
        detail::free_aligned(data_);
        data_     = nd;
        size_     = n;
        capacity_ = new_cap;
    }

private:
    T*        data_     = nullptr;
    size_type size_     = 0;   // logical element count
    size_type capacity_ = 0;   // allocated element count, >= size_, lane-aligned
};

template <typename T>
class Matrix {
    static_assert(detail::is_supported_v<T>,
                  "mlib::Matrix<T> supports only float and double");

public:
    using value_type = T;
    using size_type  = std::size_t;

    Matrix() noexcept = default;

    Matrix(size_type rows, size_type cols)
        : rows_(rows), cols_(cols), stride_(padded<T>(cols)) {
        if (rows == 0 || cols == 0) {
            rows_ = cols_ = stride_ = 0;
            return;
        }

        if (rows_ > std::numeric_limits<size_type>::max() / stride_) {
            throw std::length_error("mlib: matrix size overflow");
        }

        const size_type total = rows_ * stride_;
        data_ = detail::alloc_aligned<T>(total);
        std::memset(data_, 0, total * sizeof(T));
    }

    Matrix(const Matrix&)            = delete;
    Matrix& operator=(const Matrix&) = delete;

    Matrix(Matrix&& o) noexcept
        : data_(std::exchange(o.data_, nullptr)),
          rows_(std::exchange(o.rows_, 0)),
          cols_(std::exchange(o.cols_, 0)),
          stride_(std::exchange(o.stride_, 0)) {}

    Matrix& operator=(Matrix&& o) noexcept {
        if (this != &o) {                   
            detail::free_aligned(data_);
            data_   = std::exchange(o.data_, nullptr);
            rows_   = std::exchange(o.rows_, 0);
            cols_   = std::exchange(o.cols_, 0);
            stride_ = std::exchange(o.stride_, 0);
        }
        return *this;
    }

    ~Matrix() { detail::free_aligned(data_); }

    Matrix clone() const {
        Matrix out(rows_, cols_);
        if (data_) std::memcpy(out.data_, data_, rows_ * stride_ * sizeof(T));
        return out;
    }

    size_type rows()   const noexcept { return rows_; }
    size_type cols()   const noexcept { return cols_; }

    size_type stride() const noexcept { return stride_; }
    size_type size()   const noexcept { return rows_ * cols_; }
    bool      empty()  const noexcept { return rows_ == 0 || cols_ == 0; }

    T*       data()       noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    T*       row(size_type i)       noexcept { return data_ + i * stride_; }
    const T* row(size_type i) const noexcept { return data_ + i * stride_; }

    T&       operator()(size_type i, size_type j)       noexcept { return row(i)[j]; }
    const T& operator()(size_type i, size_type j) const noexcept { return row(i)[j]; }

    T& at(size_type i, size_type j) {
        if (i >= rows_ || j >= cols_) throw std::out_of_range("mlib::Matrix::at");
        return row(i)[j];
    }
    const T& at(size_type i, size_type j) const {
        if (i >= rows_ || j >= cols_) throw std::out_of_range("mlib::Matrix::at");
        return row(i)[j];
    }

    void fill(T v) noexcept {
        for (size_type i = 0; i < rows_; ++i) {
            T* r = row(i);                                   // hoist row base
            for (size_type j = 0; j < cols_; ++j) r[j] = v;
        }
    }

    void set_zero() noexcept {
        if (data_) 
            std::memset(data_, 0, rows_ * stride_ * sizeof(T));
    }

private:
    T*        data_   = nullptr;
    size_type rows_   = 0;
    size_type cols_   = 0;   // logical columns
    size_type stride_ = 0;   // allocated columns, >= cols_, lane-aligned
};

}  // namespace mlib
}
