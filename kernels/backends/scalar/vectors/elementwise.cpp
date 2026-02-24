#include <cmath>  

namespace mlib {
namespace kernels {

template<typename T>
void add(const T* __restrict x, const T* __restrict y, T* __restrict z, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        z[i] = x[i] + y[i];
    }
}

template<typename T>
void sub(const T* __restrict x, const T* __restrict y, T* __restrict z, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        z[i] = x[i] - y[i];
    }
}

template<typename T>
void mul(const T* __restrict x, const T* __restrict y, T* __restrict z, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        z[i] = x[i] * y[i];
    }
}

template<typename T>
void div(const T* __restrict x, const T* __restrict y, T* __restrict z, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        z[i] = x[i] / y[i];
    }
}

template<typename T>
void abs(const T* __restrict x, T* __restrict y, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        y[i] = std::abs(x[i]);
    }
}

template<typename T>
void neg(const T* __restrict x, T* __restrict y, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        y[i] = -x[i];
    }
}

} // namespace kernels
} // namespace mlib
