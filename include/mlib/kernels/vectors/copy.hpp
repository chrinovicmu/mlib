#ifndef MLIB_KERNELS_COPY_HPP
#define MLIB_KERNELS_COPY_HPP

#include <mlib/vector.hpp>
#include <cstddef>
#include <stdexcept>

namespace mlib {
namespace kernels {

template<typename T>
void copy(const T* __restrict src, T* __restrict dst, size_t count);

inline void copy_p32(const Vector<float>& x, Vector<float>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector sizes must match in copy_p32");
    }
    if (x.empty()) {
        return;
    }

    copy(x.aligned_data(), y.aligned_data(), x.size());
}

inline void copy_p64(const Vector<double>& x, Vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector sizes must match in copy_p64");
    }
    if (x.empty()) {
        return;
    }

    copy(x.aligned_data(), y.aligned_data(), x.size());
}

} // namespace kernels
} // namespace mlib

#if defined(__AVX2__)
    #include "../../backends/avx2/vectors/copy.cpp"
#elif defined(__ARM_NEON)
    #include "../../backends/neon/vectors/copy.cpp"
#else
    #include "../../backends/scalar/vectors/copy.cpp"
#endif

#endif
