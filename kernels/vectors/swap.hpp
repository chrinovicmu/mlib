#ifndef MLIB_KERNELS_SWAP_HPP
#define MLIB_KERNELS_SWAP_HPP

#include <mlib/vector.hpp>
#include <cstddef>
#include <stdexcept>

namespace mlib {
namespace kernels {

template<typename T>
void swap(T* __restrict x, T* __restrict y, size_t count);

inline void swap_p32(Vector<float>& x, Vector<float>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector sizes must match in swap_p32");
    }
    if (x.empty()) {
        return;
    }

    swap(x.aligned_data(), y.aligned_data(), x.size());
}

inline void swap_p64(Vector<double>& x, Vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector sizes must match in swap_p64");
    }
    if (x.empty()) {
        return;
    }

    swap(x.aligned_data(), y.aligned_data(), x.size());
}

} // namespace kernels
} // namespace mlib

#if defined(__AVX2__)
    #include "../../backends/avx2/vectors/swap.cpp"
#elif defined(__ARM_NEON)
    #include "../../backends/neon/vectors/swap.cpp"
#else
    #include "../../backends/scalar/vectors/swap.cpp"
#endif

#endif
