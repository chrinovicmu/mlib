#ifndef MLIB_KERNELS_SWAP_HPP
#define MLIB_KERNELS_SWAP_HPP

#include <mlib/vector.hpp>
#include <cstddef>
#include <stdexcept>

namespace mlib {
namespace kernels {

template<typename T>
void swap(Vector<T>& x, Vector<T>& y);

inline void swap_p32(Vector<float>& x, Vector<float>& y) {
    swap(x, y);
}

inline void swap_p64(Vector<double>& x, Vector<double>& y) {
    swap(x, y);
}

} // namespace kernels
} // namespace mlib

#if defined(__AVX2__)
    #include "../avx2/swap.cpp"
#elif defined(__ARM_NEON)
    #include "../neon/swap.cpp"
#else
    #include "../scalar/swap.cpp"
#endif

#endif
