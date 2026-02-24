#ifndef MLIB_KERNELS_COPY_HPP
#define MLIB_KERNELS_COPY_HPP

#include <mlib/vector.hpp>
#include <cstddef>
#include <stdexcept>

namespace mlib {
namespace kernels {

template<typename T>
void copy(const Vector<T>& x, Vector<T>& y);

inline void copy_p32(const Vector<float>& x, Vector<float>& y) {
    copy(x, y);
}

inline void copy_p64(const Vector<double>& x, Vector<double>& y) {
    copy(x, y);
}

} // namespace kernels
} // namespace mlib

#if defined(__AVX2__)
    #include "../avx2/copy.cpp"
#elif defined(__ARM_NEON)
    #include "../neon/copy.cpp"
#else
    #include "../scalar/copy.cpp"
#endif

#endif
