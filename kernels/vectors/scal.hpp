#ifndef MLIB_KERNELS_SCAL_HPP
#define MLIB_KERNELS_SCAL_HPP

#include <mlib/vector.hpp>
#include <cstddef>
#include <stdexcept>

namespace mlib {
namespace kernels {

template<typename T>
void scal(T alpha, Vector<T>& x);

inline void scal_p32(float alpha, Vector<float>& x) {
    scal(alpha, x);
}

inline void scal_p64(double alpha, Vector<double>& x) {
    scal(alpha, x);
}

} // namespace kernels
} // namespace mlib

#if defined(__AVX2__)
    #include "../avx2/scal.cpp"
#elif defined(__ARM_NEON)
    #include "../neon/scal.cpp"
#else
    #include "../scalar/scal.cpp"
#endif

#endif
