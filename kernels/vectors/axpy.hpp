#ifndef MLIB_KERNELS_AXPY_HPP
#define MLIB_KERNELS_AXPY_HPP

#include <mlib/vector.hpp> 
#include <cstdeff> 
#include <stdexcept> 

namespace mlib{
namespace  kernels{

template<typename T>
void axpy(T alpha, const Vector<T>& x, Vector<T>& y); 

inline void axpy_p32(float alpha, const Vector<float>& x, Vector<float>& y) {
    axpy(alpha, x, y);
}

inline void axpy_p64(double alpha, const Vector<double>& x, Vector<double>& y) {
    axpy(alpha, x, y);
}
}
}

#if defined(__AVX2__)
    #include "../avx2/axpy.cpp"
#elif defined(__ARM_NEON)
    #include "../neon/axpy.cpp"
#else
    #include "../scalar/axpy.cpp"
#endif

#endif

