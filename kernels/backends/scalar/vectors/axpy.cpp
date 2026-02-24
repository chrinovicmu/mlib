// backends/scalar/vectors/axpy.cpp

namespace mlib {
namespace kernels {

template<typename T>
void axpy(T alpha, const T* __restrict x, T* __restrict y, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        y[i] += alpha * x[i];
    }
}

} // namespace kernels
} // namespace mlib
