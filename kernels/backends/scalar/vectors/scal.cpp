// backends/scalar/vectors/scal.cpp
#include <cstddef>
namespace mlib {
namespace kernels {

template<typename T>
void scal(T alpha, T* __restrict x, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        x[i] *= alpha;
    }
}

} // namespace kernels
} // namespace mlib
