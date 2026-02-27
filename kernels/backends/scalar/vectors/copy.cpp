#include <cstddef>
namespace mlib {
namespace kernels {

template<typename T>
void copy(const T* __restrict src, T* __restrict dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = src[i];
    }
}

} // namespace kernels
} // namespace mlib
