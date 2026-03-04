
#include <cmath>

namespace mlib {
namespace kernels {

template<typename T>
T l1_norm(const T* __restrict x, size_t count) {
    T sum = T(0);
    for (size_t i = 0; i < count; ++i) {
        sum += std::abs(x[i]);
    }
    return sum;
}

template<typename T>
T l2_norm(const T* __restrict x, size_t count) {
    T sum_sq = T(0);
    for (size_t i = 0; i < count; ++i) {
        sum_sq += x[i] * x[i];
    }
    return std::sqrt(sum_sq);
}

} // namespace kernes
} // namespace mlib
