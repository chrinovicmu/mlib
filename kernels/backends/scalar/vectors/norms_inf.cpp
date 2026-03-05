
#include <cmath>

namespace mlib {
namespace kernels {

template<typename T>

T norm_inf(const T* __restrict x, size_t count) {
    T max_abs = T(0);

    for (size_t i = 0; i < count; ++i) {
        T abs_val = std::abs(x[i]);
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }
    return max_abs;
}

} // namespace kernels
} // namespace mlib
