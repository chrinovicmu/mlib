// kernels/scalar/swap.cpp

namespace mlib {
namespace kernels {

template<typename T>
void swap(Vector<T>& x, Vector<T>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector dimensions must match in swap");
    }
    if (x.empty()) {
        return;
    }

    const size_t n = x.size();
    T* __restrict px = x.data();
    T* __restrict py = y.data();

    for (size_t i = 0; i < n; ++i) {
        T tmp   = px[i];
        px[i]   = py[i];
        py[i]   = tmp;
    }
}

} // namespace kernels
} // namespace mlib
