
namespace mlib {
namespace kernels {

template<typename T>
void copy(const Vector<T>& x, Vector<T>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector dimensions must match in copy");
    }
    if (x.empty()) {
        return;
    }

    const size_t n = x.size();
    const T* __restrict px = x.data();
    T* __restrict py = y.data();

    for (size_t i = 0; i < n; ++i) {
        py[i] = px[i];
    }
}

} // namespace kernels
} // namespace mlib
