
namespace mlib {
namespace kernels {

template<typename T>
void scal(T alpha, Vector<T>& x) {
    if (x.empty()) return;

    const size_t n = x.size();
    T* __restrict px = x.data();

    if (alpha == T(0)) {
        for (size_t i = 0; i < n; ++i) {
            px[i] = T(0);
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            px[i] *= alpha;
        }
    }
}

} // namespace kernels
} // namespace mlib
