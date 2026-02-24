
namespace mlib {
namespace kernels {

template<typename T>
void swap(T* __restrict x, T* __restrict y, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        T tmp = x[i];
        x[i]  = y[i];
        y[i]  = tmp;
    }
}

} // namespace kernels
} // namespace mlib
