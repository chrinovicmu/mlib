
namespace mlib {
namespace kernels {

template<typename T>
T dot(const T* __restrict x, const T* __restrict y, size_t count) {
    T result = T(0);
    for (size_t i = 0; i < count; ++i) {
        result += x[i] * y[i];
    }
    return result;
}

} 
} 
