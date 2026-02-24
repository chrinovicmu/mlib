
#include <mlib/vector.hpp>
#include <cmath>          
#include <stdexcept>

namespace mlib {
namespace kernels {

template<typename T>
void add(const Vector<T>& x, const Vector<T>& y, Vector<T>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Vector sizes must match in add");
    }
    if (x.empty()) return;

    const size_t n = x.size();
    const T* __restrict px = x.data();
    const T* __restrict py = y.data();
    T* __restrict pz = z.data();

    for (size_t i = 0; i < n; ++i) {
        pz[i] = px[i] + py[i];
    }
}

template<typename T>
void sub(const Vector<T>& x, const Vector<T>& y, Vector<T>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Vector sizes must match in sub");
    }
    if (x.empty()) return;

    const size_t n = x.size();
    const T* __restrict px = x.data();
    const T* __restrict py = y.data();
    T* __restrict pz = z.data();

    for (size_t i = 0; i < n; ++i) {
        pz[i] = px[i] - py[i];
    }
}

template<typename T>
void mul(const Vector<T>& x, const Vector<T>& y, Vector<T>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Vector sizes must match in mul");
    }
    if (x.empty()) return;

    const size_t n = x.size();
    const T* __restrict px = x.data();
    const T* __restrict py = y.data();
    T* __restrict pz = z.data();

    for (size_t i = 0; i < n; ++i) {
        pz[i] = px[i] * py[i];
    }
}

template<typename T>
void div(const Vector<T>& x, const Vector<T>& y, Vector<T>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Vector sizes must match in div");
    }
    if (x.empty()) return;

    const size_t n = x.size();
    const T* __restrict px = x.data();
    const T* __restrict py = y.data();
    T* __restrict pz = z.data();

    for (size_t i = 0; i < n; ++i) {
        pz[i] = px[i] / py[i];   // note: no zero-division check – user responsibility
    }
}

template<typename T>
void abs(const Vector<T>& x, Vector<T>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector sizes must match in abs");
    }
    if (x.empty()) return;

    const size_t n = x.size();
    const T* __restrict px = x.data();
    T* __restrict py = y.data();

    for (size_t i = 0; i < n; ++i) {
        py[i] = std::abs(px[i]);
    }
}

template<typename T>
void neg(const Vector<T>& x, Vector<T>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector sizes must match in neg");
    }
    if (x.empty()) return;

    const size_t n = x.size();
    const T* __restrict px = x.data();
    T* __restrict py = y.data();

    for (size_t i = 0; i < n; ++i) {
        py[i] = -px[i];
    }
}

template void add<float>(const Vector<float>&, const Vector<float>&, Vector<float>&);
template void sub<float>(const Vector<float>&, const Vector<float>&, Vector<float>&);
template void mul<float>(const Vector<float>&, const Vector<float>&, Vector<float>&);
template void div<float>(const Vector<float>&, const Vector<float>&, Vector<float>&);
template void abs<float>(const Vector<float>&, Vector<float>&);
template void neg<float>(const Vector<float>&, Vector<float>&);

template void add<double>(const Vector<double>&, const Vector<double>&, Vector<double>&);
template void sub<double>(const Vector<double>&, const Vector<double>&, Vector<double>&);
template void mul<double>(const Vector<double>&, const Vector<double>&, Vector<double>&);
template void div<double>(const Vector<double>&, const Vector<double>&, Vector<double>&);
template void abs<double>(const Vector<double>&, Vector<double>&);
template void neg<double>(const Vector<double>&, Vector<double>&);
} // namespace kernels
} // namespace mlib
