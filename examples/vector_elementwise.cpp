
#include <mlib/vector.hpp>
#include <mlib/kernels/vectors/elementwise.hpp>

#include <iostream>
#include <iomanip>

int main()
{
    const std::size_t n = 10;
    mlib::Vector<float> a(n), b(n), result(n);

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i + 1);
        b[i] = static_cast<float>(n - i + 0.5f);
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Input vectors:\n";
    std::cout << "a = "; for (float v : a) std::cout << v << " "; std::cout << "\n";
    std::cout << "b = "; for (float v : b) std::cout << v << " "; std::cout << "\n\n";

    mlib::kernels::add_p32(a, b, result);
    std::cout << "add_p32(a, b) → result = ";
    for (float v : result) std::cout << v << " ";
    std::cout << "\n";

    mlib::kernels::sub_p32(a, b, result);
    std::cout << "sub_p32(a, b) → result = ";
    for (float v : result) std::cout << v << " ";
    std::cout << "\n";

    mlib::kernels::mul_p32(a, b, result);
    std::cout << "mul_p32(a, b) → result = ";
    for (float v : result) std::cout << v << " ";
    std::cout << "\n";

    mlib::kernels::div_p32(a, b, result);
    std::cout << "div_p32(a, b) → result = ";
    for (float v : result) std::cout << v << " ";
    std::cout << "\n";

    mlib::kernels::abs_p32(b, result);
    std::cout << "abs_p32(b)    → result = ";
    for (float v : result) std::cout << v << " ";
    std::cout << "\n";

    mlib::kernels::neg_p32(a, result);
    std::cout << "neg_p32(a)    → result = ";
    for (float v : result) std::cout << v << " ";
    std::cout << "\n";

    return 0;
}
