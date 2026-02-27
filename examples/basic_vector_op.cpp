#include <mlib/vector.hpp>
#include <mlib/kernels/vectors/axpy.hpp>
#include <mlib/kernels/vectors/scal.hpp>
#include <mlib/kernels/vectors/copy.hpp>
#include <mlib/kernels/vectors/swap.hpp>
#include <mlib/kernels/vectors/dot.hpp>

#include <iostream>
#include <iomanip>

int main()
{
    // Create small vectors for clear output
    const std::size_t n = 8;
    mlib::Vector<float> x(n), y(n), temp(n);

    // Initialize with simple values
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i + 1);
        y[i] = static_cast<float>((i + 1) * 10);
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Initial vectors:\n";
    std::cout << "x = "; for (float v : x) std::cout << v << " "; std::cout << "\n";
    std::cout << "y = "; for (float v : y) std::cout << v << " "; std::cout << "\n\n";

    // axpy: y ← 2.5 * x + y
    mlib::kernels::axpy_p32(2.5f, x, y);
    std::cout << "After axpy_p32(2.5, x, y):\n";
    std::cout << "y = "; for (float v : y) std::cout << v << " "; std::cout << "\n\n";

    // scal: x ← -1.0 * x
    mlib::kernels::scal_p32(-1.0f, x);
    std::cout << "After scal_p32(-1.0, x):\n";
    std::cout << "x = "; for (float v : x) std::cout << v << " "; std::cout << "\n\n";

    // copy: temp ← x
    mlib::kernels::copy_p32(x, temp);
    std::cout << "After copy_p32(x, temp):\n";
    std::cout << "temp = "; for (float v : temp) std::cout << v << " "; std::cout << "\n\n";

    // swap: swap x and temp
    mlib::kernels::swap_p32(x, temp);
    std::cout << "After swap_p32(x, temp):\n";
    std::cout << "x    = "; for (float v : x)    std::cout << v << " "; std::cout << "\n";
    std::cout << "temp = "; for (float v : temp) std::cout << v << " "; std::cout << "\n\n";

    // dot product
    float result = mlib::kernels::dot_p32(x, y);
    std::cout << "dot_p32(x, y) = " << result << "\n";

    return 0;
}
