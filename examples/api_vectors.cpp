// Run: ./build/examples/api_vectors
#include <mlib/mlib.hpp>

#include <cmath>
#include <initializer_list>
#include <iostream>

// Print the vector and check it against values calculated by hand.
// Use checks instead of assert(), so failures still count in Release builds.
bool check_vector(const char* operation, const mlib::Vector<float>& actual,
                  std::initializer_list<float> expected) {
    std::cout << operation << ": ";
    mlib::print(actual);

    if (actual.size() != expected.size()) {
        std::cerr << operation << ": unexpected vector size\n";
        return false;
    }
    std::size_t i = 0;
    for (float wanted : expected) {
        if (!(std::abs(actual[i] - wanted) <= 1e-5f)) {
            std::cerr << operation << ": element " << i
                      << " should be " << wanted << '\n';
            return false;
        }
        ++i;
    }
    return true;
}

bool check_value(const char* operation, double actual, double expected) {
    std::cout << operation << ": " << actual << '\n';
    if (!(std::abs(actual - expected) <= 1e-10)) {
        std::cerr << operation << ": expected " << expected << '\n';
        return false;
    }
    return true;
}

int main() {
    std::cout << "Backend: " << MLIB_TARGET_ISA_NAME << '\n';

    // Braces supply actual values. Parentheses below specify a vector size.
    mlib::Vector<float> x{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    mlib::Vector<float> y{5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    std::cout << "x: ";
    mlib::print(x);
    std::cout << "y: ";
    mlib::print(y);

    // y = 2*x + y. x stays available for the next operations.
    mlib::axpy(2.0f, x, y);
    if (!check_vector("axpy", y, {7, 8, 9, 10, 11})) return 1;

    // Scale y in place.
    mlib::scal(0.5f, y);
    if (!check_vector("scal", y, {3.5f, 4, 4.5f, 5, 5.5f})) return 1;

    // Allocate the destination before copying. Both vectors need the same size.
    mlib::Vector<float> copied(x.size());
    mlib::copy(x, copied);
    if (!check_vector("copy", copied, {1, 2, 3, 4, 5})) return 1;

    // Exchange the elements of copied and y.
    mlib::swap(copied, y);
    if (!check_vector("swap: copied", copied, {3.5f, 4, 4.5f, 5, 5.5f})) return 1;
    if (!check_vector("swap: y", y, {1, 2, 3, 4, 5})) return 1;

    // x and y now both contain {1, 2, 3, 4, 5}: dot = 1+4+9+16+25.
    float dot_product = mlib::dot(x, y);
    if (!check_value("dot", dot_product, 55.0)) return 1;

    // The same public API also accepts double-precision vectors.
    mlib::Vector<double> direction{-3.0, 4.0};
    double length = mlib::nrm2(direction);       // sqrt(9 + 16) = 5.
    double absolute_sum = mlib::asum(direction); // |-3| + |4| = 7.
    if (!check_value("nrm2", length, 5.0)) return 1;
    if (!check_value("asum", absolute_sum, 7.0)) return 1;

    std::cout << "PASS: vector API\n";
    return 0;
}
