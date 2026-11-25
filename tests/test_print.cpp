// Check the public text format and ensure that allocation padding stays hidden.
#include "harness.hpp"
#include <mlib/mlib.hpp>

#include <iomanip>
#include <sstream>

TH_TEST(print_vectors_in_both_precisions) {
    mlib::Vector<float> values{1.0f, -2.5f, 3.0f};
    for (std::size_t i = values.size(); i < values.capacity(); ++i) {
        values.data()[i] = 999.0f;
    }
    const mlib::Vector<double> precise{-4.0, 0.25};

    std::ostringstream output;
    mlib::print(values, output);
    mlib::print(precise, output);
    TH_CHECK(output.str() == "[1, -2.5, 3]\n[-4, 0.25]\n");
}

TH_TEST(print_matrix_respects_padded_rows) {
    mlib::Matrix<float> values(2, 3);
    values(0, 0) = 1.0f;
    values(0, 1) = -2.0f;
    values(0, 2) = 3.5f;
    values(1, 0) = 4.0f;
    values(1, 1) = 5.0f;
    values(1, 2) = 6.0f;
    for (std::size_t row = 0; row < values.rows(); ++row) {
        for (std::size_t col = values.cols(); col < values.stride(); ++col) {
            values.row(row)[col] = 999.0f;
        }
    }

    const auto& matrix = values;
    std::ostringstream output;
    mlib::print(matrix, output);
    TH_CHECK(output.str() == "[\n  [1, -2, 3.5],\n  [4, 5, 6]\n]\n");
}

TH_TEST(print_empty_vectors_and_matrices) {
    const mlib::Vector<float> float_vector;
    const mlib::Vector<double> double_vector;
    const mlib::Matrix<float> float_matrix;
    const mlib::Matrix<double> double_matrix;

    std::ostringstream output;
    mlib::print(float_vector, output);
    mlib::print(double_vector, output);
    mlib::print(float_matrix, output);
    mlib::print(double_matrix, output);
    TH_CHECK(output.str() == "[]\n[]\n[]\n[]\n");
}

TH_TEST(print_single_row_and_column) {
    mlib::Matrix<double> row(1, 3);
    row(0, 0) = 1.0;
    row(0, 1) = 2.0;
    row(0, 2) = 3.0;
    mlib::Matrix<double> column(2, 1);
    column(0, 0) = 4.0;
    column(1, 0) = 5.0;

    std::ostringstream output;
    mlib::print(row, output);
    mlib::print(column, output);
    TH_CHECK(output.str() == "[\n  [1, 2, 3]\n]\n[\n  [4],\n  [5]\n]\n");
}

TH_TEST(print_uses_stream_formatting) {
    const mlib::Vector<float> vector{1.5f, -2.0f};
    mlib::Matrix<double> matrix(1, 2);
    matrix(0, 0) = 3.25;
    matrix(0, 1) = 4.0;

    std::ostringstream output;
    output << std::fixed << std::setprecision(2);
    const auto flags = output.flags();
    const auto precision = output.precision();
    mlib::print(vector, output);
    mlib::print(matrix, output);

    TH_CHECK(output.str() == "[1.50, -2.00]\n[\n  [3.25, 4.00]\n]\n");
    TH_CHECK(output.flags() == flags);
    TH_CHECK(output.precision() == precision);
}
