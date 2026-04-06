#include <mlib/matrix.hpp>
#include <cmath>


/*Binary element-wise*/ 
template<typename T>
void mat_add(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    const size_t rows = A.rows();
    const size_t cols = A.cols();

    for (size_t i = 0; i < rows; ++i) {
        const T* __restrict a_row = A.row(i);
        const T* __restrict b_row = B.row(i);
        T* __restrict c_row = C.row(i);

        for (size_t j = 0; j < cols; ++j) {
            c_row[j] = a_row[j] + b_row[j];
        }
    }
}

void mat_sub(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    const size_t rows = A.rows();
    const size_t cols = A.cols();

    for (size_t i = 0; i < rows; ++i) {
        const T* __restrict a_row = A.row(i);
        const T* __restrict b_row = B.row(i);
        T* __restrict c_row = C.row(i);

        for (size_t j = 0; j < cols; ++j) {
            c_row[j] = a_row[j] - b_row[j];
        }
    }
}

template<typename T>
void mat_mul(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {   // Hadamard
    const size_t rows = A.rows();
    const size_t cols = A.cols();

    for (size_t i = 0; i < rows; ++i) {
        const T* __restrict a_row = A.row(i);
        const T* __restrict b_row = B.row(i);
        T* __restrict c_row = C.row(i);

        for (size_t j = 0; j < cols; ++j) {
            c_row[j] = a_row[j] * b_row[j];
        }
    }
}

template<typename T>
void mat_div(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    const size_t rows = A.rows();
    const size_t cols = A.cols();

    for (size_t i = 0; i < rows; ++i) {
        const T* __restrict a_row = A.row(i);
        const T* __restrict b_row = B.row(i);
        T* __restrict c_row = C.row(i);

        for (size_t j = 0; j < cols; ++j) {
            c_row[j] = a_row[j] / b_row[j];
        }
    }
}

template<typename T>
void mat_add_scalar(T alpha, const Matrix<T>& A, Matrix<T>& C) {
    const size_t rows = A.rows();
    const size_t cols = A.cols();

    for (size_t i = 0; i < rows; ++i) {
        const T* __restrict a_row = A.row(i);
        T* __restrict c_row = C.row(i);

        for (size_t j = 0; j < cols; ++j) {
            c_row[j] = a_row[j] + alpha;
        }
    }
}

template<typename T>
void mat_mul_scalar(T alpha, Matrix<T>& A) {   // in-place
    const size_t rows = A.rows();
    const size_t cols = A.cols();

    for (size_t i = 0; i < rows; ++i) {
        T* __restrict a_row = A.row(i);
        for (size_t j = 0; j < cols; ++j) {
            a_row[j] *= alpha;
        }
    }
}

template<typename T>
void mat_neg(const Matrix<T>& A, Matrix<T>& C) {
    const size_t rows = A.rows();
    const size_t cols = A.cols();

    for (size_t i = 0; i < rows; ++i) {
        const T* __restrict a_row = A.row(i);
        T* __restrict c_row = C.row(i);

        for (size_t j = 0; j < cols; ++j) {
            c_row[j] = -a_row[j];
        }
    }
}

template<typename T>
void mat_abs(const Matrix<T>& A, Matrix<T>& C) {
    const size_t rows = A.rows();
    const size_t cols = A.cols();

    for (size_t i = 0; i < rows; ++i) {
        const T* __restrict a_row = A.row(i);
        T* __restrict c_row = C.row(i);

        for (size_t j = 0; j < cols; ++j) {
            c_row[j] = std::abs(a_row[j]);
        }
    }
}
