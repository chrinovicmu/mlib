# mlib – Minimal Linear Algebra Library

**mlib** is a lightweight, modern C++ linear algebra library designed for performance, clarity, and portability. It provides efficient dense vector and matrix operations with architecture-specific backends.

## Features

- Dense vector container: `mlib::Vector<T>`
- Dense matrix container: `mlib::Matrix<T>` (row-major storage)
- Optimized Level-1 vector operations:
  - `axpy`, `scal`, `copy`, `swap`, `dot`
- Element-wise vector operations: `add`, `sub`, `mul`, `div`, `abs`, `neg`
- Norms: L1 (sum of absolute values), L2 (Euclidean norm)
- Basic matrix operations

### Supported Backends

- Scalar (portable fallback)
- AVX2 (x86-64 with FMA)
- NEON (ARM64 / AArch64, including Apple Silicon)

## Requirements

- C++17 compiler (GCC 9+, Clang 8+, MSVC 2019+)
- CMake 3.20+
- AVX2-capable CPU (for AVX2 backend)
- ARM64 CPU (for NEON backend)

## Build Instructions

From project root:

```bash
mkdir -p build && cd build
```

### Linux / x86-64 (AVX2)

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DMLIB_ENABLE_AVX2=ON \
         -DMLIB_ENABLE_NEON=OFF \
         -DMLIB_BUILD_EXAMPLES=ON
cmake --build . --parallel $(nproc)
```

### macOS / Apple Silicon (NEON)

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DMLIB_ENABLE_AVX2=OFF \
         -DMLIB_ENABLE_NEON=ON \
         -DMLIB_BUILD_EXAMPLES=ON
cmake --build . --parallel $(nproc)
```

### Run examples

```bash
./examples/basic_vector_op
./examples/vector_elementwise
```

## Usage

Include the single umbrella header for all kernels:

```cpp
#include <mlib/kernels.hpp>
```

### Vector example

```cpp
mlib::Vector<float> x(1000), y(1000);
// fill data...

mlib::kernels::axpy_p32(2.5f, x, y);
float res = mlib::kernels::dot_p32(x, y);
mlib::kernels::add_p32(a, b, result);
```

### Matrix example

```cpp
mlib::Matrix<float> A(200, 300), B(300, 150), C(200, 150);
// fill matrices...

mlib::kernels::gemm_p32(A, B, C, 1.0f, 0.0f);  // C ← A·B
```

## License

MIT License – free to use, modify, and distribute.

