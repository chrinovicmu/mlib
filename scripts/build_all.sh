# macOS (Apple Silicon → NEON)
mkdir -p build-mac && cd build-mac
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
ctest
./examples/example_basic

# Linux / x86-64 (AVX2)
mkdir -p build-linux && cd build-linux
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
ctest
./examples/example_basic
