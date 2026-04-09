# benchmarks/CMakeLists.txt

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(MLIB_ARCH_DEFINE "MLIB_ARCH_X86_64")
    set(MLIB_ARCH_FLAGS  -mavx2 -mfma)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(MLIB_ARCH_DEFINE "MLIB_ARCH_ARM")
    set(MLIB_ARCH_FLAGS  -march=armv8-a+simd)   # baseline NEON
    # For Apple Silicon SVE2 or later: -march=armv8.5-a+sve2
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# ── bench_axpy ───────────────────────────────────────────────────────────────
add_executable(bench_axpy kernels/bench_axpy.cpp)

target_include_directories(bench_axpy PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(bench_axpy PRIVATE
    -O3 -march=native -fno-tree-vectorize -std=c++17
    ${MLIB_ARCH_FLAGS}
)

target_compile_definitions(bench_axpy PRIVATE ${MLIB_ARCH_DEFINE})

target_link_libraries(bench_axpy PRIVATE pthread)
