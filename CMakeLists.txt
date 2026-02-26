cmake_minimum_required(VERSION 3.20)
project(mlib LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")


message(STATUS "System name: ${CMAKE_SYSTEM_NAME}")
message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")

set(SUPPORTED_ARCH "unknown")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)")
    set(SUPPORTED_ARCH "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64|ARM64)")
    set(SUPPORTED_ARCH "arm64")
else()
    message(WARNING "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

option(MLIB_ENABLE_AVX2      "Enable AVX2 backend (x86-64 only)"          ON)
option(MLIB_ENABLE_NEON      "Enable NEON backend (arm64 only)"           ON)
option(MLIB_BUILD_EXAMPLES   "Build example programs"                     ON)
option(MLIB_BUILD_TESTS      "Build unit tests"                           ON)

if(NOT SUPPORTED_ARCH STREQUAL "x86_64")
    set(MLIB_ENABLE_AVX2 OFF CACHE BOOL "AVX2 not supported on this arch" FORCE)
endif()

if(NOT SUPPORTED_ARCH STREQUAL "arm64")
    set(MLIB_ENABLE_NEON OFF CACHE BOOL "NEON not supported on this arch" FORCE)
endif()

add_compile_options(
    -Wall -Wextra -Wpedantic
    -Wconversion -Wsign-conversion
    -Wshadow -Wnon-virtual-dtor
)

if(CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
    add_compile_options(-O3 -DNDEBUG)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")          # macOS
    add_compile_options(-fno-common)
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_options(-fPIC)
endif()

add_library(mlib INTERFACE)
target_include_directories(mlib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(mlib INTERFACE cxx_std_17)

add_library(mlib_scalar_kernels OBJECT)
target_sources(mlib_scalar_kernels PRIVATE
    kernels/backends/scalar/vectors/axpy.cpp
    kernels/backends/scalar/vectors/copy.cpp
    kernels/backends/scalar/vectors/dot.cpp
    kernels/backends/scalar/vectors/elementwise.cpp
    kernels/backends/scalar/vectors/scal.cpp
    kernels/backends/scalar/vectors/swap.cpp
)
target_link_libraries(mlib_scalar_kernels PRIVATE mlib)

if(MLIB_ENABLE_AVX2)
    add_library(mlib_avx2_kernels OBJECT)
    target_sources(mlib_avx2_kernels PRIVATE
        kernels/backends/avx2/vectors/axpy.cpp
        kernels/backends/avx2/vectors/copy.cpp
        kernels/backends/avx2/vectors/dot.cpp
        kernels/backends/avx2/vectors/elementwise.cpp
        kernels/backends/avx2/vectors/scal.cpp
        kernels/backends/avx2/vectors/swap.cpp
    )
    target_link_libraries(mlib_avx2_kernels PRIVATE mlib)
    target_compile_options(mlib_avx2_kernels PRIVATE -mavx2 -mfma)
endif()

if(MLIB_ENABLE_NEON)
    add_library(mlib_neon_kernels OBJECT)
    target_sources(mlib_neon_kernels PRIVATE
        kernels/backends/neon/vectors/axpy.cpp
        kernels/backends/neon/vectors/copy.cpp
        kernels/backends/neon/vectors/dot.cpp
        kernels/backends/neon/vectors/elementwise.cpp
        kernels/backends/neon/vectors/scal.cpp
        kernels/backends/neon/vectors/swap.cpp
    )
    target_link_libraries(mlib_neon_kernels PRIVATE mlib)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" AND APPLE)
        target_compile_options(mlib_neon_kernels PRIVATE -march=armv8-a)
    endif()
endif()

add_library(mlib_kernels INTERFACE)
target_link_libraries(mlib_kernels INTERFACE mlib)

target_link_libraries(mlib_kernels INTERFACE mlib_scalar_kernels)

if(MLIB_ENABLE_AVX2)
    target_link_libraries(mlib_kernels INTERFACE mlib_avx2_kernels)
    target_compile_definitions(mlib_kernels INTERFACE MLIB_HAS_AVX2)
endif()

if(MLIB_ENABLE_NEON)
    target_link_libraries(mlib_kernels INTERFACE mlib_neon_kernels)
    target_compile_definitions(mlib_kernels INTERFACE MLIB_HAS_NEON)
endif()

if(MLIB_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(MLIB_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()


message(STATUS "")
message(STATUS "mlib configuration summary:")
message(STATUS "  System:          ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Build type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  AVX2 backend:    ${MLIB_ENABLE_AVX2}")
message(STATUS "  NEON backend:    ${MLIB_ENABLE_NEON}")
message(STATUS "  Examples:        ${MLIB_BUILD_EXAMPLES}")
message(STATUS "  Tests:           ${MLIB_BUILD_TESTS}")
message(STATUS "")
