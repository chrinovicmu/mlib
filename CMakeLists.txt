# =============================================================================
# mlib -- top-level build
#
# Structure of this file, in the order things must happen:
#   1. language / standard setup
#   2. options
#   3. ISA resolution (cmake/DetectTarget.cmake)
#   4. generate config.hpp from the resolved ISA
#   5. build the library from generic.cpp + the active ISA's sources
#   6. tests / benchmarks
#
# Steps 3 and 4 must precede 5, because the generated header is what
# ops.hpp's dispatch chain reads.
# =============================================================================

cmake_minimum_required(VERSION 3.20)

# VERSION here populates mlib_VERSION_MAJOR/MINOR/PATCH, which
# config.hpp.in substitutes into MLIB_VERSION_*. Bumping the version is
# therefore a one-line change here, not a two-place edit.
#
# Start with CXX only. ASM is enabled after selection for AVX-512 and RVV;
# the other backends need only the C++ compiler.
project(mlib VERSION 0.1.0 LANGUAGES CXX)

# -----------------------------------------------------------------------------
# Standard and global posture.
#
# Note what is NOT set globally: any ISA flag. Those are attached per
# source file -- see the set_source_files_properties call below, and the
# rationale in cmake/DetectTarget.cmake.
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)   # error out rather than silently using C++14
set(CMAKE_CXX_EXTENSIONS OFF)         # -std=c++17, not -std=gnu++17: no GNU extensions

# Emits build/compile_commands.json for clangd, and for the Phase 4 pass
# pipeline, which needs the exact flags each kernel was compiled with.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release. Without this, a bare `cmake -B build` produces an
# unoptimized build with no -O flag at all, which for a SIMD library means
# benchmark numbers that are off by an order of magnitude and kernels
# whose vectorization never happened.
#
# The CONFIGURATION_TYPES guard skips this for multi-config generators
# (Xcode, Visual Studio), where CMAKE_BUILD_TYPE is meaningless.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(MLIB_BUILD_TESTS "Build the test suite"   ON)
option(MLIB_BUILD_BENCH "Build benchmarks"       OFF)  # Phase 7
option(MLIB_ENABLE_LTO  "Link-time optimization" ON)

# Phase 4 flips this on. Until then, kernels compile with a plain
# `clang -c` rather than the source -> .bc -> opt -> .o pipeline. The
# option exists now so the intent is visible in `cmake -LH` output and so
# the guard below can give a real message instead of a mysterious no-op.
option(MLIB_USE_LLVM_PASSES
       "Build kernels through the custom LLVM pass pipeline" OFF)

# Resolves MLIB_ARCH and sets MLIB_ARCH_UPPER + MLIB_ACTIVE_ISA_FLAGS.
# Must run before configure_file below, which consumes MLIB_ARCH_UPPER.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/DetectTarget.cmake")

# Older copies expose mlib_isa_source() and only list a single .cpp file.
# Require the matching detector so assembly companions cannot be omitted.
if(NOT COMMAND mlib_isa_sources)
    message(FATAL_ERROR
        "mlib: incompatible CMake files. cmake/DetectTarget.cmake must define "
        "mlib_isa_sources(), which lists both C++ and assembly sources. "
        "Update the complete mlib source directory, including CMakeLists.txt, "
        "cmake/DetectTarget.cmake, and src/backends/, from the same version, "
        "then configure again. Older detectors define mlib_isa_source() instead.")
endif()

if(MLIB_ARCH STREQUAL "avx512" OR MLIB_ARCH STREQUAL "rvv")
    enable_language(ASM)
endif()

# -----------------------------------------------------------------------------
# Generated config.hpp
#
# Written into the BUILD tree, never the source tree. That is what lets
# two builds with different MLIB_ARCH values coexist against one
# checkout -- build-avx2/ and build-neon/ each get their own config.hpp
# with a different macro, and neither can contaminate the other or leave
# a stale file behind in git.
#
# @ONLY restricts substitution to @VAR@ and leaves ${VAR} untouched. The
# template contains no ${} today, but without @ONLY any future shell or
# CMake-looking text in a comment would be silently eaten.
# -----------------------------------------------------------------------------
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/mlib/config.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/mlib/config.hpp"
    @ONLY)

# -----------------------------------------------------------------------------
# Kernel source selection.
#
# generic.cpp is ALWAYS in the list, even when another ISA is active. Two
# reasons:
#   * it is the reference implementation, so having it linked makes a
#     future per-op fallback ("sve.cpp has gemm but not gemv") a pure
#     source change with no build-system work;
#   * it costs one small object file.
#
# The active ISA's sources are appended only when it differs, so a generic
# build does not list the same file twice.
# -----------------------------------------------------------------------------
# All paths come from mlib_isa_sources() in DetectTarget.cmake rather than
# being spelled out here, so src/backends/ can be reorganised without
# touching this file.
mlib_isa_sources(generic MLIB_GENERIC_SOURCE)
set(MLIB_KERNEL_SOURCES "${MLIB_GENERIC_SOURCE}")
if(NOT MLIB_ARCH STREQUAL "generic")
    list(APPEND MLIB_KERNEL_SOURCES ${MLIB_ACTIVE_ISA_SOURCES})
endif()

# Guard rather than silent no-op: someone who passes
# -DMLIB_USE_LLVM_PASSES=ON deserves to be told it does nothing yet,
# rather than to benchmark an unoptimized build and draw conclusions.
if(MLIB_USE_LLVM_PASSES)
    message(FATAL_ERROR
        "mlib: MLIB_USE_LLVM_PASSES is a Phase 4 feature and is not "
        "wired up yet.")
endif()

# -----------------------------------------------------------------------------
# The library.
#
# STATIC by default. This is a design decision, not a default inherited
# from CMake:
#
# mlib::axpy() makes a direct call to the selected namespace. LTO can
# inline C++ kernels and trait wrappers. The external .S primitives remain
# out-of-line native functions: LTO does not rewrite their instructions.
# This target is explicitly STATIC regardless of BUILD_SHARED_LIBS.
# -----------------------------------------------------------------------------
add_library(mlib STATIC ${MLIB_KERNEL_SOURCES})

# Namespaced alias so in-tree consumers write mlib::mlib exactly as
# a downstream find_package() user would. Keeps the two call sites
# identical and catches "works in-tree, breaks when installed" early.
add_library(mlib::mlib ALIAS mlib)

# Two include roots, both PUBLIC so they propagate to anything linking us:
#   * the source tree, for the hand-written headers
#   * the BINARY tree, for the generated config.hpp
# Both are wrapped in BUILD_INTERFACE so they apply while building this
# project but are NOT baked into an exported target, where the absolute
# paths would be meaningless on another machine. INSTALL_INTERFACE
# supplies the relocatable path for that case.
target_include_directories(mlib PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")

# PUBLIC, so consumers are also compiled as C++17. The headers use
# inline constexpr variables and aligned operator new, both C++17-only.
target_compile_features(mlib PUBLIC cxx_std_17)

# PRIVATE: these are our standards, not something to impose on consumers.
#   -Wconversion / -Wsign-conversion  catch the size_t/int mixing that
#       silently truncates on a large index -- the single most common
#       real bug class in numeric code.
#   -Wshadow                          catches an inner `i` hiding an
#       outer `i` in nested kernel loops.
#   -Wold-style-cast                  forces static_cast/reinterpret_cast,
#       so every conversion states which kind it is.
target_compile_options(mlib PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wsign-conversion;-Wshadow;-Wold-style-cast>")

# Attach the ISA flags to the selected backend's C++ and assembly sources.
#
# The generic guard matters: generic.cpp must never receive ISA flags, or
# the "portable fallback" would itself contain instructions the fallback
# exists to avoid. The MLIB_ACTIVE_ISA_FLAGS guard skips the call for
# ISAs whose flag list is legitimately empty (neon).
if(NOT MLIB_ARCH STREQUAL "generic" AND MLIB_ACTIVE_ISA_FLAGS)
    set_source_files_properties(
        ${MLIB_ACTIVE_ISA_SOURCES}
        PROPERTIES COMPILE_OPTIONS "${MLIB_ACTIVE_ISA_FLAGS}")
endif()

# GCC LTO can lower sizeless RVV values using the baseline link-time ISA
# instead of this source's -march=rv64gcv (observed with GCC 13). Keep the
# RVV C++ object native so its vector types retain the correct target.
# Other C++ sources can still participate in LTO.
if(MLIB_ARCH STREQUAL "rvv" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    foreach(_source IN LISTS MLIB_ACTIVE_ISA_SOURCES)
        if(_source MATCHES "\\.cpp$")
            set_property(SOURCE "${_source}" APPEND PROPERTY COMPILE_OPTIONS -fno-lto)
        endif()
    endforeach()
endif()

# LTO, and only in optimized configurations -- enabling it for Debug
# slows the build substantially and destroys the debugging experience
# for no benefit.
#
# check_ipo_supported() is queried rather than assumed because LTO
# depends on the linker and the compiler agreeing (a clang compiler with
# a mismatched system ld will fail at link, not at configure). Degrading
# to a non-LTO build with a STATUS message beats a hard failure.
if(MLIB_ENABLE_LTO AND CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_msg)
    if(_ipo_ok)
        set_property(TARGET mlib PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "mlib: LTO unavailable (${_ipo_msg})")
    endif()
endif()

# -----------------------------------------------------------------------------
# Subdirectories
#
# enable_testing() must be called in THIS file, not in tests/, for `ctest`
# to work from the top-level build directory.
# -----------------------------------------------------------------------------
if(MLIB_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
    add_subdirectory(test)
endif()

if(MLIB_BUILD_BENCH)
    add_subdirectory(bench)
endif()

# -----------------------------------------------------------------------------
# Configuration summary.
#
# Printed last so it is the final thing on screen after a configure. The
# active-ISA line in particular is what tells you whether the fallback in
# DetectTarget.cmake silently changed your request.
#
# The library-kind string is built with if/else rather than a generator
# expression, because message() does NOT evaluate generator expressions --
# it would print the literal "$<IF:$<BOOL:>,shared,static>".
# -----------------------------------------------------------------------------
set(_lib_kind "static (LTO=${MLIB_ENABLE_LTO})")

message(STATUS "")
message(STATUS "mlib ${PROJECT_VERSION}")
message(STATUS "  target      : ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  build type  : ${CMAKE_BUILD_TYPE}")
# "detected" is what the CPU probe found the silicon capable of;
# "active ISA" is what survived the compiler-flag and kernel-exists gates.
# Printing both makes a downgrade visible here rather than leaving it to be
# inferred from a disappointing benchmark.
message(STATUS "  detected    : ${MLIB_DETECTED_ISA}")
message(STATUS "  active ISA  : ${MLIB_ARCH}")
message(STATUS "  ISA flags   : ${MLIB_ACTIVE_ISA_FLAGS}")
message(STATUS "  library     : ${_lib_kind}")
message(STATUS "  LLVM passes : ${MLIB_USE_LLVM_PASSES}")
message(STATUS "  tests       : ${MLIB_BUILD_TESTS}")
message(STATUS "")
