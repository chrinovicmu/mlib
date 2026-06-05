cmake_minimum_required(VERSION 3.20)
project(ArithToSIMD LANGUAGES CXX)
 
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # -std=c++17 not -std=gnu++17
 
find_package(LLVM REQUIRED CONFIG)
 
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
 
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
 
include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
add_definitions(${LLVM_DEFINITIONS_LIST})
 
add_library(ArithToSIMD MODULE
    ArithToSIMDPass.cpp
)
 
target_compile_options(ArithToSIMD PRIVATE
    -fno-rtti
    -fno-exceptions
    -Wall
    -Wextra
    -Wno-unused-parameter
)
 
set_target_properties(ArithToSIMD PROPERTIES
    PREFIX ""                        # strip "lib" prefix
    OUTPUT_NAME "ArithToSIMD"
)
