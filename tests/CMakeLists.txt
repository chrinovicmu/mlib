
add_executable(test_vectors test_vectors.cpp)
target_link_libraries(test_vectors PRIVATE mlib mlib_kernels)

add_test(NAME VectorKernels COMMAND test_vectors)
