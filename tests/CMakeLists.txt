
function(mlib_add_test name)
    add_executable(${name} ${name}.cpp main.cpp)
    target_link_libraries(${name} PRIVATE mlib)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(${name} PRIVATE -Wall -Wextra)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

mlib_add_test(test_tensor)
mlib_add_test(test_print)
mlib_add_test(test_level1)
mlib_add_test(test_level23)
mlib_add_test(test_kernel_contracts)
