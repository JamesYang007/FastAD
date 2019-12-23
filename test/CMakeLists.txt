# All tests
set(
    TESTS
    # Unit tests
    dualnum_unittest
    vec_unittest
    mat_unittest
    math_unittest
    node_unittest
    eval_unittest
    hessian_unittest
    prod_unittest
    exgen_unittest
    jacobian_unittest
    forward_unittest
    utility_unittest
    # Integration tests
    node_inttest
    ad_inttest
)

foreach( test ${TESTS} )
    add_executable(${test} ${CMAKE_CURRENT_SOURCE_DIR}/${test}.cpp)
    # Flags passed to the C++ compiler.
    target_compile_options(${test} PRIVATE -g -Wall -Werror -Wextra)
    target_include_directories(${test} PRIVATE ${GTEST_DIR}/include)
    target_link_libraries(${test} gtest_main pthread ${PROJECT_NAME})
    add_test(${test} ${test})
endforeach()
