if (FASTAD_ENABLE_COVERAGE)
    message(STATUS "Building test suite with coverage information")
    add_compile_options(--coverage -O0 -fno-inline -fno-inline-small-functions -fno-default-inline)
endif()

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
    ifelse_unittest
    pow_unittest
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
    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
		target_compile_options(${test} PRIVATE -Werror -Wextra)
	endif()
    target_compile_options(${test} PRIVATE -g -Wall)
    target_include_directories(${test} PRIVATE ${GTEST_DIR}/include)
    if (FASTAD_ENABLE_COVERAGE)
        target_link_libraries(${test} gcov)
    endif()
    target_link_libraries(${test} gtest_main ${PROJECT_NAME})
    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
		target_link_libraries(${test} pthread)
	endif()
    add_test(${test} ${test})
endforeach()
