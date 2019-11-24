# All tests
set(
    TESTS
    # Unit tests
    dualnum_unittest
    vec_unittest
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

include_directories(${PROJECT_SOURCE_DIR}/include)

# Builds tests
# For every unit test, create executable, link with libraries,
foreach( test ${TESTS} )
    add_executable(${test} ${CMAKE_CURRENT_SOURCE_DIR}/${test}.cpp)
    target_link_libraries(${test} gtest_main pthread)

    if (LIBARMA_PATH)
        target_link_libraries(${test} ${LIBARMA_PATH} lapack blas)
    endif()

    if (LIBBOOST_SYSTEM_MT_PATH AND LIBBOOST_THREAD_MT_PATH)
        target_link_libraries(${test} ${LIBBOOST_SYSTEM_MT_PATH} ${LIBBOOST_THREAD_MT_PATH})
    endif()

    add_test(${test} ${test})
endforeach()
