# Points to the root of Google Test
set(GTEST_DIR ${PROJECT_SOURCE_DIR}/googletest/googletest)

# extra compile options for tests
add_compile_options(-pthread -isystem ${GTEST_DIR}/include)

# All tests
set(
        UNITTESTS
        dualnum_unittest
        advec_unittest
        admath_unittest
        adnode_unittest
        adeval_unittest
        adcompose_unittest
        adfunction_unittest
        hessian_unittest
        forwardvar_unittest
        ultimate_unittest
)

# Builds gtest_main.a.
# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
add_library(gtest_main.a STATIC ${GTEST_DIR}/src/gtest-all.cc ${GTEST_DIR}/src/gtest_main.cc)
target_include_directories(gtest_main.a PRIVATE ${GTEST_DIR})

# Builds tests
include_directories(${PROJECT_SOURCE_DIR}/include)

# Check if armadillo include path exists
find_path_generic(ARMA_PATH "armadillo" "include")

# Check if boost include path exists
find_path_generic(BOOST_PATH "boost" "include")

# Check if armadillo library exists
find_path_generic(LIBARMA_PATH "armadillo" "library")

# Check if boost library exists
find_path_generic(LIBBOOST_SYSTEM_MT_PATH "boost_system-mt" "library")
find_path_generic(LIBBOOST_THREAD_MT_PATH "boost_thread-mt" "library")

# For every test, create executable, link with libraries,
foreach( test ${UNITTESTS} )
    add_executable(${test} ${CMAKE_CURRENT_SOURCE_DIR}/${test}.cpp)
    target_link_libraries(${test} gtest_main.a
        ${LIBARMA_PATH}
        ${LIBBOOST_SYSTEM_MT_PATH}
        ${LIBBOOST_THREAD_MT_PATH}
        lapack
        blas
        )
    add_test(${test} ${test})
endforeach()
