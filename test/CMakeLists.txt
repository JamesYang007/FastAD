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
# if user specified path to armadillo, use it
# otherwise, find armadillo and if it exists, use it
# otherwise, raise error
if (ARMA_PATH)
    message(STATUS "Check for include path to armadillo: ${ARMA_PATH}")
    include_directories(${ARMA_PATH})
else()
find_path(ARMA_PATH NAMES armadillo)
if (ARMA_PATH)
    message(STATUS "Check for include path to armadillo: ${ARMA_PATH}")
    include_directories(${ARMA_PATH})
else()
    message(FATAL_ERROR "Check for include path to armadillo: NOT FOUND")
endif()
endif()

# Check if armadillo library exists
# if user specified path to armadillo, use it
# otherwise, find armadillo and if it exists, use it
# otherwise, raise error
if (LIBARMA_PATH)
    message(STATUS "Check for path to armadillo library: ${LIBARMA_PATH}")
    set(USE_ARMA ON)
else()
find_library(LIBARMA_PATH NAMES armadillo)
if (LIBARMA_PATH)
    message(STATUS "Check for path to armadillo library: ${LIBARMA_PATH}")
    set(USE_ARMA ON)
else()
    message(FATAL_ERROR "Check for path to armadillo library: NOT FOUND")
    set(USE_ARMA OFF)
endif()
endif()

# For every test, create executable, link with libraries,
foreach( test ${UNITTESTS} )
    add_executable(${test} ${CMAKE_CURRENT_SOURCE_DIR}/${test}.cpp)
    target_link_libraries(${test} gtest_main.a ${LIBARMA_PATH} lapack blas)
    add_test(${test} ${test})
endforeach()
