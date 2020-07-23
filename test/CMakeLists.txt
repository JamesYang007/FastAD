if (FASTAD_ENABLE_COVERAGE)
    message(STATUS "Building test suite with coverage information")
    add_compile_options(--coverage -O0 -fno-inline -fno-inline-small-functions -fno-default-inline)
endif()

# All tests
set(
    TESTS
    # Unit tests
    #vec_unittest
    #mat_unittest
    #node_unittest
    #eval_unittest
    #hessian_unittest
    #ifelse_unittest
    #pow_unittest
    #prod_unittest
    #exgen_unittest
    #jacobian_unittest
    # Integration tests
    #node_inttest
    #ad_inttest
)

########################################################################
# Utility TEST
########################################################################

add_executable(utility_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/type_traits_unittest.cpp
    )
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(utility_unittest PRIVATE -Werror -Wextra)
endif()
target_compile_options(utility_unittest PRIVATE -g -Wall)
target_include_directories(utility_unittest PRIVATE
    ${GTEST_DIR}/include ${EIGEN3_INCLUDE_DIR})
if (FASTAD_ENABLE_COVERAGE)
    target_link_libraries(utility_unittest gcov)
endif()
target_link_libraries(utility_unittest fastad_gtest_main ${PROJECT_NAME})
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_link_libraries(utility_unittest pthread)
endif()
add_test(utility_unittest utility_unittest)

########################################################################
# Core TEST
########################################################################

add_executable(core_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/type_traits_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/var_view_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dualnum_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/math_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unary_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eq_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/glue_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sum_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/forward_unittest.cpp
    )
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(core_unittest PRIVATE -Werror -Wextra)
endif()
target_compile_options(core_unittest PRIVATE -g -Wall)
target_include_directories(core_unittest PRIVATE
    ${GTEST_DIR}/include ${EIGEN3_INCLUDE_DIR})
if (FASTAD_ENABLE_COVERAGE)
    target_link_libraries(core_unittest gcov)
endif()
target_link_libraries(core_unittest fastad_gtest_main ${PROJECT_NAME})
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_link_libraries(core_unittest pthread)
endif()
add_test(core_unittest core_unittest)


