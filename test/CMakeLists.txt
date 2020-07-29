if (FASTAD_ENABLE_COVERAGE)
    message(STATUS "Building test suite with coverage information")
    add_compile_options(--coverage -O0 -fno-inline -fno-inline-small-functions -fno-default-inline)
endif()

# All tests
# TODO:
# - hessian_unittest

########################################################################
# Utility TEST
########################################################################

add_executable(utility_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/util/type_traits_unittest.cpp
    )

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(utility_unittest PRIVATE -Werror -Wextra)
endif()
target_compile_options(utility_unittest PRIVATE -g -Wall)
target_include_directories(utility_unittest PRIVATE
    ${GTEST_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
if (FASTAD_ENABLE_COVERAGE)
    target_link_libraries(utility_unittest gcov)
endif()
target_link_libraries(utility_unittest fastad_gtest_main
    ${PROJECT_NAME} Eigen3::Eigen)
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_link_libraries(utility_unittest pthread)
endif()
add_test(utility_unittest utility_unittest)

########################################################################
# Forward Core TEST
########################################################################

add_executable(forward_core_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/forward/core/dualnum_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/forward/core/forward_unittest.cpp
    )

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(forward_core_unittest PRIVATE -Werror -Wextra)
endif()
target_compile_options(forward_core_unittest PRIVATE -g -Wall)
target_include_directories(forward_core_unittest PRIVATE
    ${GTEST_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
if (FASTAD_ENABLE_COVERAGE)
    target_link_libraries(forward_core_unittest gcov)
endif()
target_link_libraries(forward_core_unittest fastad_gtest_main
    ${PROJECT_NAME} Eigen3::Eigen)
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_link_libraries(forward_core_unittest pthread)
endif()
add_test(forward_core_unittest forward_core_unittest)

########################################################################
# Reverse Core TEST
########################################################################

add_executable(reverse_core_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/reverse/core/var_view_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reverse/core/math_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reverse/core/unary_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reverse/core/binary_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reverse/core/eq_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reverse/core/glue_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reverse/core/sum_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reverse/core/eval_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reverse/core/prod_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reverse/core/pow_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reverse/core/if_else_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reverse/core/bind_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reverse/core/dot_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reverse/core/norm_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reverse/core/sqrt_unittest.cpp
    )

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(reverse_core_unittest PRIVATE -Werror -Wextra)
endif()
target_compile_options(reverse_core_unittest PRIVATE -g -Wall)
target_include_directories(reverse_core_unittest PRIVATE
    ${GTEST_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
if (FASTAD_ENABLE_COVERAGE)
    target_link_libraries(reverse_core_unittest gcov)
endif()
target_link_libraries(reverse_core_unittest fastad_gtest_main
    ${PROJECT_NAME} Eigen3::Eigen)
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_link_libraries(reverse_core_unittest pthread)
endif()
add_test(reverse_core_unittest reverse_core_unittest)

########################################################################
# Reverse Stat TEST
########################################################################

add_executable(reverse_stat_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/reverse/stat/normal_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reverse/stat/uniform_unittest.cpp
    )

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(reverse_stat_unittest PRIVATE -Werror -Wextra)
endif()
target_compile_options(reverse_stat_unittest PRIVATE -g -Wall)
target_include_directories(reverse_stat_unittest PRIVATE
    ${GTEST_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
if (FASTAD_ENABLE_COVERAGE)
    target_link_libraries(reverse_stat_unittest gcov)
endif()
target_link_libraries(reverse_stat_unittest fastad_gtest_main
    ${PROJECT_NAME} Eigen3::Eigen)
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_link_libraries(reverse_stat_unittest pthread)
endif()
add_test(reverse_stat_unittest reverse_stat_unittest)

########################################################################
# Integration TEST
########################################################################

add_executable(integration_test
    ${CMAKE_CURRENT_SOURCE_DIR}/integration/node_inttest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/integration/ad_inttest.cpp
    )

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(integration_test PRIVATE -Werror -Wextra)
endif()
target_compile_options(integration_test PRIVATE -g -Wall)
target_include_directories(integration_test PRIVATE
    ${GTEST_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
if (FASTAD_ENABLE_COVERAGE)
    target_link_libraries(integration_test gcov)
endif()
target_link_libraries(integration_test fastad_gtest_main
    ${PROJECT_NAME} Eigen3::Eigen)
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_link_libraries(integration_test pthread)
endif()
add_test(integration_test integration_test)

