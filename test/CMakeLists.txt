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
    ${CMAKE_CURRENT_SOURCE_DIR}/type_traits_unittest.cpp
    )
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(utility_unittest PRIVATE -Werror -Wextra)
endif()
target_compile_options(utility_unittest PRIVATE -g -Wall)
target_include_directories(utility_unittest PRIVATE ${GTEST_DIR}/include)
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
# Core TEST
########################################################################

add_executable(core_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/var_view_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dualnum_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/math_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unary_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eq_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/glue_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sum_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/forward_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eval_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/prod_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pow_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/if_else_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bind_unittest.cpp
    )
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(core_unittest PRIVATE -Werror -Wextra)
endif()
target_compile_options(core_unittest PRIVATE -g -Wall)
target_include_directories(core_unittest PRIVATE ${GTEST_DIR}/include)
if (FASTAD_ENABLE_COVERAGE)
    target_link_libraries(core_unittest gcov)
endif()
target_link_libraries(core_unittest fastad_gtest_main
    ${PROJECT_NAME} Eigen3::Eigen)
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_link_libraries(core_unittest pthread)
endif()
add_test(core_unittest core_unittest)

########################################################################
# Integration TEST
########################################################################

add_executable(integration_test
    ${CMAKE_CURRENT_SOURCE_DIR}/node_inttest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ad_inttest.cpp
    )
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(integration_test PRIVATE -Werror -Wextra)
endif()
target_compile_options(integration_test PRIVATE -g -Wall)
target_include_directories(integration_test PRIVATE ${GTEST_DIR}/include)
if (FASTAD_ENABLE_COVERAGE)
    target_link_libraries(integration_test gcov)
endif()
target_link_libraries(integration_test fastad_gtest_main
    ${PROJECT_NAME} Eigen3::Eigen)
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_link_libraries(integration_test pthread)
endif()
add_test(integration_test integration_test)

