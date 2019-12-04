cmake_minimum_required(VERSION 3.9)

project("FastAD" VERSION 2.1)

# enables testing
enable_testing()

# Flags passed to the preprocessor.
# Set Google Test's header directory as a system directory, such that
# the compiler doesn't generate warnings in Google Test headers.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Points to the root of Google Test
set(GTEST_DIR ${PROJECT_SOURCE_DIR}/googletest/googletest)

# if CMAKE_BUILD_TYPE not defined
if (NOT CMAKE_BUILD_TYPE)

# if binary directory ends with "release", use release mode
if (${PROJECT_BINARY_DIR} MATCHES "release$")
    set(CMAKE_BUILD_TYPE RELEASE)
# otherwise, use debug mode
else()
    set(CMAKE_BUILD_TYPE DEBUG)
endif()

endif()

message(STATUS "Compiling in ${CMAKE_BUILD_TYPE} mode")

# Flags passed to the C++ compiler.
add_compile_options(-g -Wall -Werror -Wextra -pthread -isystem ${GTEST_DIR}/include)

# Builds gtest_main.a.
# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
add_library(gtest_main STATIC ${GTEST_DIR}/src/gtest-all.cc ${GTEST_DIR}/src/gtest_main.cc)
target_include_directories(gtest_main PRIVATE ${GTEST_DIR})

# add test subdirectory
add_subdirectory(${PROJECT_SOURCE_DIR}/test ${PROJECT_BINARY_DIR}/test)

# add example subdirectory
add_subdirectory(${PROJECT_SOURCE_DIR}/example ${PROJECT_BINARY_DIR}/example)

# add benchmark subdirectory
add_subdirectory(${PROJECT_SOURCE_DIR}/benchmark ${PROJECT_BINARY_DIR}/benchmark)
