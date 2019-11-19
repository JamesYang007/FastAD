cmake_minimum_required(VERSION 3.10)

project("FastAD" VERSION 1.0)

# enables testing
enable_testing()

# Flags passed to the preprocessor.
# Set Google Test's header directory as a system directory, such that
# the compiler doesn't generate warnings in Google Test headers.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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

option(USE_ARMA "Use armadillo as matrix library" OFF)

# Flags passed to the C++ compiler.
add_compile_options(-g -Wall -Werror)

# add test subdirectory
add_subdirectory(${PROJECT_SOURCE_DIR}/test ${PROJECT_BINARY_DIR}/test)

