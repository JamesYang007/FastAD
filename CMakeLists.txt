cmake_minimum_required(VERSION 3.9)

project("FastAD" VERSION 2.0)

# enables testing
enable_testing()

# Flags passed to the preprocessor.
# Set Google Test's header directory as a system directory, such that
# the compiler doesn't generate warnings in Google Test headers.
set(CMAKE_CXX_STANDARD 17)
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

# Not needed at the moment
# TODO: in the future, we want fastad to compile certain features based on this flag
# option(USE_ARMA "Use armadillo as matrix library" OFF)

# Function to check if PATH_VAR exists as an include path if
# TYPE_VAR is 'include' otherwise as library path if TYPE_VAR is 'library.'
# If user specified PATH_VAR, use it
# Otherwise, find path with NAME_VAR and use as PATH_VAR
function (find_path_generic PATH_VAR NAME_VAR TYPE_VAR)
    if (NOT ${TYPE_VAR} STREQUAL "include" AND NOT ${TYPE_VAR} STREQUAL "library")
        message(FATAL_ERROR "TYPE_VAR must be \"include\" or \"library\"")
    endif()

    if (${PATH_VAR})
        message(STATUS "Check for ${TYPE_VAR} path to ${NAME_VAR}: ${${PATH_VAR}}")
        if (${TYPE_VAR} STREQUAL "include")
            include_directories(${${PATH_VAR}})
        endif()
    else()
        if (${TYPE_VAR} STREQUAL "include")
            find_path(${PATH_VAR} NAMES ${NAME_VAR})
        elseif (${TYPE_VAR} STREQUAL "library")
            find_library(${PATH_VAR} NAMES ${NAME_VAR})
        endif()
        if (${PATH_VAR})
            message(STATUS "Check for ${TYPE_VAR} path to ${NAME_VAR}: ${${PATH_VAR}}")
            if (${TYPE_VAR} STREQUAL "include")
                include_directories(${${PATH_VAR}})
            endif()
        else()
            message(STATUS "Check for ${TYPE_VAR} path to ${NAME_VAR}: NOT FOUND")
        endif()
    endif()
endfunction(find_path_generic)

# Flags passed to the C++ compiler.
add_compile_options(-g -Wall -Werror -pthread -isystem ${GTEST_DIR}/include)

# Builds gtest_main.a.
# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
add_library(gtest_main STATIC ${GTEST_DIR}/src/gtest-all.cc ${GTEST_DIR}/src/gtest_main.cc)
target_include_directories(gtest_main PRIVATE ${GTEST_DIR})

##################################################
# Find dependency headers and libraries

# Check if armadillo include path exists
find_path_generic(ARMA_PATH "armadillo" "include")

# Check if boost include path exists
find_path_generic(BOOST_PATH "boost" "include")

# Check if armadillo library exists
find_path_generic(LIBARMA_PATH "armadillo" "library")

# If arma include path and library exists, can use arma
if (ARMA_PATH AND LIBARMA_PATH)
    add_definitions(-DUSE_ARMA)
endif()

# TODO: if no ARMA_PATH or no LIBARMA_PATH, implement and use my 2D array; set compiler def

# Check if boost library exists
find_path_generic(LIBBOOST_SYSTEM_MT_PATH "boost_system-mt" "library")
find_path_generic(LIBBOOST_THREAD_MT_PATH "boost_thread-mt" "library")

# if boost include, and both mt libs exist
# add compiler flag USE_BOOST_THREAD_MT
# if boost include not exist, only use subset of boost in deps/ and no mt libs.
if (BOOST_PATH AND LIBBOOST_SYSTEM_MT_PATH AND LIBBOOST_THREAD_MT_PATH)
    add_definitions(-DUSE_BOOST_THREAD_MT)
elseif (NOT BOOST_PATH)
    set(BOOST_PATH ${PROJECT_SOURCE_DIR}/deps/boost)
    include_directories(${BOOST_PATH})
endif()

# add test subdirectory
add_subdirectory(${PROJECT_SOURCE_DIR}/test ${PROJECT_BINARY_DIR}/test)

# add example subdirectory
add_subdirectory(${PROJECT_SOURCE_DIR}/example ${PROJECT_BINARY_DIR}/example)

# add benchmark subdirectory
add_subdirectory(${PROJECT_SOURCE_DIR}/benchmark ${PROJECT_BINARY_DIR}/benchmark)
