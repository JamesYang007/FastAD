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
            message(FATAL_ERROR "Check for ${TYPE_VAR} path to ${NAME_VAR}: NOT FOUND")
        endif()
    endif()
endfunction(find_path_generic)

# Flags passed to the C++ compiler.
add_compile_options(-g -Wall -Werror)

# add test subdirectory
add_subdirectory(${PROJECT_SOURCE_DIR}/test ${PROJECT_BINARY_DIR}/test)

# add example subdirectory
add_subdirectory(${PROJECT_SOURCE_DIR}/example ${PROJECT_BINARY_DIR}/example)
