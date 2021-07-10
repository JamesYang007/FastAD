cmake_minimum_required(VERSION 3.10)
set(CMAKE_C_COMPILER /usr/local/bin/gcc CACHE PATH "" FORCE)
set(CMAKE_CXX_COMPILER /usr/local/bin/g++ CACHE PATH "" FORCE)

if (NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE "Release")
   #set(CMAKE_BUILD_TYPE "Debug")
endif()

project(main)

include_directories("/usr/local/include")
include_directories("include")
include_directories("../../include")

#eigen3
find_package(Eigen3)
include_directories(EIGEN3_INCLUDE_DIR)

add_executable(mp src.cpp)

#ceres
find_package(Ceres REQUIRED)
target_include_directories(mp PUBLIC ${CERES_INCLUDE_DIRS})
target_link_libraries(mp  Ceres::ceres gflags)

target_compile_features(mp PUBLIC cxx_std_20)
#target_compile_options(mp PUBLIC -Wall -Wextra -Werror)
#target_compile_options(mp PUBLIC -fsanitize=address)
add_custom_command(TARGET mp POST_BUILD
	COMMAND echo "----------------------------------------------"
	#COMMAND ./mp
	COMMAND echo "----------------------------------------------")
