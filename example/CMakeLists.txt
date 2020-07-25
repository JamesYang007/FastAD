# Build example
set(EXAMPLE "example")
add_executable(${EXAMPLE} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_include_directories(${EXAMPLE} PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(${EXAMPLE} ${PROJECT_NAME})
