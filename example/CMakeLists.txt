# Build example
set(EXAMPLE "example")
add_executable(${EXAMPLE} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(${EXAMPLE} ${PROJECT_NAME})
