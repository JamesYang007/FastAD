# Build example
set(EXAMPLE "example")
add_executable(${EXAMPLE} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(${EXAMPLE} ${PROJECT_NAME} Eigen3::Eigen)

add_executable(black_scholes ${CMAKE_CURRENT_SOURCE_DIR}/black_scholes.cpp)
target_link_libraries(black_scholes ${PROJECT_NAME} Eigen3::Eigen)
