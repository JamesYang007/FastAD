# All tests
set(
    BENCHMARKS
    sum_benchmark
    foreach_benchmark
    prod_benchmark
)

# TODO: maybe this should change from add_test to just add_executable?
foreach( benchmark ${BENCHMARKS} )
    add_executable(${benchmark} ${CMAKE_CURRENT_SOURCE_DIR}/${benchmark}.cpp)
    target_include_directories(${benchmark} PRIVATE ${GTEST_DIR}/include)
    target_link_libraries(${benchmark} gtest_main pthread ${PROJECT_NAME})
endforeach()
