# All tests
set(
    BENCHMARKS
    sum_benchmark
    prod_benchmark
    ad_benchmark
)

# Add path to static google benchmark static library
link_directories(${GBENCH_DIR}/build/src)

foreach( benchmark ${BENCHMARKS} )
    add_executable(${benchmark} ${CMAKE_CURRENT_SOURCE_DIR}/${benchmark}.cpp)
    target_include_directories(${benchmark}
        PRIVATE
        ${GBENCH_DIR}/include
        ${ADEPT_INCLUDE_DIR})
    target_link_libraries(${benchmark} benchmark benchmark_main
        pthread ${PROJECT_NAME} ${ADEPT_LIB})
endforeach()
