# All tests
set(
    BENCHMARKS
    sum_benchmark
    prod_benchmark
    ad_benchmark
)

# Add path to static google benchmark static library
link_directories(${GBENCH_DIR}/build/src)

# Try to find Adept and if exists, find path, library
find_path(ADEPT_INCLUDE_DIR adept)
find_library(ADEPT_LIB adept)

foreach( benchmark ${BENCHMARKS} )
    add_executable(${benchmark} ${CMAKE_CURRENT_SOURCE_DIR}/${benchmark}.cpp)
    target_include_directories(${benchmark} PRIVATE ${GBENCH_DIR}/include)
    target_link_libraries(${benchmark} benchmark benchmark_main pthread ${PROJECT_NAME})

    # if Adept exists, define USE_ADEPT, add path, and link
    if (ADEPT_INCLUDE_DIR AND ADEPT_LIB)
        target_compile_definitions(${benchmark} PRIVATE USE_ADEPT)
        target_include_directories(${benchmark} PRIVATE ${ADEPT_INCLUDE_DIR})
        target_link_libraries(${benchmark} ${ADEPT_LIB})
    endif()
endforeach()
