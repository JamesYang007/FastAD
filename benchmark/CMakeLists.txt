# All tests
set(
    BENCHMARKS
    normal_benchmark
    sum_benchmark
    prod_benchmark
    ad_benchmark
    constant_eager_benchmark
)

# Try to find Adept and if exists, find path, library
if (NOT DEFINED ADEPT_INCLUDE_DIR)
    find_path(ADEPT_INCLUDE_DIR adept)
endif()
if (NOT DEFINED ADEPT_LIB)
    find_library(ADEPT_LIB adept)
endif()

foreach( benchmark ${BENCHMARKS} )
    add_executable(${benchmark} ${CMAKE_CURRENT_SOURCE_DIR}/${benchmark}.cpp)
    target_link_libraries(${benchmark} benchmark benchmark_main
        ${PROJECT_NAME} Eigen3::Eigen)
    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
		target_link_libraries(${benchmark} pthread)
    endif()

    # if Adept exists, define USE_ADEPT, add path, and link
    if (ADEPT_INCLUDE_DIR AND ADEPT_LIB)
        target_compile_definitions(${benchmark} PRIVATE USE_ADEPT)
        target_include_directories(${benchmark} PRIVATE ${ADEPT_INCLUDE_DIR})
        target_link_libraries(${benchmark} ${ADEPT_LIB})
    endif()
endforeach()
