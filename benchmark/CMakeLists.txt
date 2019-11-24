# All tests
set(
    BENCHMARKS
    sum_benchmark
    foreach_benchmark
    prod_benchmark
)

include_directories(${PROJECT_SOURCE_DIR}/include)

# TODO: maybe this should change from add_test to just add_executable?
foreach( benchmark ${BENCHMARKS} )
    add_executable(${benchmark} ${CMAKE_CURRENT_SOURCE_DIR}/${benchmark}.cpp)

    target_link_libraries(${benchmark} gtest_main pthread)

    if (LIBARMA_PATH)
        target_link_libraries(${benchmark} ${LIBARMA_PATH} lapack blas)
    endif()

    if (LIBBOOST_SYSTEM_MT_PATH AND LIBBOOST_THREAD_MT_PATH)
        target_link_libraries(${benchmark} ${LIBBOOST_SYSTEM_MT_PATH} ${LIBBOOST_THREAD_MT_PATH})
    endif()

endforeach()
