# Build google benchmark
if (FASTAD_ENABLE_BENCHMARK)
    message(STATUS "Building test suite with benchmarks.")
    # Taken from https://github.com/google/benchmark
    # Recommended way to build google benchmark
    # Disable all tests within google benchmark.
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable GoogleBenchmark tests.")
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable GoogleBenchmark googletest tests.")
    set(BENCHMARK_ENABLE_ASSEMBLY_TESTS OFF CACHE BOOL "Disable GoogleBenchmark assembly tests.")
    add_subdirectory(benchmark)
endif()

# Build googletest
if (FASTAD_ENABLE_TEST)
    add_library(fastad_gtest_main STATIC ${GTEST_DIR}/src/gtest-all.cc ${GTEST_DIR}/src/gtest_main.cc)
    target_compile_features(fastad_gtest_main PRIVATE cxx_std_11)
    target_include_directories(fastad_gtest_main PRIVATE ${GTEST_DIR}/include ${GTEST_DIR})
endif()
