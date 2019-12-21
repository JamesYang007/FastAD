# Build googletest
add_library(gtest_main STATIC ${GTEST_DIR}/src/gtest-all.cc ${GTEST_DIR}/src/gtest_main.cc)
target_compile_features(gtest_main PRIVATE cxx_std_11)
target_include_directories(gtest_main PRIVATE ${GTEST_DIR}/include ${GTEST_DIR})
