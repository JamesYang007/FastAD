# Builds gtest_main.a.
# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
add_library(gtest_main STATIC ${GTEST_DIR}/src/gtest-all.cc ${GTEST_DIR}/src/gtest_main.cc)
target_compile_features(gtest_main PRIVATE cxx_std_11)
target_include_directories(gtest_main PRIVATE ${GTEST_DIR}/include ${GTEST_DIR})
