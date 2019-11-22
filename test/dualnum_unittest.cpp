#include <fastad_bits/dualnum.hpp>
#include "gtest/gtest.h"

namespace {

TEST(dualnum, constructor) {
    using namespace ad::core;
    DualNum<double> dual(2.1, 2.3);
    EXPECT_EQ(dual.w, 2.1);
    EXPECT_EQ(dual.df, 2.3);
    bool x = std::is_same<DualNum<double>::value_type, double>::value;
    EXPECT_EQ(x, 1);
}

} // end namespace
