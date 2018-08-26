#include "dualnum.h"
#include "binaryop.h"
#include "gtest/gtest.h"

namespace {

    TEST(binaryop_test, contructor) {
        using namespace ad::core::test;
        DualNum<double> n1(-2.3, 1.0);
        DualNum<double> n2(5.1002, -100);
        BinaryOpExpr<add, DualNum<double>, DualNum<double>> boe(n1, n2);

        bool x;
        x = std::is_same<decltype(n1) const&, decltype(boe.lhs)>::value;
        EXPECT_EQ(x, 1);
        x = std::is_same<decltype(n2) const&, decltype(boe.rhs)>::value;
        EXPECT_EQ(x, 1);
    }

    TEST(binaryop_test, eval) {
        using namespace ad::core::test;
        DualNum<double> n1(-2.3, 1.0);
        DualNum<double> n2(5.1002, -100);
        BinaryOpExpr<add, DualNum<double>, DualNum<double>> boe(n1, n2);
        DualNum<double> tmp = boe.eval();
        EXPECT_EQ(tmp == DualNum<double>(5.1002-2.3, 1.0-100), 1);
    }

} // end namespace
