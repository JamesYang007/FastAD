#pragma once
#include "utils.h"
#include "dualnum.h"
#include "binaryop.h"
#include "gtest/gtest.h"

namespace {
    
    TEST(dualnum_test, constructor) {
        using namespace core;
        DualNum<double> dual(2.1, 2.3);
        EXPECT_EQ(dual.x, 2.1);
        EXPECT_EQ(dual.xdot, 2.3);
        bool x = std::is_same<DualNum<double>::value_type, double>::value;
        EXPECT_EQ(x, 1);
    }

    TEST(dualnum_test, logic_eq) {
        using namespace core; 
        DualNum<double> n1(-1.2, 0.104);
        DualNum<double> n2(-1.2, 0.104);
        EXPECT_EQ(n1 == n2, 1); 
    }

    TEST(dualnum_test, eval) {
        using namespace core; 
        DualNum<double> n1(-1.2, 0.104);
        EXPECT_EQ(n1 == n1.eval(), 1); 
    }

    TEST(dualnum_test, op_eq) {
        using namespace core; 
        DualNum<double> n1(-1.2, 0.104);
        DualNum<double> n2(3.2, 1.42);
        EXPECT_EQ(n1 == n2, 0); 
        n2 = n1; // copy assignment
        EXPECT_EQ(n1 == n2, 1);

    }

    TEST(dualnum_test, add) {
        using namespace core; 
        DualNum<double> n1(-1.2, 0.104);
        DualNum<double> n2(3.2, 1.42);
        DualNum<double> n3(1., 0.);
        DualNum<double> n4;

        n4 = n1 + n2 + n3; 
        DualNum<double> test_n4(3.0, 1.524);
        EXPECT_EQ(n4 == test_n4, 1);
    }
} // end namespace
