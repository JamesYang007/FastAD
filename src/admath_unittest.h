#pragma once
#include "admath.h"
#include "gtest/gtest.h"

namespace {

    // Sin class
    TEST(admath_test, Sin) {
        using namespace ad::math;
        double y = Sin<double>::fmap(0);
        EXPECT_EQ(y==0, 1);
        y = Sin<double>::bmap(0);
        EXPECT_EQ(y==1, 1);
    }

    // sin function
    TEST(admath_test, sin) {
        using namespace ad;
        bool b;
        core::LeafNode<double> leaf;

        // return correct type?
        auto node = sin(leaf); 
        b = std::is_same<
            decltype(node)
            , typename core::UnaryNode<
                double
                , typename math::Sin<double>
                , core::LeafNode<double>
                >
            >::value;
        EXPECT_EQ(b, 1);
    }

    // operator+
    TEST(admath_test, op_add) {
        using namespace ad;
        core::LeafNode<double> leaf1(1.0);
        core::LeafNode<double> leaf2(0.0);

        bool b;
        auto binary = leaf1 + leaf2;
        b = std::is_same<
            decltype(binary)
            , core::BinaryNode<
                double
                , math::Add<double>
                , core::LeafNode<double>
                , core::LeafNode<double>
                >
            >::value;
        EXPECT_EQ(b, 1);
        
    }

    // Add
    TEST(admath_test, add) {
        using namespace ad::math; 
        EXPECT_EQ(Add<double>::fmap(-1.0, 2.1), 1.1);
        EXPECT_EQ(Add<double>::blmap(-2.01, 2341.2131), 1);
        EXPECT_EQ(Add<double>::brmap(-2.01, 2341.2131), 1);
    }
}
