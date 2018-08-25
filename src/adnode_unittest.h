#pragma once
#include "adnode.h"
#include "admath.h"
#include "gtest/gtest.h"

namespace {

    TEST(adnode_test, adnode) {
        using namespace ad::core;
        bool b;
        ADNode<double, LeafNode<double>> node(0.0, 1.0);
        EXPECT_EQ(node.w==0.0, 1);
        EXPECT_EQ(node.df==1.0, 1);
        auto& x = node.self();
        b = std::is_same<
            decltype(x)
            , LeafNode<double>&>::value;
        EXPECT_EQ(b, 1);

        ADNode<double, LeafNode<double>> const node_const(1.0, -1.0);
        EXPECT_EQ(node_const.w==1.0, 1);
        EXPECT_EQ(node_const.df==-1.0, 1);
        auto& tmp = node_const.self();
        b = std::is_same<
            decltype(tmp)
            , LeafNode<double> const&>::value;
        EXPECT_EQ(b, 1);
    }

    TEST(adnode_test, leaf) {
        using namespace ad::core;
        LeafNode<double> leaf(2.1);
        EXPECT_EQ(leaf.w==2.1, 1);
        EXPECT_EQ(leaf.df==0.0, 1);
        EXPECT_EQ(leaf.feval()==2.1, 1);
    }

    TEST(adnode_test, unary) {
        using namespace ad::core;
        bool b;
        LeafNode<double> leaf(0.0);
        UnaryNode<double, ad::math::Sin<double>, LeafNode<double>> unary(leaf);
        // Constructor ok?
        EXPECT_EQ(unary.w==0.0, 1);
        EXPECT_EQ(unary.df==0.0, 1);

        // Derived ok?
        auto& x = unary.self();
        b = std::is_same<
            decltype(x)
            , UnaryNode<double
                        , ad::math::Sin<double>
                        , LeafNode<double>
                        >&>::value;
        EXPECT_EQ(b, 1);

        UnaryNode<double, ad::math::Sin<double>, LeafNode<double>> const unary_const(leaf);
        EXPECT_EQ(unary_const.w==0.0, 1);
        EXPECT_EQ(unary_const.df==0.0, 1);
        auto& tmp = unary_const.self();
        b = std::is_same<
            decltype(tmp)
            , UnaryNode<double
                        , ad::math::Sin<double>
                        , LeafNode<double>
                        >const&>::value;
        EXPECT_EQ(b, 1);

        // feval
        EXPECT_EQ(unary.feval()==0.0, 1);
        
        // beval
        // dsin/dx(0) = cos(0) = 1
        unary.df = 1; // seed
        unary.beval();
        EXPECT_EQ(leaf.df==1, 1);
        EXPECT_EQ(unary.lhs.df==1, 1);
    }


    // More complicated example
    // LeafNode -> UnaryNode -> UnaryNode
    TEST(adnode_test, unary_complex) {
        using namespace ad;
        core::LeafNode<double> leaf(3.14);
        auto sin_unary = sin(leaf); 
        auto sin_sin_unary = sin(sin_unary);
        double fx = sin_sin_unary.feval();
        EXPECT_EQ(fx, std::sin(std::sin(3.14)));
        sin_sin_unary.df = 1;
        sin_sin_unary.beval();
        EXPECT_EQ(leaf.df, std::cos(std::sin(3.14)) * std::cos(3.14));
        EXPECT_EQ(sin_unary.df, std::cos(std::sin(3.14)));
    }


    // BinaryNode
    TEST(adnode_test, binary) {
        using namespace ad;
        core::LeafNode<double> leaf1(1.0);
        core::LeafNode<double> leaf2(0.0);

        auto binary = leaf1 + leaf2;
        // feval
        EXPECT_EQ(binary.lhs.w, 1.0);
        EXPECT_EQ(binary.rhs.w, 0.0);
        EXPECT_EQ(binary.feval(), 1.0);

        // beval
        binary.df = 1;
        binary.beval();
        EXPECT_EQ(leaf1.df, 1.0);
        EXPECT_EQ(leaf2.df, 1.0);
    }

    // BinaryNode more complex
    TEST(adnode_test, binary_complex) {
        using namespace ad;
        core::LeafNode<double> leaf1(1.0);
        core::LeafNode<double> leaf2(0.0);
        core::LeafNode<double> leaf3(2.0);
        core::LeafNode<double> leaf4(-3.0);

        auto binary = leaf1 + leaf2 + leaf3 + leaf4;
        // feval
        EXPECT_EQ(binary.feval(), 0.0);

        // beval
        binary.df = 1;
        binary.beval();
        EXPECT_EQ(leaf1.df, 1.0);
        EXPECT_EQ(leaf2.df, 1.0);
        EXPECT_EQ(leaf3.df, 1.0);
        EXPECT_EQ(leaf4.df, 1.0);
    }

}
