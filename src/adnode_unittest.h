#pragma once
#include "adnode.h"
#include "admath.h"
#include "gtest/gtest.h"

namespace {

    TEST(adnode_test, memory) {
        using namespace ad;
        double x1 = 2.0, x2 = 1.31, x3 = -3.14;
        double dfs[3] = {0};
        auto leaf1 = make_leaf(x1, dfs);
        auto leaf2 = make_var(x2, dfs+1); // same thing as make_leaf
        auto leaf3 = make_leaf(x3, dfs+2);
        
        auto res = leaf1 + sin(leaf2 + leaf3);
        //std::cout << "Mark" << std::endl;
        EXPECT_EQ(res.feval(), x1 + std::sin(x2 + x3));
    }

    TEST(adnode_test, adnode) {
        using namespace ad::core;
        bool b;
        LeafNode<double> node(0.0, 0, 1.0);
        EXPECT_EQ(node.w==0.0, 1);
        EXPECT_EQ(node.df==1.0, 1);
        auto& x = node.self();
        b = std::is_same<
            decltype(x)
            , LeafNode<double>&>::value;
        EXPECT_EQ(b, 1);

        LeafNode<double> const node_const(1.0, 0, -1.0);
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
        LeafNode<double> leaf(2.1, 0, 0);
        EXPECT_EQ(leaf.w==2.1, 1);
        EXPECT_EQ(leaf.df==0.0, 1);
        EXPECT_EQ(leaf.feval()==2.1, 1);
    }

    TEST(adnode_test, unary) {
        using namespace ad::core;
        bool b;
        double df;
        LeafNode<double> leaf(0.0, &df, 0);
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
        unary.beval(1);
        EXPECT_EQ(df==1, 1);
        EXPECT_EQ(unary.lhs.df==1, 1);
    }


    // More complicated example
    // LeafNode -> UnaryNode -> UnaryNode
    TEST(adnode_test, unary_complex) {
        using namespace ad;
        double df;
        core::LeafNode<double> leaf(3.14, &df, 0.0);
        auto sin_unary = sin(leaf); 
        auto sin_sin_unary = sin(sin_unary);
        double fx = sin_sin_unary.feval();
        EXPECT_EQ(fx, std::sin(std::sin(3.14)));
        sin_sin_unary.beval(1);
        EXPECT_EQ(df, std::cos(std::sin(3.14)) * std::cos(3.14));
        // This is because sin_unary is copied
        EXPECT_EQ(sin_sin_unary.lhs.df, std::cos(std::sin(3.14)));
        //EXPECT_EQ(sin_unary.df, std::cos(std::sin(3.14))); 
        EXPECT_EQ(sin_unary.df, 0); 
    }


    // BinaryNode
    TEST(adnode_test, binary) {
        using namespace ad;
        double df1, df2;
        core::LeafNode<double> leaf1(1.0, &df1, 0.0);
        core::LeafNode<double> leaf2(0.0, &df2, 0.0);

        auto binary = leaf1 + leaf2;
        // feval
        EXPECT_EQ(binary.lhs.w, 1.0);
        EXPECT_EQ(binary.rhs.w, 0.0);
        EXPECT_EQ(binary.feval(), 1.0);

        // beval
        binary.beval(1);
        EXPECT_EQ(df1, 1.0);
        EXPECT_EQ(df2, 1.0);
    }

    // BinaryNode more complex
    TEST(adnode_test, binary_complex) {
        using namespace ad;
        double df1,df2,df3,df4;
        core::LeafNode<double> leaf1(1.0, &df1, 0.0);
        core::LeafNode<double> leaf2(0.0, &df2, 0.0);
        core::LeafNode<double> leaf3(2.0, &df3, 0.0);
        core::LeafNode<double> leaf4(-3.0, &df4, 0.0);

        auto&& binary = leaf1 + leaf2 + leaf3 + leaf4;
        // feval
        EXPECT_EQ(binary.feval(), 0.0);
        EXPECT_EQ(binary.lhs.w, 3.0);

        // beval
        binary.beval(1);
        EXPECT_EQ(df1, 1.0);
        EXPECT_EQ(df2, 1.0);
        EXPECT_EQ(df3, 1.0);
        EXPECT_EQ(df4, 1.0);
    }

    // Glueing style
    TEST(adnode_test, glue) {
        using namespace ad::core;
        LeafNode<double> w1(1.0);
        LeafNode<double> w2(2.0);
        LeafNode<double> w3(3.0);
        LeafNode<double> w4(4.0);

        auto subexpr = w1 * w2;
        EXPECT_EQ(subexpr.lhs.w, 1.0);
        EXPECT_EQ(subexpr.lhs.w_ptr, w1.w_ptr);
        EXPECT_EQ(subexpr.rhs.w, 2.0);
        EXPECT_EQ(subexpr.rhs.w_ptr, w2.w_ptr);

        auto subtree = (w3 = w1 * w2);
        EXPECT_EQ(subtree.lhs.w, 3.0);
        EXPECT_EQ(subtree.lhs.w_ptr, w3.w_ptr);
        EXPECT_EQ(subtree.rhs.lhs.w, 1.0);
        EXPECT_EQ(subtree.rhs.rhs.w, 2.0);
        EXPECT_EQ(subtree.rhs.lhs.w_ptr, w1.w_ptr);
        EXPECT_EQ(subtree.rhs.rhs.w_ptr, w2.w_ptr);

        auto subtree2 = (w4 = w3*w3);
        EXPECT_EQ(subtree2.lhs.w, 4.0);
        EXPECT_EQ(subtree2.lhs.w_ptr, w4.w_ptr);
        EXPECT_EQ(subtree2.rhs.lhs.w, 3.0);
        EXPECT_EQ(subtree2.rhs.rhs.w, 3.0);
        EXPECT_EQ(subtree2.rhs.lhs.w_ptr, w3.w_ptr);
        EXPECT_EQ(subtree2.rhs.rhs.w_ptr, w3.w_ptr);

        //auto expr = (subtree, subtree2);
        auto expr = (
                w3 = w1 * w2
                , w4 = w3 * w3
                , w4 = w4
                );
        EXPECT_EQ(expr.lhs.lhs.lhs.w, 3.0);

        ad::Evaluate(expr);
        EXPECT_EQ(w4.w, 4.0);
        EXPECT_EQ(w3.w, 2.0);
        EXPECT_EQ(w2.w, 2.0);
        EXPECT_EQ(w1.w, 1.0);

        EXPECT_NE(w4.df_ptr, nullptr);
        EXPECT_NE(w3.df_ptr, nullptr);
        EXPECT_NE(w2.df_ptr, nullptr);
        EXPECT_NE(w1.df_ptr, nullptr);

        ad::EvaluateAdj(expr);
        EXPECT_EQ(w4.df, 1.0);
        EXPECT_EQ(w3.df, 2*w3.w);
        EXPECT_EQ(w2.df, 2*w2.w*w1.w*w1.w);
        EXPECT_EQ(w1.df, 2*w1.w*w2.w*w2.w);
    }

    // Use-case
    TEST(adnode_test, use_case) {
        using namespace ad;
        double x1 = -0.201, x2 = 1.2241;
        Var<double> w1(x1);
        Var<double> w2(x2);

        Var<double> w3;
        Var<double> w4;
        Var<double> w5;
        auto expr = (
                w3 = w1 * sin(w2)
                , w4 = w3 + w1*w2
                , w5 = exp(w4*w3)
                );
        autodiff(expr);
        //Evaluate(expr);

        EXPECT_EQ(w5.w, std::exp((x1*std::sin(x2) + x1*x2)*(x1*std::sin(x2))));
        EXPECT_EQ(w4.w, x1*std::sin(x2) + x1*x2);
        EXPECT_EQ(w3.w, x1*std::sin(x2));

        //EvaluateAdj(expr);
        EXPECT_EQ(w5.df, 1);
        EXPECT_EQ(w4.df, w3.w * w5.w);
        EXPECT_EQ(w3.df, (w3.w+w4.w) * w5.w);
        EXPECT_EQ(w2.df, w5.w*x1*x1*(std::cos(x2)*(std::sin(x2)+x2) + std::sin(x2)*(1+std::cos(x2))));
        EXPECT_EQ(w1.df, w5.w * 2 * x1 * std::sin(x2) *(std::sin(x2) + x2));
    }
}
