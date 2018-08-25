#pragma once
#include "admath.h"
#include <gtest/gtest.h>
#include <ctime>
#include <iostream>

namespace {
    // +,sin
    TEST(ultimate, sample1) {
        using namespace ad;
        double x1 = 2.0, x2 = 1.31, x3 = -3.14;
        auto leaf1 = make_leaf(x1);
        auto leaf2 = make_var(x2); // same thing as make_leaf
        auto leaf3 = make_leaf(x3);
        
        auto res = leaf1 + sin(leaf2 + leaf3);
        EXPECT_EQ(res.feval(), x1 + std::sin(x2 + x3));
        res.df = 1;
        res.beval(); 
        EXPECT_EQ(leaf1.df, 1);
        EXPECT_EQ(leaf2.df, std::cos(x2 + x3));
        EXPECT_EQ(leaf3.df, std::cos(x2 + x3));
    }

    // +,*,sin
    TEST(ultimate, sample2) {
        using namespace ad;
        double x1 = 1.2041, x2 = -2.2314;
        auto leaf1 = make_leaf(x1);
        auto leaf2 = make_leaf(x2);

        auto res = leaf1 * leaf2 + sin(leaf1);
        EXPECT_EQ(res.feval(), x1*x2 + std::sin(x1));
        res.df=1;
        res.beval();
        EXPECT_EQ(leaf1.df, x2 + std::cos(x1));
        EXPECT_EQ(leaf2.df, x1);
    }

    // +,-,*,/,sin
    TEST(ultimate, sample3) {
        using namespace ad;
        double x1 = 1.2041, x2 = -2.2314;
        auto leaf1 = make_leaf(x1);
        auto leaf2 = make_leaf(x2);

        auto res = leaf1 * leaf2 + sin(leaf1 + leaf2) * leaf2 - leaf1/leaf2;
        EXPECT_EQ(res.feval(), x1*x2 + std::sin(x1+x2)*x2 - x1/x2);
        res.df=1;
        res.beval();
        EXPECT_EQ(leaf1.df, 
                    x2 + std::cos(x1 + x2) * x2 - 1./x2);
        EXPECT_EQ(leaf2.df, 
                    x1 + std::cos(x1 + x2) * x2 + std::sin(x1+x2) + x1/(x2*x2));
    }

    // x*z + sin(cos(x+y))*y - x/exp(z)
    // +,-,*,/,sin,cos,exp
    TEST(ultimate, sample4) {
        using namespace ad;
        double x1 = 1.5928, x2 = -0.291, x3 = 5.1023;
        auto leaf1 = make_leaf(x1);
        auto leaf2 = make_leaf(x2);
        auto leaf3 = make_leaf(x3);

        auto res = 
            leaf1 * leaf3 + sin(cos(leaf1 + leaf2)) * leaf2 - leaf1/exp(leaf3);
        EXPECT_EQ(res.feval(), 
                x1*x3 + std::sin(std::cos(x1+x2))*x2 - x1/std::exp(x3));
        res.df=1;
        res.beval();
        EXPECT_EQ(leaf1.df, 
                    x3 - x2*std::cos(std::cos(x1+x2))*std::sin(x1+x2) - std::exp(-x3));
        EXPECT_EQ(leaf2.df, 
                    std::sin(std::cos(x1+x2)) 
                    - x2*std::cos(std::cos(x1+x2))*std::sin(x1+x2));
        EXPECT_EQ(leaf3.df, x1 + x1*std::exp(-x3));
    }

    // Benchmarks over ntrials trials
    // 2-dim
    TEST(ultimate, benchmark1) {
        using namespace ad;
        size_t ntrials = 1e6;
        for (size_t i=0; i < ntrials; ++i) {
        double x1 = 1.2041, x2 = -2.2314;
        auto leaf1 = make_leaf(x1);
        auto leaf2 = make_leaf(x2);

        auto res = leaf1 * leaf2 + sin(leaf1 + leaf2) * leaf2 - leaf1/leaf2;
        EXPECT_EQ(res.feval(), x1*x2 + std::sin(x1+x2)*x2 - x1/x2);
        res.df=1;
        res.beval();
        EXPECT_EQ(leaf1.df, 
                    x2 + std::cos(x1 + x2) * x2 - 1./x2);
        EXPECT_EQ(leaf2.df, 
                    x1 + std::cos(x1 + x2) * x2 + std::sin(x1+x2) + x1/(x2*x2));
        }
    }

    // Benchmarks over ntrials trials
    // 3-dim
    TEST(ultimate, benchmark2) {
        using namespace ad;
        size_t ntrials = 1e6;
        for (size_t i=0; i < ntrials; ++i) {
            double x1 = -192.104, x2 = 23.2491, x3 = 0.2311023;
            auto leaf1 = make_leaf(x1);
            auto leaf2 = make_leaf(x2);
            auto leaf3 = make_leaf(x3);

            auto res = 
                leaf1 * leaf3 + sin(cos(leaf1 + leaf2)) * leaf2 - leaf1/exp(leaf3);
            EXPECT_EQ(res.feval(), 
                    x1*x3 + std::sin(std::cos(x1+x2))*x2 - x1/std::exp(x3));
            res.df=1;
            res.beval();
            EXPECT_EQ(leaf1.df, 
                        x3 - x2*std::cos(std::cos(x1+x2))*std::sin(x1+x2) - std::exp(-x3));
            EXPECT_EQ(leaf2.df, 
                        std::sin(std::cos(x1+x2)) 
                        - x2*std::cos(std::cos(x1+x2))*std::sin(x1+x2));
            EXPECT_EQ(leaf3.df, x1 + x1*std::exp(-x3));
        }
    }

}
