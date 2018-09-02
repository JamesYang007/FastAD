#include "ultimate_unittest.h"

namespace {
    // +,sin
    TEST(ultimate, sample1) {
        using namespace ad;
        double x1 = 2.0, x2 = 1.31, x3 = -3.14;
        double dfs[3] = {0};
        Var<double> leaf1(x1 ,dfs);
        Var<double> leaf2(x2 ,dfs+1);
        Var<double> leaf3(x3 ,dfs+2);
        
        auto res = leaf1 + sin(leaf2 + leaf3);
        EXPECT_EQ(res.feval(), x1 + std::sin(x2 + x3));
        res.beval(1); 
        EXPECT_EQ(dfs[0], 1);
        EXPECT_EQ(dfs[1], std::cos(x2 + x3));
        EXPECT_EQ(dfs[2], std::cos(x2 + x3));
    }

    // +,*,sin
    TEST(ultimate, sample2) {
        using namespace ad;
        double x1 = 1.2041, x2 = -2.2314;
        double dfs[2] = {0,0};
        Var<double> leaf1(x1 ,dfs);
        Var<double> leaf2(x2 ,dfs+1);

        EXPECT_EQ(leaf1.feval(), x1);
        EXPECT_EQ(leaf2.w, x2);

        auto res = leaf1 * leaf2 + sin(leaf1);
        EXPECT_EQ(res.feval(), x1*x2 + std::sin(x1));
        res.beval(1);
        EXPECT_EQ(dfs[0], x2 + std::cos(x1));
        EXPECT_EQ(dfs[1], x1);
    }

    // +,-,*,/,sin
    TEST(ultimate, sample3) {
        using namespace ad;
        double x1 = 1.2041, x2 = -2.2314;
        double dfs[2];
        Var<double> leaf1(x1 ,dfs);
        Var<double> leaf2(x2 ,dfs+1);

        auto res = leaf1 * leaf2 + sin(leaf1 + leaf2) * leaf2 - leaf1/leaf2;
        EXPECT_EQ(res.feval(), x1*x2 + std::sin(x1+x2)*x2 - x1/x2);
        res.beval(1);
        EXPECT_EQ(dfs[0], 
                    x2 + std::cos(x1 + x2) * x2 - 1./x2);
        EXPECT_EQ(dfs[1], 
                    x1 + std::cos(x1 + x2) * x2 + std::sin(x1+x2) + x1/(x2*x2));
    }

    // x*z + sin(cos(x+y))*y - x/exp(z)
    // +,-,*,/,sin,cos,exp
    TEST(ultimate, sample4) {
        using namespace ad;
        double x1 = 1.5928, x2 = -0.291, x3 = 5.1023;
        double dfs[3];
        Var<double> leaf1(x1 ,dfs);
        Var<double> leaf2(x2 ,dfs+1);
        Var<double> leaf3(x3 ,dfs+2);

        auto res = 
            leaf1 * leaf3 + sin(cos(leaf1 + leaf2)) * leaf2 - leaf1/exp(leaf3);
        EXPECT_EQ(res.feval(), 
                x1*x3 + std::sin(std::cos(x1+x2))*x2 - x1/std::exp(x3));
        res.beval(1);
        EXPECT_EQ(dfs[0], 
                    x3 - x2*std::cos(std::cos(x1+x2))*std::sin(x1+x2) - std::exp(-x3));
        EXPECT_EQ(dfs[1],
                    std::sin(std::cos(x1+x2)) 
                    - x2*std::cos(std::cos(x1+x2))*std::sin(x1+x2));
        EXPECT_EQ(dfs[2], x1 + x1*std::exp(-x3));
    }

    // Benchmarks over ntrials trials
    // 2-dim
    TEST(ultimate, benchmark1) {
        using namespace ad;
        size_t ntrials = 1e6;
        double x1 = 1.2041, x2 = -2.2314;
        double dfs[2] = {0};
        for (size_t i=0; i < ntrials; ++i) {
        Var<double> leaf1(x1 ,dfs);
        Var<double> leaf2(x2 ,dfs+1);

        auto res = leaf1 * leaf2 + sin(leaf1 + leaf2) * leaf2 - leaf1/leaf2;
        EXPECT_EQ(res.feval(), x1*x2 + std::sin(x1+x2)*x2 - x1/x2);
        res.beval(1);
        
        EXPECT_EQ(dfs[0], 
                    x2 + std::cos(x1 + x2) * x2 - 1./x2);
        EXPECT_EQ(dfs[1], 
                    x1 + std::cos(x1 + x2) * x2 + std::sin(x1+x2) + x1/(x2*x2));
        dfs[0] = 0; dfs[1]=0;
        }
    }

    // Benchmarks over ntrials trials
    // 3-dim
    TEST(ultimate, benchmark2) {
        using namespace ad;
        size_t ntrials = 1e6;
        for (size_t i=0; i < ntrials; ++i) {
        double x1 = 1.5928, x2 = -0.291, x3 = 5.1023;
        double dfs[3];
        Var<double> leaf1(x1 ,dfs);
        Var<double> leaf2(x2 ,dfs+1);
        Var<double> leaf3(x3 ,dfs+2);

        auto res = 
            leaf1 * leaf3 + sin(cos(leaf1 + leaf2)) * leaf2 - leaf1/exp(leaf3);
        EXPECT_EQ(res.feval(), 
                x1*x3 + std::sin(std::cos(x1+x2))*x2 - x1/std::exp(x3));
        res.beval(1);
        EXPECT_EQ(dfs[0], 
                    x3 - x2*std::cos(std::cos(x1+x2))*std::sin(x1+x2) - std::exp(-x3));
        EXPECT_EQ(dfs[1],
                    std::sin(std::cos(x1+x2)) 
                    - x2*std::cos(std::cos(x1+x2))*std::sin(x1+x2));
        EXPECT_EQ(dfs[2], x1 + x1*std::exp(-x3));
        dfs[0] = 0; dfs[1]=0; dfs[2]=0;
        }
    }

}
