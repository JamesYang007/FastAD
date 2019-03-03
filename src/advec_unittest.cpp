#include "advec_unittest.hpp"

namespace {

    // Both xi's and dfs
    TEST(advec_test, constructor) {
        using namespace ad;
        double dfs[2] = {0};
        double x1 = 1.0, x2 = 2.0;
        double xs[2] = {x1, x2};
        Vec<double> vec({x1, x2}, dfs);

        for (size_t i=0; i < 2; ++i) {
            EXPECT_EQ(vec[i].w, xs[i]);
            EXPECT_EQ(vec[i].df, 0);
            EXPECT_EQ(vec[i].w_ptr, &vec[i].w);
            EXPECT_EQ(vec[i].df_ptr, dfs+i);
        }
    }

    // Only xi's
    TEST(advec_test, constructor_2) {
        using namespace ad;
        double x1 = -2.0, x2=3.1;
        double xs[2] = {x1, x2};
        Vec<double> vec({x1, x2});

        for (size_t i=0; i < 2; ++i) {
            EXPECT_EQ(vec[i].w, xs[i]);
            EXPECT_EQ(vec[i].df, 0);
            EXPECT_EQ(vec[i].w_ptr, &vec[i].w);
            EXPECT_EQ(vec[i].df_ptr, &vec[i].df);
        }
    }

    // push_back, emplace_back, compatibility with other features
    TEST(advec_test, memory) {
        using namespace ad;
        double x1 = 1.5928, x2 = -0.291, x3 = 5.1023;
        double dfs[2] = {0};
        Vec<double> vec({x1}, dfs);
        Var<double> w3(x3);
        vec.emplace_back(x2, dfs+1);
        vec.push_back(w3);

        // push_back
        EXPECT_EQ(vec[2].w, x3);
        EXPECT_EQ(vec[2].df_ptr, &w3.df);

        // autodiff
        Var<double> z;
        autodiff(
                z = vec[0] * vec[2] + sin(cos(vec[0] + vec[1])) * vec[1] - vec[0]/exp(vec[2])
                );
        EXPECT_EQ(z.w, 
                x1*x3 + std::sin(std::cos(x1+x2))*x2 - x1/std::exp(x3));
        EXPECT_EQ(dfs[0], 
                    x3 - x2*std::cos(std::cos(x1+x2))*std::sin(x1+x2) - std::exp(-x3));
        EXPECT_EQ(dfs[1],
                    std::sin(std::cos(x1+x2)) 
                    - x2*std::cos(std::cos(x1+x2))*std::sin(x1+x2));
        EXPECT_EQ(w3.df, x1 + x1*std::exp(-x3));
    }
}
