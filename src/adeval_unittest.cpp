#include "adeval_unittest.h"

namespace {

    auto f = [](ad::Vec<double> const& vec)
    {
        return ad::sin(vec[0])*ad::cos(vec[1]) + vec[2] + vec[3]*vec[4];
    };

    inline auto g(ad::Vec<double> const& vec)
    {
        return ad::sum(vec.begin(), vec.end(), [](ad::Var<double> const& var)
                {return ad::sin(var);});
    };
        
    // Univariate function
    TEST(adeval_test, autodiff_uni) {
        using namespace ad;
        double x[] = {-1, 2, 3.1, 4.2, 10.2};
        arma::Mat<double> res = autodiff(f, x, x + 5);

        EXPECT_DOUBLE_EQ(res(0,0), std::cos(x[0])*std::cos(x[1]));
        EXPECT_DOUBLE_EQ(res(0,1), -std::sin(x[0])*std::sin(x[1]));
        EXPECT_DOUBLE_EQ(res(0,2), 1);
        EXPECT_DOUBLE_EQ(res(0,3), x[4]);
        EXPECT_DOUBLE_EQ(res(0,4), x[3]);
    }

    // Multivariate function
    TEST(adeval_test, autodiff_multi) {
        using namespace ad;
        double x[] = {-1, 2, 3.1, 4.2, 10.2};
        auto F = std::make_tuple(f, g);
        arma::Mat<double> res = autodiff(F, x, x + 5);
        
        EXPECT_DOUBLE_EQ(res(0,0), std::cos(x[0])*std::cos(x[1]));
        EXPECT_DOUBLE_EQ(res(0,1), -std::sin(x[0])*std::sin(x[1]));
        EXPECT_DOUBLE_EQ(res(0,2), 1);
        EXPECT_DOUBLE_EQ(res(0,3), x[4]);
        EXPECT_DOUBLE_EQ(res(0,4), x[3]);

        for (int i = 0; i < 5; ++i)
            EXPECT_DOUBLE_EQ(res(1,i), std::cos(x[i]));
    }
    
    // Complex Multivariate function
    // Copy and paste the f,g,... line 
    // Compilation time increased, but run-time is same
    TEST(adeval_test, autodiff_multi_complex) {
        using namespace ad;
        double x[] = {-1, 2, 3.1, 4.2, 10.2};
        auto F = std::make_tuple(
                f, g, f, g, f, g, f, g, f, g, f, g, f, g, f, g,
                f, g, f, g, f, g, f, g, f, g, f, g, f, g, f, g
                );
        arma::Mat<double> res = autodiff(F, x, x + 5);
        
        EXPECT_DOUBLE_EQ(res(0,0), std::cos(x[0])*std::cos(x[1]));
        EXPECT_DOUBLE_EQ(res(0,1), -std::sin(x[0])*std::sin(x[1]));
        EXPECT_DOUBLE_EQ(res(0,2), 1);
        EXPECT_DOUBLE_EQ(res(0,3), x[4]);
        EXPECT_DOUBLE_EQ(res(0,4), x[3]);

        for (int i = 0; i < 5; ++i) {
            EXPECT_DOUBLE_EQ(res(1,i), std::cos(x[i]));
        }
    }

} // end namespace
