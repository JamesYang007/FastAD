#include "adeval_unittest.h"

namespace {

    // TEST for Function Object
    //
    // Possible user-defined functions
    auto&& F = FUNCTION(double, 
            ad::sin(x[0])*ad::cos(x[1]),
            x[2] + x[3]*x[4], 
            w[0] + w[1]
            );

    auto&& G = FUNCTION(double,
            ad::sum(x.begin(), x.end(), [](ad::Var<double> const& var)
                {return ad::sin(var);}),
            w[0]*w[0] - ad::sum(x.begin(), x.end(), [](ad::Var<double> const& var)
                {return ad::cos(var);})
            );

    auto&& H = FUNCTION(double, x[0]*x[4]);
    
    // GTest user-defined functions
    template <class T, class Iter>
    void f_test(arma::Mat<T> const& res, size_t i, Iter begin)
    {
        EXPECT_DOUBLE_EQ(res(i,0), std::cos(*begin)*std::cos(*std::next(begin)));
        EXPECT_DOUBLE_EQ(res(i,1), -std::sin(*begin)*std::sin(*std::next(begin)));
        EXPECT_DOUBLE_EQ(res(i,2), 1);
        for(size_t j = 0; j < 3; ++j) ++begin;
        EXPECT_DOUBLE_EQ(res(i,4), *begin);
        EXPECT_DOUBLE_EQ(res(i,3), *(++begin));
    }

    template <class T, class Iter>
    void g_test(arma::Mat<T> const& res, size_t i, Iter begin)
    {
        T sum = 0; auto it = begin;
        for (size_t j = 0; j < res.n_cols; ++it, ++j)
            sum += std::sin(*it);
        for (size_t j = 0; j < res.n_cols; ++begin, ++j)
            EXPECT_DOUBLE_EQ(res(i,j), 2*sum*std::cos(*begin) + std::sin(*begin));
    }

    template <class T, class Iter>
    void h_test(arma::Mat<T> const& res, size_t i, Iter begin)
    {
        EXPECT_DOUBLE_EQ(res(i,4), *begin);
        EXPECT_DOUBLE_EQ(res(i,1), 0);
        EXPECT_DOUBLE_EQ(res(i,2), 0);
        EXPECT_DOUBLE_EQ(res(i,3), 0);
        for (size_t j = 0; j < 4; ++j) ++begin;
        EXPECT_DOUBLE_EQ(res(i,0), *begin);
    }

//                                                           ================================================================================================
    
    // Scalar Function f:R^n -> R
    TEST(adeval_test, function_scalar) {
        using namespace ad;
        double x[] = {0.1, 2.3, -1., 4.1, -5.21};
        double y[] = {2.1, 5.3, -1.23, 0.0012, -5.13};
        arma::Mat<double> res(1,5);
        auto&& f = FUNCTION(double, 
                ad::sin(x[0])*ad::cos(x[1]),
                x[2] + x[3]*x[4], 
                w[0] + w[1]
                );
        auto&& F = make_function(f);

        auto test_core = [&F, &res](double* begin, double* end) mutable {
            auto&& expr = F(begin, end);
            autodiff(expr);
            for (size_t i = 0; i < 5; ++i) 
                res(0,i) = *(F.x[i].df_ptr);
            f_test(res, 0, begin);
        };

        // We try both to see if Function member variable Vector is correctly
        // cleared and re-reserve capacity

        // with x
        test_core(x, x+5);
        // with y
        test_core(y, y+5);
    }

    // Vector Function f:R^n -> R^m
    TEST(adeval_test, function_vector) {
        using namespace ad;
        double x[] = {0.1, 2.3, -1., 4.1, -5.21};
        double y[] = {2.1, 5.3, -1.23, 0.0012, -5.13};

        auto&& F_long = make_function(F, G, H);

        auto test_core = [&F_long](double* begin, double* end) mutable {
            auto&& expr = F_long(begin, end);
            autodiff(expr);
            arma::Mat<double> res = jacobian(F_long);
            f_test(res, 0, begin);
            g_test(res, 1, begin);
            h_test(res, 2, begin);
        };
        test_core(x, x+5);
        test_core(y, y+5);
    }

    // Complex Vector Function
    TEST(adeval_test, function_vector_complex) {
        using namespace ad;
        constexpr size_t n = 1e3;
        std::vector<double> x;
        std::default_random_engine gen;
        std::normal_distribution<double> dist(0.0,1.0);

        for (size_t i = 0; i < n; ++i) 
            x.push_back(dist(gen));

        auto&& F_long = make_function(
                F, G, H, F, G, H, F, G, H, F,
                F, G, H, F, G, H, F, G, H, F
                );

        using Iter = decltype(x.begin());
        auto test_core = [&F_long](Iter begin, Iter end) mutable {
            auto&& expr = F_long(begin, end);
            autodiff(expr);
            arma::Mat<double> res = jacobian(F_long);
            f_test(res, 0, begin);
            g_test(res, 1, begin);
            h_test(res, 2, begin);
        };
        test_core(x.begin(), x.end());
    }


} // end namespace
