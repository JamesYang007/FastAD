#define _USE_MATH_DEFINES
#include <fastad_bits/math.hpp>
#include <fastad_bits/hessian.hpp>
#include "gtest/gtest.h"

#ifdef USE_ARMA

#include <armadillo>

#endif

namespace ad {

auto F_lmda = [](const auto& x, const auto&) {
    return ad::sin(x[0]) * ad::exp(x[0]) - x[0] + ad::tan(x[0]);
};

auto G_lmda = [](const auto& x, const auto&) {
    return ad::sin(x[0]) * ad::cos(x[1]);
};

auto H_lmda = [](const auto& x, const auto&) {
    return ad::sin(x[0]) + x[0] * x[0] + x[1] * x[1] + ad::cos(x[2] * x[3]);
};

struct hessian_fixture: ::testing::Test
{
protected:

#ifdef USE_ARMA

    // Test functions
    // WOLFRAM-ALPHA HARD-CODED NUMBERS
    // Test Hessian of a H_lmda
    template <class Mat>
    void h_hess_test(Mat const& mat)
    {
        EXPECT_NEAR(mat(0, 0), 1.15853, 1e-5);
        for (int i = 0; i < 2; ++i)
            for (int j = i + 1; j < 4; ++j)
                EXPECT_NEAR(mat(i, j), 0., 1e-5);
        EXPECT_NEAR(mat(1, 1), 2., 1e-5);
        EXPECT_NEAR(mat(2, 2), -13.5017, 1e-4);
        EXPECT_NEAR(mat(2, 3), -9.58967, 1e-5);
        EXPECT_NEAR(mat(3, 3), -7.59469, 1e-5);
    }

    // Test gradient of a H_lmda computed while computing hessian
    template <class Mat>
    void h_grad_test(Mat const& mat)
    {
        EXPECT_NEAR(mat(0, 0), 2.5403, 1e-4);
        EXPECT_NEAR(mat(0, 1), 4, 1e-1);
        EXPECT_NEAR(mat(0, 2), 2.14629, 1e-5);
        EXPECT_NEAR(mat(0, 3), 1.60972, 1e-5);
    }

#endif

};

#ifdef USE_ARMA

// Hessian (second derivative) of a univariate function
// Verify that algorithm approach is correct
TEST_F(hessian_fixture, one_dimensional) {
    using T = double;
    Vec<ForwardVar<T>> x = {{2.1}};
    x[0].get_value().set_adjoint(1);
    auto gen = make_exgen<ForwardVar<T>>(F_lmda);
    auto expr = std::get<0>(gen.generate(x));
    autodiff(expr);

    T value = x[0].get_value().get_value(); // get forward variable value
    double deriv = 
        (std::cos(value) + std::sin(value)) *
        std::exp(value) - 
        1 + 1 / (std::cos(value) * std::cos(value));
    double hessian = 
        2 * (std::cos(value)*std::exp(value) + 
        std::sin(value) / 
        (std::cos(value) * std::cos(value) * std::cos(value)));
    double deriv_test = expr.get_value().get_adjoint();
    double hessian_test = x[0].get_adjoint().get_adjoint();
    EXPECT_DOUBLE_EQ(deriv_test, deriv);
    EXPECT_DOUBLE_EQ(hessian_test, hessian);
}

// Hessian of a bivariate function
TEST_F(hessian_fixture, two_dimensional) {
    using T = double;
    T x[] = { M_PI / 3 , M_PI / 6 };
    arma::Mat<T> hess;
    hessian(hess, x, x + 2, G_lmda);
    EXPECT_DOUBLE_EQ(hess(0, 0), -0.75);
    EXPECT_DOUBLE_EQ(hess(1, 1), -0.75);
    EXPECT_DOUBLE_EQ(hess(0, 1), -0.25);
    EXPECT_DOUBLE_EQ(hess(1, 0), -0.25);
}

// Hessian multivariate
TEST_F(hessian_fixture, multi_dimensional) {
    using T = double;
    // DON'T CHANGE THESE NUMBERS
    T x[] = { 1.,2.,3.,4. };
    arma::Mat<T> hess;
    arma::Mat<T> grad;
    hessian(hess, grad, x, x + 4, H_lmda);
    h_hess_test(hess);
    h_grad_test(grad);
}

#endif

} // namespace ad
