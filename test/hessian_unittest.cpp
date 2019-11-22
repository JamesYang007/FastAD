#define _USE_MATH_DEFINES

#include <fastad_bits/adeval.hpp>
#include <fastad_bits/admath.hpp>
#include <fastad_bits/adhessian.hpp>
#include "gtest/gtest.h"
#include <armadillo>

namespace {

auto&& F_lmda = MAKE_LMDA(
    ad::sin(x[0]) * ad::exp(x[0]) - x[0] + ad::tan(x[0])
);
auto&& G_lmda = MAKE_LMDA(
    ad::sin(x[0]) * ad::cos(x[1])
);
auto&& H_lmda = MAKE_LMDA(
    ad::sin(x[0]) + x[0] * x[0] + x[1] * x[1] + ad::cos(x[2] * x[3])
);

// =========================================================================================
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


// =========================================================================================


// Hessian (second derivative) of a univariate function
TEST(hessian, one_dimensional) {
    using namespace ad;
    using T = double;
    ForwardVar<T> x(2.1, 1);

    auto&& F = make_function<ForwardVar<T>>(F_lmda);

    auto&& expr = F(&x, &x + 1);
    autodiff(expr);
    double deriv = (std::cos(x.w) + std::sin(x.w))*std::exp(x.w) - 1 + 1 / (std::cos(x.w) * std::cos(x.w));
    double hessian = 2 * (std::cos(x.w)*std::exp(x.w) + std::sin(x.w) / (std::cos(x.w) * std::cos(x.w) * std::cos(x.w)));
    double deriv_test = F.x[0].df.w;
    double hessian_test = F.x[0].df.df;
    EXPECT_DOUBLE_EQ(deriv_test, deriv);
    EXPECT_DOUBLE_EQ(hessian_test, hessian);

}

// Hessian of a bivariate function
TEST(hessian, two_dimensional) {
    using namespace ad;
    using T = double;
    T x[] = { M_PI / 3 , M_PI / 6 };
    arma::Mat<T> hess;
    hessian(hess, G_lmda, x, x + 2);
    EXPECT_DOUBLE_EQ(hess(0, 0), -0.75);
    EXPECT_DOUBLE_EQ(hess(1, 1), -0.75);
    EXPECT_DOUBLE_EQ(hess(0, 1), -0.25);
    EXPECT_DOUBLE_EQ(hess(1, 0), -0.25);
}

// Hessian multivariate
TEST(hessian, multi_dimensional) {
    using namespace ad;
    using T = double;
    // DON'T CHANGE THESE NUMBERS
    T x[] = { 1.,2.,3.,4. };
    arma::Mat<T> hess;
    arma::Mat<T> grad;
    hessian(hess, grad, H_lmda, x, x + 4);
    h_hess_test(hess);
    h_grad_test(grad);
}

} // end namespace
