#define _USE_MATH_DEFINES

#include <fastad/adeval.hpp>
#include <fastad/admath.hpp>
#include <fastad/adjacobian.hpp>
#include "gtest/gtest.h"
#include <armadillo>

namespace {

auto F_lmda = MAKE_LMDA(
    ad::sin(x[0]) + x[1],
    w[0] * x[1]
);

auto G_lmda = MAKE_LMDA(
    ad::cos(x[0])
);

auto H_lmda = MAKE_LMDA(
    ad::log(x[0])
);

auto F = ad::make_function(F_lmda);
auto G = ad::make_function(G_lmda);
auto H = ad::make_function(H_lmda);

// Scalar of Scalar(x_0, x_1)
TEST(adcompose, scalar_scalar) {
    using namespace ad;
    double x[] = { M_PI, 1. };
    auto G_F = compose(G, F);
    auto&& expr = G_F(x, x + 2);
    autodiff(expr);

    double tmp = -std::sin((std::sin(x[0]) + x[1])*x[1]);
    EXPECT_DOUBLE_EQ(G_F.x[0].df, tmp*std::cos(x[0])*x[1]);
    EXPECT_DOUBLE_EQ(G_F.x[1].df, tmp*(2 * x[1] + std::sin(x[0])));
}

// Scalar of Scalar of Scalar(x_0, x_1)
TEST(adcompose, scalar_scalar_scalar) {
    using namespace ad;
    double x[] = { 0.268* M_PI, 2.3 };
    auto H_G_F = compose(H, G, F);
    auto&& expr = H_G_F(x, x + 2);
    auto f_val = autodiff(expr);

    double tmp = -std::sin((std::sin(x[0]) + x[1])*x[1]);
    double tmp2 = std::exp(-f_val);
    EXPECT_DOUBLE_EQ(H_G_F.x[0].df, tmp2 * tmp * std::cos(x[0])*x[1]);
    EXPECT_DOUBLE_EQ(H_G_F.x[1].df, tmp2 * tmp * (2 * x[1] + std::sin(x[0])));
}


// Scalar of Vector(x_0, x_1)
TEST(adcompose, scalar_vector) {
    using namespace ad;
    double x[] = { 0.268 * M_PI, 2.3 };
    auto F = make_function(F_lmda);
    auto FG = make_function(F_lmda, G_lmda);
    auto F_FG = compose(F, FG);

    auto&& expr = F_FG(x, x + 2);
    autodiff(expr);

    // check
    double x_tilde[2] = { 0 };
    x_tilde[0] = Evaluate(F(x, x + 2));
    x_tilde[1] = Evaluate(G(x, x + 2));

    arma::Mat<double> grad_F;
    arma::Mat<double> jacobi_FG;
    jacobian(grad_F, x_tilde, x_tilde + 2, F_lmda);
    jacobian(jacobi_FG, x, x + 2, F_lmda, G_lmda);
    arma::Mat<double> comp = grad_F * jacobi_FG;

    EXPECT_DOUBLE_EQ(F_FG.x[0].df, comp(0, 0));
    EXPECT_DOUBLE_EQ(F_FG.x[1].df, comp(0, 1));
}

// Vector of Vector
TEST(adcompose, vector_vector) {
    using namespace ad;
    double x[] = { 0.2, 1.59 };
    auto FG = make_function(F_lmda, G_lmda);
    auto GH = make_function(G_lmda, H_lmda);
    auto FG_GH = compose(FG, GH);
    auto&& expr = FG_GH(x, x + 2);
    autodiff(expr);

    arma::Mat<double> FG_GH_jacobi;
    jacobian(FG_GH_jacobi, FG_GH);

    // test
    double x_tilde[2] = { 0 };
    auto G = make_function(G_lmda);
    auto H = make_function(H_lmda);

    x_tilde[0] = Evaluate(G(x, x + 2));
    x_tilde[1] = Evaluate(H(x, x + 2));

    arma::Mat<double> FG_jacobi;
    arma::Mat<double> GH_jacobi;
    jacobian(FG_jacobi, x_tilde, x_tilde + 2, F_lmda, G_lmda);
    jacobian(GH_jacobi, x, x + 2, G_lmda, H_lmda);
    arma::Mat<double> comp = FG_jacobi * GH_jacobi;

    // Compare
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            EXPECT_DOUBLE_EQ(FG_GH_jacobi(i, j), comp(i, j));
}

} // end namespace 
