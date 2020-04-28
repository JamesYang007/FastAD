#include "gtest/gtest.h"
#include <fastad_bits/pow.hpp>
#include <fastad_bits/math.hpp>

namespace ad {
namespace core {

struct pow_fixture : ::testing::Test
{
protected:
    Var<double> x{1.}, y{2.}, z{3.}, w{4.};
};

TEST_F(pow_fixture, pow_positive_exp)
{
    auto expr = pow<1>(x + y);
    double value = expr.feval();
    EXPECT_DOUBLE_EQ(value, 3.);
    expr.beval(1.);
    EXPECT_DOUBLE_EQ(x.get_adjoint(), 1.);
    EXPECT_DOUBLE_EQ(y.get_adjoint(), 1.);
}

TEST_F(pow_fixture, pow_positive_exp_complicated)
{
    auto expr = pow<3>(x + y * z);
    double value = expr.feval();
    EXPECT_DOUBLE_EQ(value, 343.);
    expr.beval(1.);
    EXPECT_DOUBLE_EQ(x.get_adjoint(), 3 * 49);
    EXPECT_DOUBLE_EQ(y.get_adjoint(), 3 * 49 * 3);
    EXPECT_DOUBLE_EQ(z.get_adjoint(), 3 * 49 * 2);
}

TEST_F(pow_fixture, pow_zero_exp)
{
    auto expr = pow<0>(x + y * z);
    double value = expr.feval();
    EXPECT_DOUBLE_EQ(value, 1.);
    expr.beval(1.);
    EXPECT_DOUBLE_EQ(x.get_adjoint(), 0);
    EXPECT_DOUBLE_EQ(y.get_adjoint(), 0);
    EXPECT_DOUBLE_EQ(z.get_adjoint(), 0);
}

TEST_F(pow_fixture, pow_zero_exp_zero_value)
{
    auto expr = pow<0>(x - x);
    double value = expr.feval();
    EXPECT_DOUBLE_EQ(value, 1.);
    expr.beval(1.);
    EXPECT_DOUBLE_EQ(x.get_adjoint(), 0);
}

TEST_F(pow_fixture, pow_negative_exp_pos_value)
{
    auto expr = pow<-1>(x + y);
    double value = expr.feval();
    EXPECT_DOUBLE_EQ(value, 1./3);
    expr.beval(1.);
    EXPECT_DOUBLE_EQ(x.get_adjoint(), -1./9);
    EXPECT_DOUBLE_EQ(y.get_adjoint(), -1./9);
}

TEST_F(pow_fixture, pow_negative_large_exp_pos_value)
{
    auto expr = pow<-3>(x + y);
    double value = expr.feval();
    EXPECT_DOUBLE_EQ(value, 1./27);
    expr.beval(1.);
    EXPECT_DOUBLE_EQ(x.get_adjoint(), -1./27);
    EXPECT_DOUBLE_EQ(y.get_adjoint(), -1./27);
}

TEST_F(pow_fixture, pow_on_pow)
{
    auto expr = pow<2>(pow<1>(x + y));
    double value = expr.feval();
    EXPECT_DOUBLE_EQ(value, 9.);
    expr.beval(1.);
    EXPECT_DOUBLE_EQ(x.get_adjoint(), 2 * 3);
    EXPECT_DOUBLE_EQ(y.get_adjoint(), 2 * 3);

    x.reset_adjoint();
    y.reset_adjoint();

    expr.beval(1.);
    EXPECT_DOUBLE_EQ(x.get_adjoint(), 2 * 3);
    EXPECT_DOUBLE_EQ(y.get_adjoint(), 2 * 3);
}

} // namespace core
} // namespace ad
