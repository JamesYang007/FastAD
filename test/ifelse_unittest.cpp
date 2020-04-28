#include "gtest/gtest.h"
#include <fastad_bits/ifelse.hpp>
#include <fastad_bits/math.hpp>

namespace ad {
namespace core {

struct ifelse_fixture : ::testing::Test
{
protected:
    Var<double> x{1.}, y{2.}, z{3.}, w{4.};
};

TEST_F(ifelse_fixture, cond_expr_simple)
{
    auto expr = (x < y); 
    bool cond = expr.feval();
    EXPECT_TRUE(cond);
}

TEST_F(ifelse_fixture, cond_expr_two_cond)
{
    auto expr = (x < y) || (z >= w); 
    bool cond = expr.feval();
    EXPECT_TRUE(cond);
}

TEST_F(ifelse_fixture, cond_expr_three_cond)
{
    auto expr = (x > y) || ((z >= w) && (x == z)); 
    bool cond = expr.feval();
    EXPECT_FALSE(cond);
}

TEST_F(ifelse_fixture, ifelse_simple)
{
    auto expr = if_else(x < y, x, y);
    double value = expr.feval();
    EXPECT_DOUBLE_EQ(value, 1.);
    expr.beval(1.);
    EXPECT_DOUBLE_EQ(x.get_adjoint(), 1.); // updated
    EXPECT_DOUBLE_EQ(y.get_adjoint(), 0.); // unchanged
}

TEST_F(ifelse_fixture, ifelse_simple_negated)
{
    auto expr = if_else(x >= y, x, y);
    double value = expr.feval();
    EXPECT_DOUBLE_EQ(value, 2.);
    expr.beval(1.);
    EXPECT_DOUBLE_EQ(y.get_adjoint(), 1.); // updated
    EXPECT_DOUBLE_EQ(x.get_adjoint(), 0.); // unchanged
}

TEST_F(ifelse_fixture, ifelse_complicated)
{
    auto expr = if_else(
        (x < y) && (z < w),
        x * y + z,
        x);
    double value = expr.feval();
    EXPECT_DOUBLE_EQ(value, 5.);
    expr.beval(1.);
    EXPECT_DOUBLE_EQ(x.get_adjoint(), 2.);
    EXPECT_DOUBLE_EQ(y.get_adjoint(), 1.);
    EXPECT_DOUBLE_EQ(z.get_adjoint(), 1.);
}

TEST_F(ifelse_fixture, if_on_if)
{
    auto expr = 
        if_else(
            (x < y),
            if_else(
                z < w,
                x * y + z,
                x
            ),
            ad::constant(0.)
        );
    double value = expr.feval();
    EXPECT_DOUBLE_EQ(value, 5.);

    expr.beval(1.);
    EXPECT_DOUBLE_EQ(x.get_adjoint(), 2.);
    EXPECT_DOUBLE_EQ(y.get_adjoint(), 1.);
    EXPECT_DOUBLE_EQ(z.get_adjoint(), 1.);

    x.reset_adjoint();
    y.reset_adjoint();
    z.reset_adjoint();

    expr.beval(1.);
    EXPECT_DOUBLE_EQ(x.get_adjoint(), 2.);
    EXPECT_DOUBLE_EQ(y.get_adjoint(), 1.);
    EXPECT_DOUBLE_EQ(z.get_adjoint(), 1.);
}

} // namespace core
} // namespace ad
