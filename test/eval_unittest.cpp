#include <tuple>
#include <fastad_bits/eval.hpp>
#include "base_fixture.hpp"
#include "gtest/gtest.h"

namespace ad {

struct adeval_fixture: ::testing::Test
{
protected:
    MockExpr<double> expr;
    std::tuple<MockExpr<double>, MockExpr<double>> tup;

    adeval_fixture()
        : expr(1.)
        , tup(1., 2.)
    {}
};

TEST_F(adeval_fixture, evaluate)
{
    EXPECT_DOUBLE_EQ(evaluate(expr), 1.);
}

TEST_F(adeval_fixture, evaluate_adj)
{
    evaluate_adj(expr);
    evaluate_adj(expr); // second time shouldn't change 
    EXPECT_DOUBLE_EQ(expr.get_adjoint(), 1.);
}

TEST_F(adeval_fixture, autodiff)
{
    EXPECT_DOUBLE_EQ(autodiff(expr), 1.);
    EXPECT_DOUBLE_EQ(expr.get_adjoint(), 1.);
}

TEST_F(adeval_fixture, autodiff_tup)
{
    autodiff(tup);
    autodiff(tup);
    EXPECT_DOUBLE_EQ(std::get<0>(tup).get_value(), 1.);
    EXPECT_DOUBLE_EQ(std::get<1>(tup).get_value(), 2.);
    EXPECT_DOUBLE_EQ(std::get<0>(tup).get_adjoint(), 1.);
    EXPECT_DOUBLE_EQ(std::get<1>(tup).get_adjoint(), 1.);
}

} // namespace ad
