#include <random>
#include <array>
#include <time.h>
#include <fastad_bits/node.hpp>
#include "base_fixture.hpp"
#include "gtest/gtest.h"

namespace ad {
namespace core {

struct adnode_fixture: ::testing::Test
{
protected:
    double df_x, df_y;
    double df_expr_x;
    double df_expr_y;
    LeafNode<double> leaf_x, leaf_y; 
    MockExpr<double> expr_x;    // generic mock expression
                                // the only common feature about all nodes is how they handle dualnum
    MockExpr<double> expr_y; 
    UnaryNode<double, MockUnary<double>, MockExpr<double>> unary;
    BinaryNode<double, MockBinary<double>, MockExpr<double>, MockExpr<double>> binary_x;
    BinaryNode<double, MockBinary<double>, MockExpr<double>, MockExpr<double>> binary_xy;
    EqNode<double, MockExpr<double>> eq_x;
    EqNode<double, MockExpr<double>> eq_y;
    GlueNode<double, EqNode<double, MockExpr<double>>, 
        EqNode<double, MockExpr<double>>> glue_xy;
    MockExpr<double> exprs[3] = {1., 2., 3.};

    double seed = 3.;

    using mock_lmda_t = std::function<MockExpr<double>(const MockExpr<double>&)>;
    mock_lmda_t mock_lmda = 
        [](const MockExpr<double>& expr) {
            MockExpr<double> tmp(expr);
            tmp.set_value(tmp.get_value() * 2);
            return tmp;
        };

    std::array<ConstNode<double>, 3> const_exprs = {1., 2., 3.};

    adnode_fixture()
        : ::testing::Test()
        , df_x(0.0)
        , df_y(0.0)
        , df_expr_x(0.0)
        , df_expr_y(0.0)
        , leaf_x(3, &df_x)
        , leaf_y(4, &df_y)
        , expr_x(3.0, &df_expr_x)
        , expr_y(4.0, &df_expr_y)
        , unary(expr_x)
        , binary_x(expr_x, expr_x)
        , binary_xy(expr_x, expr_y)
        , eq_x(leaf_x, expr_x)
        , eq_y(leaf_y, expr_y)
        , glue_xy(eq_x, eq_y)
    {}

};

// ForEach
TEST_F(adnode_fixture, foreach_feval)
{
    auto foreach = ad::for_each(exprs, exprs + 3, mock_lmda);
    EXPECT_DOUBLE_EQ(foreach.feval(), 6.);   // last expression forward-evaluated 
}

TEST_F(adnode_fixture, foreach_beval)
{
    auto foreach = ad::for_each(exprs, exprs + 3, mock_lmda);
    foreach.beval(seed);
    EXPECT_DOUBLE_EQ(foreach.get_adjoint(), seed);         // adjoint set to seed
    EXPECT_DOUBLE_EQ(exprs[2].get_adjoint(), seed );       // last adjoint set to seed 
    EXPECT_DOUBLE_EQ(exprs[1].get_adjoint(), 0.);          // no seed passed
    EXPECT_DOUBLE_EQ(exprs[0].get_adjoint(), 0.);          // no seed passed
}

TEST_F(adnode_fixture, foreach_degenerate)
{
    auto foreach = ad::for_each(exprs, exprs, mock_lmda);
    auto fwdval = foreach.feval();
    EXPECT_DOUBLE_EQ(fwdval, 0.);
    foreach.beval(seed);
    EXPECT_DOUBLE_EQ(foreach.get_adjoint(), seed);         // adjoint set to seed
    EXPECT_DOUBLE_EQ(exprs[2].get_adjoint(), 0. );         // noop 
    EXPECT_DOUBLE_EQ(exprs[1].get_adjoint(), 0.);          // noop 
    EXPECT_DOUBLE_EQ(exprs[0].get_adjoint(), 0.);          // noop 
}

} // namespace core
} // namespace ad
