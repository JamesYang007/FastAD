#include <random>
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

TEST_F(adnode_fixture, leafnode_feval)
{
    // check forward evaluation
    EXPECT_DOUBLE_EQ(leaf_x.feval(), 3);
}

TEST_F(adnode_fixture, leafnode_get_value)
{
    EXPECT_DOUBLE_EQ(leaf_x.get_value(), 3);
}

TEST_F(adnode_fixture, leafnode_beval)
{
    // check backward evaluation
    leaf_x.beval(seed);
    EXPECT_DOUBLE_EQ((leaf_x.get_adjoint()), seed);
    EXPECT_DOUBLE_EQ(df_x, seed);
}

TEST_F(adnode_fixture, unarynode_feval) 
{
    // check forward evaluation
    EXPECT_DOUBLE_EQ(unary.feval(), 6.);
}

TEST_F(adnode_fixture, unarynode_beval) 
{
    // check backward evaluation
    unary.beval(seed);
    EXPECT_DOUBLE_EQ(unary.get_adjoint(), seed);        // unary has a copy of expr inside it
    EXPECT_DOUBLE_EQ(expr_x.get_curr_adjoint(), 0);     // expression should not have been modified
    EXPECT_DOUBLE_EQ(df_expr_x, 2*seed);                // expr copy in unary has this adjoint value
}

// BinaryNode
TEST_F(adnode_fixture, binarynode_feval) 
{
    // check forward evaluation
    EXPECT_DOUBLE_EQ(binary_xy.feval(), 7.);
}

TEST_F(adnode_fixture, binarynode_beval_same_expr)
{
    // check backward evaluation
    binary_x.beval(seed); // seed * df/dx + seed * df/dx; df/dx = 1
    EXPECT_DOUBLE_EQ(df_expr_x, 2*seed);
}

TEST_F(adnode_fixture, binarynode_beval_diff_expr)
{
    // check backward evaluation
    binary_xy.beval(seed);    // seed * df/dx; seed * df/dy; df/dx = df/dy = 1
    EXPECT_DOUBLE_EQ(df_expr_x, seed);
    EXPECT_DOUBLE_EQ(df_expr_y, seed);
}

// EqNode
TEST_F(adnode_fixture, eqnode_feval)
{
    // check forward evaluation
    EXPECT_DOUBLE_EQ(eq_x.feval(), 3.0);
}

TEST_F(adnode_fixture, eqnode_beval)
{
    // check backward evaluation
    eq_x.beval(seed);
    EXPECT_DOUBLE_EQ(leaf_x.get_adjoint(), seed);   // full adjoint updated
    EXPECT_DOUBLE_EQ(expr_x.get_curr_adjoint(), 0.); // current adjoint should not be updated
    EXPECT_DOUBLE_EQ(expr_x.get_adjoint(), seed);   // expression adjoints updated
}

// GlueNode 
TEST_F(adnode_fixture, gluenode_feval)
{
    // check forward evaluation
    EXPECT_DOUBLE_EQ(glue_xy.feval(), 4.);    // take the right side value
}

TEST_F(adnode_fixture, gluenode_beval_same_expr)
{
    // check backward evaluation
    glue_xy.beval(seed);
    EXPECT_DOUBLE_EQ(glue_xy.get_adjoint(), seed);    // adjoint simply set to seed
    EXPECT_DOUBLE_EQ(leaf_y.get_adjoint(), seed);    // right side of GlueNode gets seed
    EXPECT_DOUBLE_EQ(leaf_x.get_adjoint(), 0.);     // left side of GlueNode does not get seed
}

TEST_F(adnode_fixture, gluenode_glue_size_2)
{
    using eq_t = EqNode<double, double>;
    using glue_t = GlueNode<double, eq_t, eq_t>; 
    EXPECT_EQ(details::glue_size<glue_t>::value, static_cast<size_t>(2));
}

TEST_F(adnode_fixture, gluenode_glue_size_3)
{
    using eq_t = EqNode<double, double>;
    using glue_t = GlueNode<double, eq_t, eq_t>; 
    using glue_glue_t = GlueNode<double, glue_t, eq_t>;
    EXPECT_EQ(details::glue_size<glue_glue_t>::value, static_cast<size_t>(3));
}

// ConstNode
TEST_F(adnode_fixture, constnode_feval)
{
    auto constnode = ad::constant(2.);
    EXPECT_DOUBLE_EQ(constnode.feval(), 2.);
}

TEST_F(adnode_fixture, constnode_beval)
{
    auto constnode = ad::constant(2.);
    constnode.feval();
    EXPECT_DOUBLE_EQ(constnode.get_adjoint(), 0.);
}

// SumNode
TEST_F(adnode_fixture, sumnode_feval)
{
    auto sumnode = sum(exprs, exprs + 3, mock_lmda);
    EXPECT_DOUBLE_EQ(sumnode.feval(), 12.);
}

TEST_F(adnode_fixture, sumnode_beval)
{
    auto sumnode = sum(exprs, exprs + 3, mock_lmda);
    sumnode.beval(seed);
    EXPECT_DOUBLE_EQ(exprs[0].get_adjoint(), seed);
    EXPECT_DOUBLE_EQ(exprs[1].get_adjoint(), seed);
    EXPECT_DOUBLE_EQ(exprs[2].get_adjoint(), seed);
}

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

} // namespace core
} // namespace ad
