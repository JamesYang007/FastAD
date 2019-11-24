#include <fastad_bits/node.hpp>
#include <fastad_bits/math.hpp>
#include <fastad_bits/vec.hpp>
#include <fastad_bits/eval.hpp>
#include "gtest/gtest.h"

namespace ad {
namespace core {

struct node_integration_fixture: ::testing::Test
{
protected:
};

////////////////////////////////////////////////////////////
// LeafNode, UnaryNode Integration Test 
////////////////////////////////////////////////////////////

// LeafNode -> UnaryNode 

TEST_F(node_integration_fixture, leaf_unary) 
{
    Var<double> x(3.1);
    auto expr = ad::sin(x);
    EXPECT_DOUBLE_EQ(autodiff(expr), std::sin(3.1));
    EXPECT_DOUBLE_EQ(x.get_adjoint(), std::cos(3.1));
}

// LeafNode -> UnaryNode -> UnaryNode

TEST_F(node_integration_fixture, leaf_unary_unary)
{
    Var<double> x(3.1);
    auto expr = ad::sin(ad::log(x));
    EXPECT_DOUBLE_EQ(autodiff(expr), std::sin(std::log(3.1)));
    EXPECT_DOUBLE_EQ(x.get_adjoint(), std::cos(std::log(3.1)) / 3.1);
}

////////////////////////////////////////////////////////////
// LeafNode, BinaryNode Integration Test 
////////////////////////////////////////////////////////////

// LeafNode, LeafNode -> BinaryNode

TEST_F(node_integration_fixture, leaf_binary)
{
    Var<double> x(1.), y(2.);
    auto expr = x + y;
    EXPECT_DOUBLE_EQ(autodiff(expr), 3.);
    EXPECT_DOUBLE_EQ(x.get_adjoint(), 1.);
    EXPECT_DOUBLE_EQ(y.get_adjoint(), 1.);
}

// LeafNode, LeafNode -> BinaryNode, LeafNode -> BinaryNode

TEST_F(node_integration_fixture, leaf_leaf_binary)
{
    Var<double> x(1.), y(2.), z(3.);
    auto expr = (x + y) - z;
    EXPECT_DOUBLE_EQ(autodiff(expr), 0.);
    EXPECT_DOUBLE_EQ(x.get_adjoint(), 1.);
    EXPECT_DOUBLE_EQ(y.get_adjoint(), 1.);
    EXPECT_DOUBLE_EQ(z.get_adjoint(), -1.);
}

////////////////////////////////////////////////////////////
// LeafNode, EqNode Integration Test 
////////////////////////////////////////////////////////////

// LeafNode, LeafNode -> EqNode
TEST_F(node_integration_fixture, leaf_eq)
{
    Var<double> x(1.), y(2.);
    auto expr = make_eq(x, y);
    EXPECT_DOUBLE_EQ(autodiff(expr), 2.);
    EXPECT_DOUBLE_EQ(x.get_adjoint(), 1.);
    EXPECT_DOUBLE_EQ(y.get_adjoint(), 1.);
}

////////////////////////////////////////////////////////////
// LeafNode, UnaryNode, BinaryNode Integration Test 
////////////////////////////////////////////////////////////

TEST_F(node_integration_fixture, leaf_unary_binary) {
    double x1 = 2.0, x2 = 1.31, x3 = -3.14;
    double dfs[3] = { 0 };
    Var<double> leaf1(x1, dfs);
    Var<double> leaf2(x2, dfs + 1);
    Var<double> leaf3(x3, dfs + 2);

    auto res = leaf1 + sin(leaf2 + leaf3);
    EXPECT_EQ(res.feval(), x1 + std::sin(x2 + x3));
    res.beval(1);
    EXPECT_EQ(dfs[0], 1);
    EXPECT_EQ(dfs[1], std::cos(x2 + x3));
    EXPECT_EQ(dfs[2], std::cos(x2 + x3));
}

TEST_F(node_integration_fixture, leaf_unary_binary_2) {
    double x1 = 1.2041, x2 = -2.2314;
    double dfs[2] = { 0 };
    Var<double> leaf1(x1, dfs);
    Var<double> leaf2(x2, dfs + 1);

    EXPECT_EQ(leaf1.feval(), x1);
    EXPECT_EQ(leaf2.get_value(), x2);

    auto res = leaf1 * leaf2 + sin(leaf1);
    EXPECT_EQ(res.feval(), x1*x2 + std::sin(x1));
    res.beval(1);
    EXPECT_EQ(dfs[0], x2 + std::cos(x1));
    EXPECT_EQ(dfs[1], x1);
}

TEST_F(node_integration_fixture, leaf_unary_binary_3) {
    double x1 = 1.2041, x2 = -2.2314;
    double dfs[2] = { 0 };
    Var<double> leaf1(x1, dfs);
    Var<double> leaf2(x2, dfs + 1);

    auto res = leaf1 * leaf2 + sin(leaf1 + leaf2) * leaf2 - leaf1 / leaf2;
    EXPECT_EQ(res.feval(), x1*x2 + std::sin(x1 + x2)*x2 - x1 / x2);
    res.beval(1);
    EXPECT_EQ(dfs[0],
        x2 + std::cos(x1 + x2) * x2 - 1. / x2);
    EXPECT_EQ(dfs[1],
        x1 + std::cos(x1 + x2) * x2 + std::sin(x1 + x2) + x1 / (x2*x2));
}

TEST_F(node_integration_fixture, leaf_unary_binary_4) {
    double x1 = 1.5928, x2 = -0.291, x3 = 5.1023;
    double dfs[3] = { 0 };
    Var<double> leaf1(x1, dfs);
    Var<double> leaf2(x2, dfs + 1);
    Var<double> leaf3(x3, dfs + 2);

    auto res =
        leaf1 * leaf3 + sin(cos(leaf1 + leaf2)) * leaf2 - leaf1 / exp(leaf3);
    EXPECT_EQ(res.feval(),
        x1*x3 + std::sin(std::cos(x1 + x2))*x2 - x1 / std::exp(x3));
    res.beval(1);
    EXPECT_EQ(dfs[0],
        x3 - x2 * std::cos(std::cos(x1 + x2))*std::sin(x1 + x2) - std::exp(-x3));
    EXPECT_EQ(dfs[1],
        std::sin(std::cos(x1 + x2))
        - x2 * std::cos(std::cos(x1 + x2))*std::sin(x1 + x2));
    EXPECT_EQ(dfs[2], x1 + x1 * std::exp(-x3));
}

////////////////////////////////////////////////////////////
// LeafNode, UnaryNode, EqNode Integration Test 
////////////////////////////////////////////////////////////

// LeafNode, (LeafNode -> UnaryNode) -> EqNode
TEST_F(node_integration_fixture, leaf_unary_eq)
{
    Var<double> x(1.), y(2.);
    auto expr = (x = ad::tan(y));
    EXPECT_DOUBLE_EQ(autodiff(expr), std::tan(2.));
    EXPECT_DOUBLE_EQ(x.get_adjoint(), 1.);
    EXPECT_DOUBLE_EQ(y.get_adjoint(), 1./(std::cos(2.) * std::cos(2.)));
}

////////////////////////////////////////////////////////////
// LeafNode, UnaryNode, BinaryNode, EqNode, GlueNode Integration Test 
////////////////////////////////////////////////////////////

TEST_F(node_integration_fixture, leaf_binary_eq_glue) 
{
    LeafNode<double> w1(1.0), w2(2.0), w3(3.0), w4(4.0);
    auto expr = (w3 = w1 * w2, w4 = w3 * w3);
    expr.feval();
    expr.beval(1);
    EXPECT_EQ(w4.get_adjoint(), 1.0);
    EXPECT_EQ(w3.get_adjoint(), 2 * w3.get_value());
    EXPECT_EQ(w2.get_adjoint(), 2 * w2.get_value()*w1.get_value()*w1.get_value());
    EXPECT_EQ(w1.get_adjoint(), 2 * w1.get_value()*w2.get_value()*w2.get_value());
}

TEST_F(node_integration_fixture, leaf_unary_binary_eq_glue) 
{
    double x1 = -0.201, x2 = 1.2241;
    LeafNode<double> w1(x1), w2(x2), w3, w4, w5;
    auto expr = (
        w3 = w1 * sin(w2)
        , w4 = w3 + w1 * w2
        , w5 = exp(w4*w3)
        );
    expr.feval();
    EXPECT_EQ(w5.get_value(), std::exp((x1*std::sin(x2) + x1 * x2)*(x1*std::sin(x2))));
    EXPECT_EQ(w4.get_value(), x1*std::sin(x2) + x1 * x2);
    EXPECT_EQ(w3.get_value(), x1*std::sin(x2));

    expr.beval(1);
    EXPECT_EQ(w5.get_adjoint(), 1);
    EXPECT_EQ(w4.get_adjoint(), w3.get_value() * w5.get_value());
    EXPECT_EQ(w3.get_adjoint(), (w3.get_value() + w4.get_value()) * w5.get_value());
    EXPECT_EQ(w2.get_adjoint(), 
            w5.get_value()*x1*x1*
            (std::cos(x2)*(std::sin(x2) + x2) + std::sin(x2)*(1 + std::cos(x2))));
    EXPECT_EQ(w1.get_adjoint(), w5.get_value() * 2 * x1 * std::sin(x2) *(std::sin(x2) + x2));
}

TEST_F(node_integration_fixture, sumnode) {
    Var<double> vec[3] = { 0.203104, 1.4231, -1.231 };
    auto expr = ad::sum(vec, vec + 3,
        [](Var<double> const& v) {return ad::cos(ad::sin(v)*v); });
    EXPECT_EQ(expr.get_value(), 0);
    EXPECT_EQ(expr.get_adjoint(), 0);

    double actual_sum = 0;
    for (size_t i = 0; i < 3; ++i) {
        actual_sum += std::cos(std::sin(vec[i].get_value())*vec[i].get_value());
    }

    EXPECT_DOUBLE_EQ(expr.feval(), actual_sum);
    expr.beval(1);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(vec[i].get_adjoint(),
            -std::sin(std::sin(vec[i].get_value()) *
            vec[i].get_value()) * 
            (std::cos(vec[i].get_value()) *
            vec[i].get_value() + 
            std::sin(vec[i].get_value())));
    }
    EXPECT_DOUBLE_EQ(expr.get_value(), actual_sum);
}

TEST_F(node_integration_fixture, foreach) {
    Vec<double> vec({ 100., 20., -10. });
    vec.emplace_back(1e-3);
    Vec<double> prod(4);
    prod[0] = vec[0];
    auto&& expr = ad::for_each(
        boost::counting_iterator<size_t>(1)
        , boost::counting_iterator<size_t>(4)
        , [&vec, &prod](size_t i) {return prod[i] = prod[i - 1] * vec[i]; }
    );

    double actual = 1;
    for (size_t i = 0; i < vec.size(); ++i) {
        actual *= vec[i].get_value();
    }

    Var<double> res, w4;
    auto expr2 = (res = expr, w4 = res * res + vec[0]);
    expr2.feval();
    expr2.beval(1);

    EXPECT_DOUBLE_EQ(res.get_value(), actual);
    EXPECT_DOUBLE_EQ(w4.get_value(), actual*actual + vec[0].get_value());
    for (size_t i = 0; i < vec.size(); ++i) {
        EXPECT_DOUBLE_EQ(vec[i].get_adjoint(),
            ((i == 0) ? 1 : 0) + 2 * actual * actual / vec[i].get_value());
    }
}

} // namespace core
} // namespace ad
