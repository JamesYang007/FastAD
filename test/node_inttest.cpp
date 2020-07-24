#include "gtest/gtest.h"
#include <fastad_bits/eval.hpp>
#include <fastad_bits/math.hpp>
#include <fastad_bits/var.hpp>
#include <fastad_bits/eq.hpp>
#include <fastad_bits/glue.hpp>
#include <fastad_bits/sum.hpp>
#include <fastad_bits/for_each.hpp>

namespace ad {
namespace core {

struct node_integration_fixture: ::testing::Test
{
protected:
    using value_t = double;
    std::vector<value_t> val_buf;

    node_integration_fixture()
        : val_buf()
    {}

    template <class T>
    void bind(T& expr)
    {
        size_t buf_size = expr.bind_size();
        val_buf.resize(buf_size);
        expr.bind(val_buf.data());
    }
};

////////////////////////////////////////////////////////////
// LeafNode, UnaryNode Integration Test 
////////////////////////////////////////////////////////////

// LeafNode -> UnaryNode 

TEST_F(node_integration_fixture, leaf_unary) 
{
    Var<double> x(3.1);
    auto expr = ad::sin(x);
    bind(expr);
    EXPECT_DOUBLE_EQ(autodiff(expr), std::sin(3.1));
    EXPECT_DOUBLE_EQ(x.get_adj(0,0), std::cos(3.1));
}

// LeafNode -> -UnaryNode 

TEST_F(node_integration_fixture, leaf_unary_minus) 
{
    Var<double> x(3.1);
    auto expr = -x;
    bind(expr);
    EXPECT_DOUBLE_EQ(autodiff(expr), -3.1);
    EXPECT_DOUBLE_EQ(x.get_adj(0,0), -1.);
}

// LeafNode -> UnaryNode -> UnaryNode

TEST_F(node_integration_fixture, leaf_unary_unary)
{
    Var<double> x(3.1);
    auto expr = ad::sin(ad::log(x));
    bind(expr);
    EXPECT_DOUBLE_EQ(autodiff(expr), std::sin(std::log(3.1)));
    EXPECT_DOUBLE_EQ(x.get_adj(0,0), std::cos(std::log(3.1)) / 3.1);
}

////////////////////////////////////////////////////////////
// LeafNode, BinaryNode Integration Test 
////////////////////////////////////////////////////////////

// LeafNode, LeafNode -> BinaryNode

TEST_F(node_integration_fixture, leaf_binary)
{
    Var<double> x(1.), y(2.);
    auto expr = x + y;
    bind(expr);
    EXPECT_DOUBLE_EQ(autodiff(expr), 3.);
    EXPECT_DOUBLE_EQ(x.get_adj(0,0), 1.);
    EXPECT_DOUBLE_EQ(y.get_adj(0,0), 1.);
}

// LeafNode, LeafNode -> BinaryNode, LeafNode -> BinaryNode

TEST_F(node_integration_fixture, leaf_leaf_binary)
{
    Var<double> x(1.), y(2.), z(3.);
    auto expr = (x + y) - z;
    bind(expr);
    EXPECT_DOUBLE_EQ(autodiff(expr), 0.);
    EXPECT_DOUBLE_EQ(x.get_adj(0,0), 1.);
    EXPECT_DOUBLE_EQ(y.get_adj(0,0), 1.);
    EXPECT_DOUBLE_EQ(z.get_adj(0,0), -1.);
}

////////////////////////////////////////////////////////////
// LeafNode, UnaryNode, BinaryNode Integration Test 
////////////////////////////////////////////////////////////

TEST_F(node_integration_fixture, leaf_unary_binary) {
    double x1 = 2.0, x2 = 1.31, x3 = -3.14;
    double dfs[3] = { 0 };
    VarView<double> leaf1(&x1, dfs);
    VarView<double> leaf2(&x2, dfs+1);
    VarView<double> leaf3(&x3, dfs+2);

    auto expr = leaf1 + ad::sin(leaf2 + leaf3);
    bind(expr);
    EXPECT_EQ(expr.feval(), x1 + std::sin(x2 + x3));
    expr.beval(1,0,0);
    EXPECT_EQ(dfs[0], 1);
    EXPECT_EQ(dfs[1], std::cos(x2 + x3));
    EXPECT_EQ(dfs[2], std::cos(x2 + x3));
}

TEST_F(node_integration_fixture, leaf_unary_binary_2) {
    double x1 = 1.2041, x2 = -2.2314;
    double dfs[2] = { 0 };
    VarView<double> leaf1(&x1, dfs);
    VarView<double> leaf2(&x2, dfs + 1);

    EXPECT_EQ(leaf1.feval(), x1);
    EXPECT_EQ(leaf2.get(), x2);

    auto expr = leaf1 * leaf2 + sin(leaf1);
    bind(expr);
    EXPECT_EQ(expr.feval(), x1*x2 + std::sin(x1));
    expr.beval(1,0,0);
    EXPECT_EQ(dfs[0], x2 + std::cos(x1));
    EXPECT_EQ(dfs[1], x1);
}

TEST_F(node_integration_fixture, leaf_unary_binary_3) {
    double x1 = 1.2041, x2 = -2.2314;
    double dfs[2] = { 0 };
    VarView<double> leaf1(&x1, dfs);
    VarView<double> leaf2(&x2, dfs + 1);

    auto expr = leaf1 * leaf2 + sin(leaf1 + leaf2) * leaf2 - leaf1 / leaf2;
    bind(expr);
    EXPECT_EQ(expr.feval(), x1*x2 + std::sin(x1 + x2)*x2 - x1 / x2);
    expr.beval(1,0,0);
    EXPECT_EQ(dfs[0],
        x2 + std::cos(x1 + x2) * x2 - 1. / x2);
    EXPECT_EQ(dfs[1],
        x1 + std::cos(x1 + x2) * x2 + std::sin(x1 + x2) + x1 / (x2*x2));
}

TEST_F(node_integration_fixture, leaf_unary_binary_4) {
    double x1 = 1.5928, x2 = -0.291, x3 = 5.1023;
    double dfs[3] = { 0 };
    VarView<double> leaf1(&x1, dfs);
    VarView<double> leaf2(&x2, dfs + 1);
    VarView<double> leaf3(&x3, dfs + 2);

    auto expr =
        leaf1 * leaf3 + sin(cos(leaf1 + leaf2)) * leaf2 - leaf1 / exp(leaf3);
    bind(expr);
    EXPECT_EQ(expr.feval(),
        x1*x3 + std::sin(std::cos(x1 + x2))*x2 - x1 / std::exp(x3));
    expr.beval(1,0,0);
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
    bind(expr);
    EXPECT_DOUBLE_EQ(autodiff(expr), std::tan(2.));
    EXPECT_DOUBLE_EQ(x.get_adj(0,0), 1.);
    EXPECT_DOUBLE_EQ(y.get_adj(0,0), 1./(std::cos(2.) * std::cos(2.)));
}

////////////////////////////////////////////////////////////
// LeafNode, UnaryNode, BinaryNode, EqNode, GlueNode Integration Test 
////////////////////////////////////////////////////////////

TEST_F(node_integration_fixture, leaf_binary_eq_glue) 
{
    Var<double> w1(1.0), w2(2.0), w3(3.0), w4(4.0);
    auto expr = (w3 = w1 * w2, w4 = w3 * w3);
    bind(expr);

    EXPECT_DOUBLE_EQ(expr.feval(), 4.);
    EXPECT_DOUBLE_EQ(w1.get(), 1.);
    EXPECT_DOUBLE_EQ(w2.get(), 2.);
    EXPECT_DOUBLE_EQ(w3.get(), 2.);
    EXPECT_DOUBLE_EQ(w4.get(), 4.);

    expr.beval(1,0,0);
    EXPECT_EQ(w4.get_adj(0,0), 1.0);
    EXPECT_EQ(w3.get_adj(0,0), 2 * w3.get());
    EXPECT_EQ(w2.get_adj(0,0), 2 * w2.get()*w1.get()*w1.get());
    EXPECT_EQ(w1.get_adj(0,0), 2 * w1.get()*w2.get()*w2.get());
}

TEST_F(node_integration_fixture, leaf_unary_binary_eq_glue) 
{
    double x1 = -0.201, x2 = 1.2241;
    Var<double> w1(x1), w2(x2), w3, w4, w5;
    auto expr = (
        w3 = w1 * sin(w2),
        w4 = w3 + w1 * w2,
        w5 = exp(w4*w3)
        );
    bind(expr);

    expr.feval();
    EXPECT_EQ(w5.get(), std::exp((x1*std::sin(x2) + x1 * x2)*(x1*std::sin(x2))));
    EXPECT_EQ(w4.get(), x1*std::sin(x2) + x1 * x2);
    EXPECT_EQ(w3.get(), x1*std::sin(x2));

    expr.beval(1,0,0);
    EXPECT_EQ(w5.get_adj(0,0), 1);
    EXPECT_EQ(w4.get_adj(0,0), w3.get() * w5.get());
    EXPECT_EQ(w3.get_adj(0,0), (w3.get() + w4.get()) * w5.get());
    EXPECT_EQ(w2.get_adj(0,0), 
            w5.get()*x1*x1*
            (std::cos(x2)*(std::sin(x2) + x2) + std::sin(x2)*(1 + std::cos(x2))));
    EXPECT_EQ(w1.get_adj(0,0), w5.get() * 2 * x1 * std::sin(x2) *(std::sin(x2) + x2));
}

TEST_F(node_integration_fixture, sumnode) {
    Var<double> vec[3] = { 0.203104, 1.4231, -1.231 };
    auto expr = ad::sum(vec, vec + 3,
        [](Var<double> const& v) {return ad::cos(ad::sin(v)*v); });
    bind(expr);

    double actual_sum = 0;
    for (size_t i = 0; i < 3; ++i) {
        actual_sum += std::cos(std::sin(vec[i].get())*vec[i].get());
    }

    EXPECT_DOUBLE_EQ(expr.feval(), actual_sum);
    expr.beval(1,0,0);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(vec[i].get_adj(0,0),
            -std::sin(std::sin(vec[i].get()) *
            vec[i].get()) * 
            (std::cos(vec[i].get()) *
            vec[i].get() + 
            std::sin(vec[i].get())));
    }
    EXPECT_DOUBLE_EQ(expr.get(), actual_sum);

    // Reset adjoint and re-evaluate
    vec[0].reset_adj();
    vec[1].reset_adj();
    vec[2].reset_adj();

    EXPECT_DOUBLE_EQ(expr.feval(), actual_sum);

    expr.beval(1,0,0);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(vec[i].get_adj(0,0),
            -std::sin(std::sin(vec[i].get()) *
            vec[i].get()) * 
            (std::cos(vec[i].get()) *
            vec[i].get() + 
            std::sin(vec[i].get())));
    }
}

TEST_F(node_integration_fixture, foreach) {
    std::vector<ad::Var<double>> vec({ 100., 20., -10. });
    vec.emplace_back(1e-3);
    std::vector<ad::Var<double>> prod(4);
    static_cast<ad::VarView<double>&>(prod[0]) = vec[0];
    auto it_prev = prod.begin();
    auto vec_it = vec.begin();
    auto expr = ad::for_each(std::next(prod.begin()), 
                             prod.end(), 
                             [&](const auto& cur) 
                             { return cur = (*it_prev++) * (*++vec_it); }
        );

    double actual = 1;
    for (size_t i = 0; i < vec.size(); ++i) {
        actual *= vec[i].get();
    }

    Var<double> res, w4;
    auto expr2 = (res = expr, w4 = res * res + vec[0]);
    bind(expr2);

    expr2.feval();
    expr2.beval(1,0,0);

    EXPECT_DOUBLE_EQ(res.get(), actual);
    EXPECT_DOUBLE_EQ(w4.get(), actual*actual + vec[0].get());

    for (size_t i = 0; i < vec.size(); ++i) {
        EXPECT_DOUBLE_EQ(vec[i].get_adj(0,0),
            ((i == 0) ? 1 : 0) + 2 * actual * actual / vec[i].get());
    }
}

} // namespace core
} // namespace ad
