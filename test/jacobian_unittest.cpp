#include <array>
#include <fastad_bits/jacobian.hpp>
#include "exgen_fixture.hpp"
#include "gtest/gtest.h"

namespace ad {
namespace details {

struct jacobian_fixture: ::testing::Test
{
protected:
    std::array<double, 4> adj = {-1};
    std::array<double, 4>::iterator adj_ptr = adj.begin();
    Vec<double> x;
    using expr_t = std::decay_t<decltype(make_eq(x[1], x[0]))>;
    using expr2_t = std::decay_t<decltype(make_eq(x[1], ad::constant(2)))>;
    expr_t expr;
    expr2_t expr2;

    jacobian_fixture()
        : x({1., 2.})
        , expr(make_eq(x[1], x[0]))
        , expr2(make_eq(x[1], ad::constant(2)))
    {}
};

TEST_F(jacobian_fixture, jacobian_unpack)
{
    auto tup = std::make_tuple(expr);
    jacobian_unpack(adj_ptr, x, tup);
    EXPECT_DOUBLE_EQ(adj[0], 1.);
    EXPECT_DOUBLE_EQ(adj[1], 1.);
}

TEST_F(jacobian_fixture, jacobian_unpack_twice)
{
    auto tup = std::make_tuple(expr);
    auto tup2 = std::make_tuple(expr2);
    jacobian_unpack(adj_ptr, x, tup);
    jacobian_unpack(adj_ptr + 2, x, tup2);
    EXPECT_DOUBLE_EQ(adj[2], 0.);
    EXPECT_DOUBLE_EQ(adj[3], 1.);
}

TEST_F(jacobian_fixture, jacobian_one_lmda_exgen)
{
    double values[2] = {1., 2.};
    jacobian(values, values + 2, adj_ptr, make_exgen<double>(core::f_lmda_no_opt));
    EXPECT_DOUBLE_EQ(adj[0], 1.);
    EXPECT_DOUBLE_EQ(adj[1], 0.);
}

TEST_F(jacobian_fixture, jacobian_two_lmda_exgen)
{
    double values[2] = {1., 2.};
    jacobian(values, values + 2, 
            adj_ptr, make_exgen<double>(core::f_lmda_no_opt, core::f_lmda_no_opt));
    EXPECT_DOUBLE_EQ(adj[0], 1.);
    EXPECT_DOUBLE_EQ(adj[1], 0.);
    EXPECT_DOUBLE_EQ(adj[2], 1.);
    EXPECT_DOUBLE_EQ(adj[3], 0.);
}

TEST_F(jacobian_fixture, jacobian_one_lmda)
{
    double values[2] = {1., 2.};
    jacobian(values, values + 2, adj_ptr, core::f_lmda_no_opt);
    EXPECT_DOUBLE_EQ(adj[0], 1.);
    EXPECT_DOUBLE_EQ(adj[1], 0.);
}

TEST_F(jacobian_fixture, jacobian_two_lmda)
{
    double values[2] = {1., 2.};
    jacobian(values, values + 2, adj_ptr, core::f_lmda_no_opt, core::f_lmda_no_opt);
    EXPECT_DOUBLE_EQ(adj[0], 1.);
    EXPECT_DOUBLE_EQ(adj[1], 0.);
    EXPECT_DOUBLE_EQ(adj[2], 1.);
    EXPECT_DOUBLE_EQ(adj[3], 0.);
}

#ifdef USE_ARMA

TEST_F(jacobian_fixture, jacobian_two_lmda_arma)
{
    double values[2] = {1., 2.};
    arma::Mat<double> jacobi(2, 2);
    jacobian(jacobi, values, values + 2, 
            core::f_lmda_no_opt, core::f_lmda_no_opt);
    EXPECT_DOUBLE_EQ(jacobi(0,0), 1.);
    EXPECT_DOUBLE_EQ(jacobi(0,1), 0.);
    EXPECT_DOUBLE_EQ(jacobi(1,0), 1.);
    EXPECT_DOUBLE_EQ(jacobi(1,1), 0.);
}

#endif
        
} // namespace details
} // namespace ad
