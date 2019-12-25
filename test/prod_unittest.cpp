#include <fastad_bits/prod.hpp>
#include "gtest/gtest.h"

namespace ad {
namespace core {

struct prod_fixture: ::testing::Test
{
protected:
    Var<double> leaves[3] = { 1., 2., 3. };
    double seed = 3.;

    using lmda_t = std::function<
        decltype(ad::sin(std::declval<Var<double>>()))(const Var<double>&)
        >;
    lmda_t lmda = 
        [](const Var<double>& expr) {
            return ad::sin(expr);
        };

    // bad practice and dangerous to write lambda function like this, 
    // but product should be robust enough to handle this
    using leaflmda_t = std::function<Var<double>(const Var<double>&)>;
    leaflmda_t leaflmda = 
        [](const Var<double>& expr) {
            Var<double> tmp(expr);
            tmp.set_value(tmp.get_value() * 2); 
            return tmp;
        };
};

TEST_F(prod_fixture, prodnode_feval_one)
{
    auto&& expr = ad::prod(leaves, leaves + 1, lmda);
    double expected = std::sin(1);
    EXPECT_DOUBLE_EQ(expr.feval(), expected);
}

TEST_F(prod_fixture, prodnode_beval_one)
{
    auto&& expr = ad::prod(leaves, leaves + 1, lmda);
    expr.beval(seed);
    double expected = seed * std::cos(1);
    EXPECT_DOUBLE_EQ(leaves[0].get_adjoint(), expected);
}

TEST_F(prod_fixture, prodnode_feval) 
{
    auto&& expr = ad::prod(leaves, leaves + 3, lmda);
    double expected = std::sin(1) * std::sin(2) * std::sin(3);
    EXPECT_DOUBLE_EQ(expr.feval(), expected);
}

TEST_F(prod_fixture, prodnode_beval) 
{
    auto&& expr = ad::prod(leaves, leaves + 3, lmda);
    expr.beval(seed);
    EXPECT_DOUBLE_EQ(expr.get_adjoint(), seed);         // expr adjoint set to seed
    EXPECT_DOUBLE_EQ(leaves[2].get_adjoint(),           // last expression adjoint set to seed
            seed * std::sin(leaves[0].get_value()) * std::sin(leaves[1].get_value()) * std::cos(leaves[2].get_value()));
    EXPECT_DOUBLE_EQ(leaves[1].get_adjoint(), 
            seed * std::sin(leaves[0].get_value()) * std::cos(leaves[1].get_value()) * std::sin(leaves[2].get_value()));
    EXPECT_DOUBLE_EQ(leaves[0].get_adjoint(), 
            seed * std::cos(leaves[0].get_value()) * std::sin(leaves[1].get_value()) * std::sin(leaves[2].get_value()));
}

TEST_F(prod_fixture, prodnode_leaflmda_feval) 
{
    auto&& expr = ad::prod(leaves, leaves + 3, leaflmda);
    double expected = 48.;
    EXPECT_DOUBLE_EQ(expr.feval(), expected);
}

TEST_F(prod_fixture, prodnode_leaflmda_beval) 
{
    auto&& expr = ad::prod(leaves, leaves + 3, leaflmda);
    expr.beval(seed);
    EXPECT_DOUBLE_EQ(expr.get_adjoint(), seed);         // expr adjoint set to seed
    EXPECT_DOUBLE_EQ(leaves[2].get_adjoint(),           // last expression adjoint set to seed
            seed * leaves[0].get_value() * leaves[1].get_value());
    EXPECT_DOUBLE_EQ(leaves[1].get_adjoint(), 
            seed * leaves[0].get_value() * leaves[2].get_value());
    EXPECT_DOUBLE_EQ(leaves[0].get_adjoint(), 
            seed * leaves[1].get_value() * leaves[2].get_value());
}

} // namespace core
} // namespace ad
