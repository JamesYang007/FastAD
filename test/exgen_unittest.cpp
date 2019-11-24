#include <fastad_bits/exgen.hpp>
#include "exgen_fixture.hpp"
#include "gtest/gtest.h"

namespace ad {
namespace core {

struct exgen_fixture: ::testing::Test
{
protected:
    Exgen<double, decltype(f_lmda_no_opt)> gen_no_opt;
    Exgen<double, decltype(f_lmda_opt)> gen_opt;
    Exgen<double, decltype(f_lmda_no_opt), decltype(f_lmda_no_opt)> mult_gen_no_opt;    // both no optimization needed
    Exgen<double, decltype(f_lmda_opt), decltype(f_lmda_opt)> mult_gen_opt; // both optimization needed
    Exgen<double, decltype(f_lmda_opt), decltype(f_lmda_no_opt)> mult_gen; // one needs optimization, one does not
    Vec<double> x;  
    double seed = 3.2;

    exgen_fixture()
        : gen_no_opt(f_lmda_no_opt)
        , gen_opt(f_lmda_opt)
        , mult_gen_no_opt(f_lmda_no_opt, f_lmda_no_opt)
        , mult_gen_opt(f_lmda_opt, f_lmda_opt)
        , mult_gen(f_lmda_opt, f_lmda_no_opt)
        , x({1., 2.})
    {}

    template <class Expr>
    void check_expr(Expr& expr, double adjoint) 
    {
        EXPECT_DOUBLE_EQ(expr.feval(), 1.);
        expr.beval(seed);
        EXPECT_DOUBLE_EQ(x[0].get_adjoint(), adjoint);
    }

};

// Lmda function returns non-glue node or eq-node
// with optimization size specified
TEST_F(exgen_fixture, univariate_generate_no_glue)
{
    auto lmda = [](const Vec<double>& x, const Vec<double>&) {
        return make_eq(x[1], x[0]);
    };
    auto gen = make_exgen<double>(lmda);
    auto expr = std::get<0>(gen.template generate<0>(x));
    check_expr(expr, seed);
}

// Should not lead to memory error
TEST_F(exgen_fixture, univariate_generate_no_glue_memcheck)
{
    auto lmda = [](const Vec<double>& x, const Vec<double>&) {
        return make_eq(x[1], x[0]);
    };
    auto gen = make_exgen<double>(lmda);
    auto expr = std::get<0>(gen.template generate<0>(x));
    // gen contains placeholder of size 1
    auto gen_mv = std::move(gen);  // move gen to gen_mv
    // now gen_mv has placeholder of size 1, but should not point to anything in gen
    expr = std::get<0>(gen_mv.template generate<0>(x));
    check_expr(expr, seed);
}

TEST_F(exgen_fixture, univariate_generate_no_opt)
{
    auto expr = std::get<0>(gen_no_opt.generate(x));
    check_expr(expr, seed);
}

TEST_F(exgen_fixture, univariate_generate_opt)
{
    auto expr = std::get<0>(gen_opt.template generate<3>(x));
    check_expr(expr, seed);
}

TEST_F(exgen_fixture, multivariate_generate_no_opt)
{
    auto tup = mult_gen_no_opt.generate(x);
    static_assert(std::tuple_size<decltype(tup)>::value == 2);

    auto expr1 = std::get<0>(tup);
    auto&& expr2 = std::get<1>(tup);

    check_expr(expr1, seed);
    // backwards evaluation without resetting variable adjoints
    // leads to directional derivative in the direction of (seed1,..., seedn)
    // in this example, the direction is (seed, seed)
    check_expr(expr2, 2 * seed);
}

TEST_F(exgen_fixture, multivariate_generate_opt)
{
    auto tup = mult_gen_opt.template generate<3, 3>(x);
    static_assert(std::tuple_size<decltype(tup)>::value == 2);

    auto expr1 = std::get<0>(tup);
    auto&& expr2 = std::get<1>(tup);

    check_expr(expr1, seed);
    check_expr(expr2, 2 * seed);
}

TEST_F(exgen_fixture, multivariate_generate)
{
    // must specify both optimization parameters (3)
    auto tup = mult_gen_opt.template generate<3, 3>(x);
    static_assert(std::tuple_size<decltype(tup)>::value == 2);

    auto expr1 = std::get<0>(tup);
    auto&& expr2 = std::get<1>(tup);

    check_expr(expr1, seed);
    check_expr(expr2, 2 * seed);
}

TEST_F(exgen_fixture, make_exgen)
{
    // template parameter is placeholder value type
    auto gen = make_exgen<double>(f_lmda_no_opt);
    auto tup = gen.generate(x);
    auto expr1 = std::get<0>(tup);
    check_expr(expr1, seed);
}

TEST_F(exgen_fixture, make_exgen_large)
{
    Vec<double> x(1000);  
    for (auto& var : x) {
        var.set_value(1.);
    }

    // template parameter is placeholder value type
    auto gen = make_exgen<double>(f_lmda_no_opt);
    auto tup = gen.generate(x);
    auto expr1 = std::get<0>(tup);
    expr1.feval();
    expr1.beval(seed);
}

} // namespace core
} // namespace ad
