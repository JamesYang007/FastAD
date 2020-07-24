#include "gtest/gtest.h"
#include <tuple>
#include <fastad_bits/unary.hpp>
#include <fastad_bits/eval.hpp>
#include "base_fixture.hpp"

namespace ad {

struct eval_fixture: base_fixture
{
protected:
    using unary_t = MockUnary;
    using scl_unary_t = core::UnaryNode<unary_t, scl_expr_view_t>;

    scl_unary_t scl_unary;
    std::tuple<scl_unary_t, scl_unary_t> scl_tup;

    std::vector<value_t> val_buf;

    eval_fixture()
        : base_fixture()
        , scl_unary(scl_expr)
        , scl_tup(scl_expr, scl_expr)
        , val_buf(100, 0)   // obscene amount of buffer
    {
        scl_unary.bind(val_buf.data());
        std::get<0>(scl_tup).bind(val_buf.data());
        std::get<1>(scl_tup).bind(val_buf.data());
    }
};

TEST_F(eval_fixture, evaluate)
{
    EXPECT_DOUBLE_EQ(evaluate(scl_unary), 2.*scl_expr.get());
}

TEST_F(eval_fixture, evaluate_adj)
{
    evaluate_adj(scl_unary);
    evaluate_adj(scl_unary); // second time SHOULD change adj
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(0,0), 4.);
}

TEST_F(eval_fixture, autodiff)
{
    EXPECT_DOUBLE_EQ(autodiff(scl_unary), 2.*scl_expr.get());
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(0,0), 2.);
}

TEST_F(eval_fixture, autodiff_tup)
{
    autodiff(scl_tup);
    autodiff(scl_tup);
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(0,0), 8.);
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(0,0), 8.);
}

} // namespace ad
