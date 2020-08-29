#include "gtest/gtest.h"
#include <tuple>
#include <fastad_bits/reverse/core/unary.hpp>
#include <fastad_bits/reverse/core/eval.hpp>
#include <testutil/base_fixture.hpp>

namespace ad {

struct eval_fixture: base_fixture
{
protected:
    using unary_t = MockUnary;
    using scl_unary_t = core::UnaryNode<unary_t, scl_expr_view_t>;

    scl_unary_t scl_unary;
    std::tuple<scl_unary_t, scl_unary_t> scl_tup;

    eval_fixture()
        : base_fixture()
        , scl_unary(scl_expr)
        , scl_tup(scl_expr, scl_expr)
    {
        val_buf.resize(100);   // obscene amount of buffer
        adj_buf.resize(100);   // obscene amount of buffer
        ptr_pack_t ptr_pack(val_buf.data(), adj_buf.data());
        scl_unary.bind_cache(ptr_pack);
        std::get<0>(scl_tup).bind_cache(ptr_pack);
        std::get<1>(scl_tup).bind_cache(ptr_pack);
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

} // namespace ad
