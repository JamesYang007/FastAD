#include "gtest/gtest.h"
#include <testutil/base_fixture.hpp>
#include <fastad_bits/reverse/core/eq.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/reverse/core/for_each.hpp>

namespace ad {
namespace core {

struct for_each_fixture : base_fixture
{
protected:
    using scl_eq_t = OpEqNode<AddEq, scl_expr_view_t, Constant<value_t, scl>>;
    using scl_for_each_t = ForEachIterNode<std::vector<scl_eq_t>>;

    scl_for_each_t scl_for_each;
    value_t seed = 3.14;

    for_each_fixture()
        : base_fixture()
        , scl_for_each({{scl_expr, 1.}, {scl_expr, 1.}})
    {
        auto size_pack = scl_for_each.bind_cache_size();
        val_buf.resize(size_pack(0));
        adj_buf.resize(size_pack(1));
        ptr_pack_t ptr_pack(val_buf.data(), adj_buf.data());
        scl_for_each.bind_cache(ptr_pack);
    }
};

TEST_F(for_each_fixture, scl_feval)
{
    value_t orig = scl_expr.get();
    value_t res = scl_for_each.feval();
    EXPECT_DOUBLE_EQ(res, orig + 2.);
    EXPECT_DOUBLE_EQ(scl_expr.get(), res);
}

TEST_F(for_each_fixture, scl_beval)
{
    scl_for_each.beval(seed);
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(), seed);
}

} // namespace core
} // namespace ad
