#include "gtest/gtest.h"
#include <array>
#include <fastad_bits/reverse/core/unary.hpp>
#include <testutil/base_fixture.hpp>

namespace ad {
namespace core {

struct unary_fixture : base_fixture
{
protected:
    using unary_t = MockUnary;
    using scl_unary_t = UnaryNode<unary_t, scl_expr_view_t>;
    using vec_unary_t = UnaryNode<unary_t, vec_expr_view_t>;
    using mat_unary_t = UnaryNode<unary_t, mat_expr_view_t>;
    using scl_scl_unary_t = UnaryNode<unary_t, scl_unary_t>;

    scl_unary_t scl_unary;
    vec_unary_t vec_unary;
    mat_unary_t mat_unary;
    scl_scl_unary_t scl_scl_unary;

    value_t seed = 3.14;

    std::vector<value_t> val_buf;

    unary_fixture()
        : base_fixture()
        , scl_unary(scl_expr)
        , vec_unary(vec_expr)
        , mat_unary(mat_expr)
        , scl_scl_unary(scl_unary)
        , val_buf(std::max(vec_size, mat_size), 0)
    {
        // IMPORTANT: bind value for unary nodes.
        // No two unary node expressions can be used in a single test.
        scl_unary.bind(val_buf.data());
        vec_unary.bind(val_buf.data());
        mat_unary.bind(val_buf.data());
        scl_scl_unary.bind(val_buf.data());
    }
};

TEST_F(unary_fixture, scl_feval) 
{
    EXPECT_DOUBLE_EQ(scl_unary.feval(), scl_expr.get() * 2);
}

TEST_F(unary_fixture, scl_beval) 
{
    scl_unary.beval(seed,0,0, util::beval_policy::single); // last two ignored
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(0,0), 2.*seed); 
}

TEST_F(unary_fixture, vec_feval) 
{
    auto& res = vec_unary.feval();
    for (size_t i = 0; i < vec_size; ++i) {
        EXPECT_DOUBLE_EQ(res(i), vec_expr.get()(i) * 2);
    }
}

TEST_F(unary_fixture, vec_beval) 
{
    vec_unary.beval(seed, 1,0, util::beval_policy::single); // last param ignored
    vec_unary.beval(seed, 3,0, util::beval_policy::single); // last param ignored
    for (size_t i = 0; i < vec_size; ++i) {
        if (i == 1 || i == 3) {
            EXPECT_DOUBLE_EQ(vec_expr.get_adj(i,0), 2.*seed); 
        } else  {
            EXPECT_DOUBLE_EQ(vec_expr.get_adj(i,0), 0.); 
        }
    }
}

TEST_F(unary_fixture, mat_feval) 
{
    auto& res = mat_unary.feval();
    for (size_t i = 0; i < mat_rows; ++i) {
        for (size_t j = 0; j < mat_cols; ++j) {
            EXPECT_DOUBLE_EQ(res(i,j), mat_expr.get()(i,j) * 2);
        }
    }
}

TEST_F(unary_fixture, mat_beval) 
{
    mat_unary.beval(seed, 1, 2, util::beval_policy::single);
    mat_unary.beval(seed, 0, 1, util::beval_policy::single);
    mat_unary.beval(seed, 1, 0, util::beval_policy::single);
    for (size_t i = 0; i < mat_rows; ++i) {
        for (size_t j = 0; j < mat_cols; ++j) {
            if ((i == 1 && j == 2) ||
                (i == 0 && j == 1) ||
                (i == 1 && j == 0)) {
                EXPECT_DOUBLE_EQ(mat_expr.get_adj(i,j), 2.*seed); 
            } else  {
                EXPECT_DOUBLE_EQ(mat_expr.get_adj(i,j), 0.); 
            }
        }
    }
}

TEST_F(unary_fixture, scl_scl_feval) 
{
    EXPECT_DOUBLE_EQ(scl_scl_unary.feval(), scl_expr.get() * 4);
}

TEST_F(unary_fixture, scl_scl_beval) 
{
    scl_scl_unary.beval(seed,0,0, util::beval_policy::single); // last two ignored
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(0,0), 4.*seed); 
}

} // namespace core
} // namespace ad
