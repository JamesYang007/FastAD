#include "gtest/gtest.h"
#include <testutil/base_fixture.hpp>
#include <fastad_bits/reverse/core/unary.hpp>
#include <fastad_bits/reverse/core/eq.hpp>
#include <fastad_bits/reverse/core/glue.hpp>

namespace ad {
namespace core {

struct glue_fixture : base_fixture
{
protected:
    using unary_t = MockUnary;
    using scl_unary_t = UnaryNode<unary_t, scl_expr_view_t>;
    using vec_unary_t = UnaryNode<unary_t, vec_expr_view_t>;
    using mat_unary_t = UnaryNode<unary_t, mat_expr_view_t>;
    using scl_eq_t = EqNode<scl_expr_view_t, scl_unary_t>;
    using vec_eq_t = EqNode<vec_expr_view_t, vec_unary_t>;
    using mat_eq_t = EqNode<mat_expr_view_t, mat_unary_t>;
    using scl_glue_t = GlueNode<scl_eq_t, scl_unary_t>;
    using vec_glue_t = GlueNode<vec_eq_t, vec_unary_t>;
    using mat_glue_t = GlueNode<mat_eq_t, mat_unary_t>;

    scl_expr_t scl_place;
    vec_expr_t vec_place;
    mat_expr_t mat_place;

    scl_glue_t scl_glue;
    vec_glue_t vec_glue;
    mat_glue_t mat_glue;

    value_t seed = 3.14;
    Eigen::ArrayXd vseed;
    Eigen::ArrayXXd mseed;

    glue_fixture()
        : base_fixture()
        , scl_place()
        , vec_place(vec_size)
        , mat_place(mat_rows, mat_cols)
        , scl_glue({scl_place, scl_expr}, scl_place)
        , vec_glue({vec_place, vec_expr}, vec_place)
        , mat_glue({mat_place, mat_expr}, mat_place)
        , vseed(vec_size)
        , mseed(mat_rows, mat_cols)
    {
        vseed << 2.3, 1.4, -2.3, 0.3, 1.3;
        mseed << 1.32, 4.24, 1.644, 
                -0.23, 23.1, 4.24;

        auto size_pack = vec_glue.bind_cache_size();
        size_pack = size_pack.max(mat_glue.bind_cache_size());
        val_buf.resize(size_pack(0));
        adj_buf.resize(size_pack(1));

        // IMPORTANT: bind value for unary nodes.
        // No two unary node expressions can be used in a single test.
        ptr_pack_t ptr_pack(val_buf.data(), adj_buf.data());
        scl_glue.bind_cache(ptr_pack);
        vec_glue.bind_cache(ptr_pack);
        mat_glue.bind_cache(ptr_pack);
    }
};

TEST_F(glue_fixture, scl_feval)
{
    value_t res = scl_glue.feval();
    EXPECT_DOUBLE_EQ(res, 4.*scl_expr.get());
    // check that placeholder value has been modified
    EXPECT_DOUBLE_EQ(scl_place.get(), 2.*scl_expr.get());
}

TEST_F(glue_fixture, scl_beval)
{
    scl_glue.beval(seed);
    EXPECT_DOUBLE_EQ(scl_place.get_adj(), 2.*seed);
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(), 4.*seed);
}

TEST_F(glue_fixture, vec_feval)
{
    Eigen::VectorXd res = vec_glue.feval();
    Eigen::VectorXd actual_p = 2 * vec_expr.get();
    check_eq(res, 2 * actual_p);
    check_eq(vec_place.get(), actual_p);
}

TEST_F(glue_fixture, vec_beval)
{
    vec_glue.feval();
    vec_glue.beval(vseed);
    check_eq(vec_place.get_adj(), 2 * vseed);
    check_eq(vec_expr.get_adj(), 4 * vseed);
}

TEST_F(glue_fixture, mat_feval)
{
    Eigen::MatrixXd res = mat_glue.feval();
    check_eq(res, 4 * mat_expr.get());
    check_eq(mat_place.get(), 2 * mat_expr.get());
}

TEST_F(glue_fixture, mat_beval)
{
    mat_glue.beval(mseed);
    check_eq(mat_place.get_adj(), 2 * mseed);
    check_eq(mat_expr.get_adj(), 4 * mseed);
}

} // namespace core
} // namespace ad
