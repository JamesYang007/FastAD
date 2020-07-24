#include "gtest/gtest.h"
#include <fastad_bits/unary.hpp>
#include <fastad_bits/eq.hpp>
#include "base_fixture.hpp"

namespace ad {
namespace core {

struct eq_fixture : base_fixture
{
protected:
    using unary_t = MockUnary;
    using scl_unary_t = UnaryNode<unary_t, scl_expr_view_t>;
    using vec_unary_t = UnaryNode<unary_t, vec_expr_view_t>;
    using mat_unary_t = UnaryNode<unary_t, mat_expr_view_t>;
    using scl_scl_unary_t = UnaryNode<unary_t, scl_unary_t>;
    using vec_vec_unary_t = UnaryNode<unary_t, vec_unary_t>;
    using mat_mat_unary_t = UnaryNode<unary_t, mat_unary_t>;
    using scl_eq_t = EqNode<scl_expr_view_t, scl_scl_unary_t>;
    using vec_eq_t = EqNode<vec_expr_view_t, vec_vec_unary_t>;
    using mat_eq_t = EqNode<mat_expr_view_t, mat_mat_unary_t>;

    scl_expr_t scl_place;
    vec_expr_t vec_place;
    mat_expr_t mat_place;
    scl_eq_t scl_eq;
    vec_eq_t vec_eq;
    mat_eq_t mat_eq;

    value_t seed = 3.14;

    std::vector<value_t> val_buf;

    eq_fixture()
        : base_fixture()
        , scl_place()
        , vec_place(vec_size)
        , mat_place(mat_rows, mat_cols)
        , scl_eq(scl_place, {scl_expr})
        , vec_eq(vec_place, {vec_expr})
        , mat_eq(mat_place, {mat_expr})
        , val_buf(std::max(vec_size, mat_size), 0)
    {
        // IMPORTANT: bind value for unary nodes.
        // No two unary node expressions can be used in a single test.
        scl_eq.bind(val_buf.data());
        vec_eq.bind(val_buf.data());
        mat_eq.bind(val_buf.data());
    }
};

TEST_F(eq_fixture, scl_feval)
{
    value_t res = scl_eq.feval();
    EXPECT_DOUBLE_EQ(res, 4.*scl_expr.get());

    // check that placeholder value has been modified
    EXPECT_DOUBLE_EQ(scl_place.get(), 4.*scl_expr.get());
}

TEST_F(eq_fixture, scl_beval)
{
    scl_eq.beval(seed, 0,0, util::beval_policy::single);    // last two ignored
    EXPECT_DOUBLE_EQ(scl_place.get_adj(0,0), seed);
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(0,0), 4.*seed);
}

TEST_F(eq_fixture, vec_feval)
{
    Eigen::VectorXd res = vec_eq.feval();
    for (int i = 0; i < res.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), 4.*vec_expr.get(i,0));

        // check that placeholder value has been modified
        EXPECT_DOUBLE_EQ(vec_place.get(i,0), 4.*vec_expr.get(i,0));
    }
}

TEST_F(eq_fixture, vec_beval)
{
    vec_eq.beval(seed, 2,0, util::beval_policy::single);    // last ignored
    for (size_t i = 0; i < vec_size; ++i) {
        value_t actual = (i == 2) ? seed : 0;
        EXPECT_DOUBLE_EQ(vec_place.get_adj(i,0), actual);
        EXPECT_DOUBLE_EQ(vec_expr.get_adj(i,0), 4.*actual);
    }
}

TEST_F(eq_fixture, mat_feval)
{
    Eigen::MatrixXd res = mat_eq.feval();
    for (int i = 0; i < res.rows(); ++i) {
        for (int j = 0; j < res.cols(); ++j) {
            EXPECT_DOUBLE_EQ(res(i,j), 4.*mat_expr.get(i,j));

            // check that placeholder value has been modified
            EXPECT_DOUBLE_EQ(mat_place.get(i,j), 4.*mat_expr.get(i,j));
        }
    }
}

TEST_F(eq_fixture, mat_beval)
{
    mat_eq.beval(seed,1,1, util::beval_policy::single);
    mat_eq.beval(seed,0,2, util::beval_policy::single);
    for (size_t i = 0; i < mat_rows; ++i) {
        for (size_t j = 0; j < mat_cols; ++j) {
            value_t actual = ((i == 1 && j == 1) ||
                              (i == 0 && j == 2)) ? seed : 0;
            EXPECT_DOUBLE_EQ(mat_place.get_adj(i,j), actual);
            EXPECT_DOUBLE_EQ(mat_expr.get_adj(i,j), 4.*actual);
        }
    }
}

TEST_F(eq_fixture, vec_nested_eq_feval)
{
    Var<value_t, ad::vec> y(vec_size);
    auto expr = (y = vec_eq);
    expr.bind(val_buf.data());

    Eigen::VectorXd res = expr.feval();
    for (int i = 0; i < res.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), 4.*vec_expr.get(i,0));
        // check that placeholder value has been modified
        EXPECT_DOUBLE_EQ(vec_place.get(i,0), 4.*vec_expr.get(i,0));
        EXPECT_DOUBLE_EQ(y.get(i,0), 4.*vec_expr.get(i,0));
    }
}

TEST_F(eq_fixture, vec_nested_eq_beval)
{
    Var<value_t, ad::vec> y(vec_size);
    auto expr = (y = vec_eq);
    expr.bind(val_buf.data());

    expr.beval(seed, 2, 0, util::beval_policy::single);
    expr.beval(seed, 0, 0, util::beval_policy::single);
    for (size_t i = 0; i < vec_size; ++i) {
        value_t actual = (i == 0 || i == 2) ? seed : 0;
        EXPECT_DOUBLE_EQ(y.get_adj(i,0), actual);
        EXPECT_DOUBLE_EQ(vec_place.get_adj(i,0), actual);
        EXPECT_DOUBLE_EQ(vec_expr.get_adj(i,0), actual * 4.);
    }
}

} // namespace core
} // namespace ad
