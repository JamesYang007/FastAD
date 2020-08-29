#include "gtest/gtest.h"
#include <testutil/base_fixture.hpp>
#include <fastad_bits/reverse/core/unary.hpp>
#include <fastad_bits/reverse/core/eq.hpp>

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
    Eigen::ArrayXd vseed;
    Eigen::ArrayXXd mseed;

    eq_fixture()
        : base_fixture()
        , scl_place()
        , vec_place(vec_size)
        , mat_place(mat_rows, mat_cols)
        , scl_eq(scl_place, {scl_expr})
        , vec_eq(vec_place, {vec_expr})
        , mat_eq(mat_place, {mat_expr})
        , vseed(vec_size)
        , mseed(mat_rows, mat_cols)
    {
        vseed << 2.3, 1.4, -2.3, 0.3, 1.3;
        mseed << 1.32, 4.24, 1.644, 
                -0.23, 23.1, 4.24;

        auto size_pack = vec_eq.bind_cache_size();
        size_pack = size_pack.max(mat_eq.bind_cache_size());
        val_buf.resize(size_pack(0));
        adj_buf.resize(size_pack(1));

        // IMPORTANT: bind value for unary nodes.
        // No two unary node expressions can be used in a single test.
        ptr_pack_t ptr_pack(val_buf.data(), adj_buf.data());
        scl_eq.bind_cache(ptr_pack);
        vec_eq.bind_cache(ptr_pack);
        mat_eq.bind_cache(ptr_pack);
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
    scl_eq.beval(seed);
    EXPECT_DOUBLE_EQ(scl_place.get_adj(), seed);
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(), 4.*seed);
}

TEST_F(eq_fixture, vec_feval)
{
    Eigen::VectorXd res = vec_eq.feval();
    Eigen::VectorXd actual = 4 * vec_expr.get();
    check_eq(res, actual);
    // check that placeholder value has been modified
    check_eq(vec_place.get(), actual);
}

TEST_F(eq_fixture, vec_beval)
{
    vec_eq.feval();
    vec_eq.beval(seed);
    Eigen::VectorXd actual = seed * Eigen::VectorXd::Ones(vec_size);
    check_eq(actual, vec_place.get_adj());
    check_eq(4 * actual, vec_expr.get_adj());
}

TEST_F(eq_fixture, mat_feval)
{
    Eigen::MatrixXd res = mat_eq.feval();
    Eigen::MatrixXd actual = 4. * mat_expr.get();
    check_eq(res, actual);
    check_eq(mat_place.get(), actual);
}

TEST_F(eq_fixture, mat_beval)
{
    mat_eq.beval(mseed);
    Eigen::MatrixXd actual_p = mseed;
    Eigen::MatrixXd actual = 4 * mseed;
    check_eq(mat_place.get_adj(), actual_p);
    check_eq(mat_expr.get_adj(), actual);
}

TEST_F(eq_fixture, vec_nested_eq_feval)
{
    Var<value_t, ad::vec> y(vec_size);
    auto expr = (y = vec_eq);
    this->bind(expr);
    Eigen::VectorXd res = expr.feval();

    Eigen::VectorXd actual = 4. * vec_expr.get();
    check_eq(res, actual);
    check_eq(vec_place.get(), actual);
    check_eq(y.get(), actual);
}

TEST_F(eq_fixture, vec_nested_eq_beval)
{
    Var<value_t, ad::vec> y(vec_size);
    auto expr = (y = vec_eq);
    this->bind(expr);
    expr.feval();
    expr.beval(vseed);
    check_eq(y.get_adj(), vseed);
    check_eq(vec_place.get_adj(), vseed);
    check_eq(vec_expr.get_adj(), 4. * vseed);
}

///////////////////////////////////////
// OpEqNode Tests
///////////////////////////////////////

TEST_F(eq_fixture, opeq_scl_feval_alias)
{
    auto expr = (scl_expr *= scl_expr); 
    this->bind(expr);
    value_t orig = scl_expr.get();
    value_t res = expr.feval();
    EXPECT_DOUBLE_EQ(res, orig * orig);
}

TEST_F(eq_fixture, opeq_scl_beval_alias)
{
    auto expr = (scl_expr *= scl_expr); 
    this->bind(expr);
    double orig = scl_expr.get();
    expr.feval();
    expr.beval(seed);
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(), seed*2*orig);
}

TEST_F(eq_fixture, opeq_scl_feval_noalias)
{
    auto expr = (scl_expr /= scl_unary_t(scl_expr)); 
    this->bind(expr);
    double orig = scl_expr.get();
    double res = expr.feval();
    EXPECT_DOUBLE_EQ(res, orig / unary_t::fmap(orig));
}

TEST_F(eq_fixture, opeq_scl_beval_noalias)
{
    auto expr = (scl_expr /= scl_unary_t(scl_expr)); 
    this->bind(expr);
    expr.feval();
    expr.beval(seed);
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(), 0.);
}

TEST_F(eq_fixture, opeq_vec_scl_feval_noalias)
{
    auto expr = (vec_expr *= scl_expr); 
    this->bind(expr);
    Eigen::VectorXd orig = vec_expr.get();
    Eigen::VectorXd res = expr.feval();
    Eigen::VectorXd actual = orig * scl_expr.get();
    check_eq(res, actual);
}

TEST_F(eq_fixture, opeq_vec_scl_beval_noalias)
{
    auto expr = (vec_expr *= scl_expr); 
    this->bind(expr);
    Eigen::VectorXd orig = vec_expr.get();
    expr.feval();
    expr.beval(vseed);
    check_eq(vec_expr.get_adj(), vseed * scl_expr.get());
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(), (vseed * orig.array()).sum());
}

TEST_F(eq_fixture, opeq_vec_vec_feval_alias)
{
    auto expr = (vec_expr *= vec_expr); 
    this->bind(expr);
    Eigen::VectorXd orig = vec_expr.get();
    Eigen::VectorXd res = expr.feval();
    check_eq(res, orig.array() * orig.array());
}

TEST_F(eq_fixture, opeq_vec_vec_beval_alias)
{
    auto expr = (vec_expr *= vec_expr); 
    this->bind(expr);
    Eigen::VectorXd orig = vec_expr.get();
    expr.feval();
    expr.beval(vseed);
    check_eq(vec_expr.get_adj(), vseed * 2 * orig.array());
}

TEST_F(eq_fixture, opeq_vec_vec_feval_noalias)
{
    auto expr = (vec_expr *= vec_unary_t(vec_expr)); 
    this->bind(expr);
    Eigen::VectorXd orig = vec_expr.get();
    Eigen::VectorXd res = expr.feval();
    check_eq(res, orig.array() * unary_t::fmap(orig.array()));
}

TEST_F(eq_fixture, opeq_vec_vec_beval_noalias)
{
    auto expr = (vec_expr *= vec_unary_t(vec_expr)); 
    this->bind(expr);
    Eigen::VectorXd orig = vec_expr.get();
    expr.feval();
    expr.beval(vseed);
    Eigen::VectorXd actual = 
        vseed * (unary_t::fmap(orig.array()) + 
                 unary_t::bmap(orig.array(), 0, 0));
    check_eq(vec_expr.get_adj(), actual);
}

} // namespace core
} // namespace ad
