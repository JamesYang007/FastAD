#include "base_fixture.hpp"
#include <fastad_bits/dot.hpp>
#include <fastad_bits/unary.hpp>

namespace ad {
namespace core {

struct dot_fixture: base_fixture
{
protected:
    using unary_t = MockUnary;
    using vec_unary_t = UnaryNode<unary_t, vec_expr_view_t>;
    using mat_unary_t = UnaryNode<unary_t, mat_expr_view_t>;
    using dot_vars_t = MatVecDotNode<mat_expr_view_t, 
                                     vec_expr_view_t>;
    using dot_unary_t = MatVecDotNode<mat_unary_t,
                                      vec_unary_t>;

    dot_vars_t dot_vars;
    dot_unary_t dot_unary;

    value_t seed = 0.010332;

    dot_fixture()
        : base_fixture(3) // change vec_size to 3
        , dot_vars(mat_expr, vec_expr)
        , dot_unary({mat_expr}, {vec_expr})
    {
        // initialize vector since not default size
        vec_initialize();

        // resize to whatever is the max memory needed
        val_buf.resize(dot_vars.bind_size());
        val_buf.resize(dot_unary.bind_size());

        // bind all expressions to buffer
        dot_vars.bind(val_buf.data());
        dot_unary.bind(val_buf.data());
    }

    void vec_initialize() 
    {
        auto vec_raw = vec_expr.get();
        vec_raw(0) = 0.312332;
        vec_raw(1) = -3.213;
        vec_raw(2) = 9.135;
    }
};

TEST_F(dot_fixture, dot_vars_feval)
{
    Eigen::VectorXd actual = dot_vars.feval();
    Eigen::VectorXd expected = mat_expr.get() * vec_expr.get();
    for (int i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(actual(i), expected(i));
    }
}

TEST_F(dot_fixture, dot_vars_beval)
{
    dot_vars.feval();
    dot_vars.beval(seed, 0, 0, util::beval_policy::single);

    for (size_t j = 0; j < vec_expr.size(); ++j) {
        EXPECT_DOUBLE_EQ(vec_expr.get_adj(j,0), seed*mat_expr.get(0,j));
        EXPECT_DOUBLE_EQ(mat_expr.get_adj(0,j), seed*vec_expr.get(j,0));
    }

    // all other matrix adjoints should be 0
    for (size_t i = 1; i < mat_rows; ++i) {
        for (size_t j = 0; j < mat_cols; ++j) {
            EXPECT_DOUBLE_EQ(mat_expr.get_adj(i,j), 0.);
        }
    }
}

TEST_F(dot_fixture, dot_unary_feval)
{
    Eigen::VectorXd actual = dot_unary.feval();
    Eigen::VectorXd expected = 
        unary_t::fmap(mat_expr.get().array()).matrix() * 
        unary_t::fmap(vec_expr.get().array()).matrix();
    for (int i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(actual(i), expected(i));
    }
}

TEST_F(dot_fixture, dot_unary_beval)
{
    dot_unary.feval();
    dot_unary.beval(seed, 1, 0, util::beval_policy::single);

    for (size_t j = 0; j < vec_expr.size(); ++j) {
        EXPECT_DOUBLE_EQ(vec_expr.get_adj(j,0), 
                seed * 2. * unary_t::fmap(mat_expr.get(1,j)));
        EXPECT_DOUBLE_EQ(mat_expr.get_adj(1,j), 
                seed * 2. * unary_t::fmap(vec_expr.get(j,0)));
    }

    // all other matrix adjoints should be 0
    for (size_t i = 0; i < mat_rows; ++i) {
        if (i == 1) continue;
        for (size_t j = 0; j < mat_cols; ++j) {
            EXPECT_DOUBLE_EQ(mat_expr.get_adj(i,j), 0.);
        }
    }
}

TEST_F(dot_fixture, dot_constant_var_feval)
{
    auto mat_const = ad::constant(Eigen::MatrixXd(mat_expr.get()));
    auto dot_constant_var = ad::dot(mat_const, vec_expr);

    this->bind(dot_constant_var);

    Eigen::VectorXd actual = dot_constant_var.feval();
    Eigen::VectorXd expected = mat_expr.get() * vec_expr.get();
    for (int i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(actual(i), expected(i));
    }
}

TEST_F(dot_fixture, dot_constant_var_beval)
{
    auto mat_const = ad::constant(Eigen::MatrixXd(mat_expr.get()));
    auto dot_constant_var = ad::dot(mat_const, vec_expr);

    this->bind(dot_constant_var);

    dot_constant_var.feval();
    dot_constant_var.beval(seed, 0, 0, util::beval_policy::single);

    for (size_t j = 0; j < vec_expr.size(); ++j) {
        EXPECT_DOUBLE_EQ(vec_expr.get_adj(j,0), seed*mat_const.get(0,j));
    }
}

TEST_F(dot_fixture, dot_constant)
{
    auto mat_const = ad::constant(Eigen::MatrixXd(mat_expr.get()));
    auto vec_const = ad::constant(Eigen::VectorXd(vec_expr.get()));

    auto dot_constant = ad::dot(mat_const, vec_const);

    static_assert(std::is_same_v<
            std::decay_t<decltype(dot_constant)>,
            Constant<value_t, ad::vec> >);

    Eigen::VectorXd actual = dot_constant.feval();
    Eigen::VectorXd expected = mat_expr.get() * vec_expr.get();
    for (int i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(actual(i), expected(i));
    }
}

} // namespace core
} // namespace ad
