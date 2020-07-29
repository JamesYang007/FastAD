#include "gtest/gtest.h"
#include <fastad_bits/reverse/core/unary.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/reverse/core/prod.hpp>
#include <testutil/base_fixture.hpp>

namespace ad {
namespace core {

struct prod_fixture : base_fixture
{
protected:
    using unary_t = MockUnary;
    using scl_unary_t = UnaryNode<unary_t, scl_expr_view_t>;
    using vec_unary_t = UnaryNode<unary_t, vec_expr_view_t>;
    using mat_unary_t = UnaryNode<unary_t, mat_expr_view_t>;

    size_t size = 5;
    value_t seed = 3.14;

    std::vector<scl_expr_t> scl_exprs;
    std::vector<vec_expr_t> vec_exprs;
    std::vector<mat_expr_t> mat_exprs;
    std::vector<value_t> val_buf;

    prod_fixture()
        : base_fixture()
        , scl_exprs(size)
        , vec_exprs(size, vec_expr_t{vec_size})
        , mat_exprs(size, mat_expr_t{mat_rows, mat_cols})
        , val_buf((size+1)*std::max(vec_size, mat_size), 0)
    {
        for (auto& expr : scl_exprs) this->scl_initialize(expr);
        for (auto& expr : vec_exprs) this->vec_initialize(expr);
        for (auto& expr : mat_exprs) this->mat_initialize(expr);
    }

    // Creates a prod node depending on UnaryType (each expression)
    // Already binds the node to val_buf.
    template <class ShapeType>
    auto make_prod()
    {
        if constexpr (std::is_same_v<ShapeType, ad::scl>) {
            auto prodnode = prod(scl_exprs.begin(), scl_exprs.end(), 
                               [](const auto& x) { return scl_unary_t(x); });
            prodnode.bind(val_buf.data());
            return prodnode;
        } else if constexpr (std::is_same_v<ShapeType, ad::vec>) {
            auto prodnode = prod(vec_exprs.begin(), vec_exprs.end(), 
                               [](const auto& x) { return vec_unary_t(x); });
            prodnode.bind(val_buf.data());
            return prodnode;
        } else {
            auto prodnode = prod(mat_exprs.begin(), mat_exprs.end(), 
                               [](const auto& x) { return mat_unary_t(x); });
            prodnode.bind(val_buf.data());
            return prodnode;
        }
    }

    template <class Unary = unary_t, class VecType>
    value_t prod_except(const VecType& v, size_t off,
                        size_t i, size_t j)
    {
        value_t out = 1;
        for (size_t k = 0; k < v.size(); ++k) {
            if (k != off) out *= Unary::fmap(v[k].get(i,j)); 
        }
        return out;
    }

    template <class Unary = unary_t, class VecType>
    value_t prod_except(const VecType& v, size_t i_off, size_t j_off)
    {
        value_t out = 1;
        for (size_t k = 0; k < v.rows(); ++k) {
            for (size_t l = 0; l < v.cols(); ++l) {
                if (k != i_off || l != j_off) {
                    out *= Unary::fmap(v.get(k,l)); 
                }
            }
        }
        return out;
    }
};

// Prod (iter) TEST

TEST_F(prod_fixture, scl_feval)
{
    auto scl_prod = make_prod<ad::scl>();
    value_t res = scl_prod.feval();
    EXPECT_DOUBLE_EQ(res, std::pow(2.*scl_expr.get(), size));
}

TEST_F(prod_fixture, scl_beval)
{
    auto scl_prod = make_prod<ad::scl>();
    scl_prod.feval();
    scl_prod.beval(seed, 0,0, util::beval_policy::single);    // last two ignored
    for (size_t k = 0; k < scl_exprs.size(); ++k) {
        value_t p = prod_except(scl_exprs, k, 0,0);
        EXPECT_DOUBLE_EQ(scl_exprs[k].get_adj(0,0), seed * p * 2.);
    }
}

TEST_F(prod_fixture, vec_feval)
{
    auto vec_prod = make_prod<ad::vec>();
    Eigen::VectorXd res = vec_prod.feval();
    for (int i = 0; i < res.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), 
            std::pow(2.*vec_expr.get(i,0), size));
    }
}

TEST_F(prod_fixture, vec_beval)
{
    auto vec_prod = make_prod<ad::vec>();
    vec_prod.feval();
    vec_prod.beval(seed, 1,0, util::beval_policy::single);    // last ignored
    vec_prod.beval(seed, 3,0, util::beval_policy::single);    // last ignored
    for (size_t k = 0; k < vec_exprs.size(); ++k) {
        for (size_t i = 0; i < vec_size; ++i) {
            value_t p = prod_except(vec_exprs, k, i, 0);
            value_t actual = (i == 1 || i == 3) ? seed*2.*p : 0;
            EXPECT_DOUBLE_EQ(vec_exprs[k].get_adj(i,0), actual);
        }        
    }
}

TEST_F(prod_fixture, mat_feval)
{
    auto mat_prod = make_prod<ad::mat>();
    Eigen::MatrixXd res = mat_prod.feval();
    for (int i = 0; i < res.rows(); ++i) {
        for (int j = 0; j < res.rows(); ++j) {
            EXPECT_DOUBLE_EQ(res(i,j), 
                std::pow(2.*mat_expr.get(i,j), size));
        }
    }
}

TEST_F(prod_fixture, mat_beval)
{
    auto mat_prod = make_prod<ad::mat>();
    mat_prod.feval();
    mat_prod.beval(seed, 1,1, util::beval_policy::single);    // last ignored
    mat_prod.beval(seed, 0,0, util::beval_policy::single);    // last ignored
    for (size_t k = 0; k < mat_exprs.size(); ++k) {
        for (size_t i = 0; i < mat_rows; ++i) {
            for (size_t j = 0; j < mat_cols; ++j) {
                value_t p = prod_except(mat_exprs, k, i, j);
                value_t actual = ((i == 0 && j == 0) ||
                                  (i == 1 && j == 1)) ? seed*2*p : 0;
                EXPECT_DOUBLE_EQ(mat_exprs[k].get_adj(i,j), actual);
            }
        }        
    }
}

TEST_F(prod_fixture, scl_constant)
{
    auto prodnode = prod(scl_exprs.begin(), scl_exprs.end(),
                       [](const auto& x) { return ad::constant(x.get()); });
    static_assert(std::is_same_v<
            std::decay_t<decltype(prodnode)>,
            Constant<double, ad::scl> >);
    EXPECT_DOUBLE_EQ(prodnode.feval(), std::pow(scl_expr.get(), size));
}

TEST_F(prod_fixture, vec_constant)
{
    auto prodnode = prod(vec_exprs.begin(), vec_exprs.end(),
                       [](const auto& x) { 
                            Eigen::VectorXd res = x.get();
                            return ad::constant(res); 
                        });
    static_assert(std::is_same_v<
            std::decay_t<decltype(prodnode)>,
            Constant<double, ad::vec> >);
    auto res = prodnode.feval();
    for (int i = 0; i < res.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), std::pow(vec_expr.get(i,0), size));
    }
}

// Prod (expr) TEST

TEST_F(prod_fixture, scl_expr_feval)
{
    auto scl_prod = ad::prod(scl_unary_t(scl_expr));
    val_buf.resize(scl_prod.bind_size());
    scl_prod.bind(val_buf.data());
    value_t res = scl_prod.feval();
    EXPECT_DOUBLE_EQ(res, 2.*scl_expr.get());
}

TEST_F(prod_fixture, scl_expr_beval)
{
    auto scl_prod = ad::prod(scl_unary_t(scl_expr));
    val_buf.resize(scl_prod.bind_size());
    scl_prod.bind(val_buf.data());
    scl_prod.feval();
    scl_prod.beval(seed, 0,0, util::beval_policy::single);    // last two ignored
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(0,0), 2.*seed);
}

TEST_F(prod_fixture, vec_expr_feval)
{
    auto vec_prod = ad::prod(vec_unary_t(vec_expr));
    val_buf.resize(vec_prod.bind_size());
    vec_prod.bind(val_buf.data());
    value_t res = vec_prod.feval();
    value_t actual = 1;
    for (size_t i = 0; i < vec_expr.size(); ++i) {
        actual *= 2.*vec_expr.get(i,0);
    }
    EXPECT_DOUBLE_EQ(res, actual);
}

TEST_F(prod_fixture, vec_expr_beval)
{
    auto vec_prod = ad::prod(vec_unary_t(vec_expr));
    val_buf.resize(vec_prod.bind_size());
    vec_prod.bind(val_buf.data());
    vec_prod.feval();
    vec_prod.beval(seed, 0,0, util::beval_policy::single);    // last two ignored

    for (size_t i = 0; i < vec_expr.size(); ++i) {
        value_t p = prod_except(vec_expr, i, 0);
        EXPECT_DOUBLE_EQ(vec_expr.get_adj(i,0), 2.*seed * p);
    }
}

TEST_F(prod_fixture, mat_expr_feval)
{
    auto mat_prod = ad::prod(mat_unary_t(mat_expr));
    val_buf.resize(mat_prod.bind_size());
    mat_prod.bind(val_buf.data());
    value_t res = mat_prod.feval();
    value_t actual = 1;
    for (size_t i = 0; i < mat_expr.rows(); ++i) {
        for (size_t j = 0; j < mat_expr.cols(); ++j) {
            actual *= 2.*mat_expr.get(i,j);
        }
    }
    EXPECT_DOUBLE_EQ(res, actual);
}

TEST_F(prod_fixture, mat_expr_beval)
{
    auto mat_prod = ad::prod(mat_unary_t(mat_expr));
    val_buf.resize(mat_prod.bind_size());
    mat_prod.bind(val_buf.data());
    mat_prod.feval();
    mat_prod.beval(seed, 0,0, util::beval_policy::single);    // last two ignored
    for (size_t i = 0; i < mat_expr.rows(); ++i) {
        for (size_t j = 0; j < mat_expr.cols(); ++j) {
            value_t p = prod_except(mat_expr, i, j);
            EXPECT_DOUBLE_EQ(mat_expr.get_adj(i,j), 2.*seed*p);
        }
    }
}

TEST_F(prod_fixture, scl_expr_constant)
{
    auto prodnode = ad::prod(ad::constant(scl_expr.get()));
    static_assert(std::is_same_v<
            std::decay_t<decltype(prodnode)>,
            Constant<double, ad::scl> >);
    EXPECT_DOUBLE_EQ(prodnode.feval(), scl_expr.get());
}

TEST_F(prod_fixture, vec_expr_constant)
{
    auto prodnode = ad::prod(ad::constant(vec_expr.get()));
    static_assert(std::is_same_v<
            std::decay_t<decltype(prodnode)>,
            Constant<double, ad::scl> >);
    auto res = prodnode.feval();
    value_t actual = 1;
    for (size_t i = 0; i < vec_expr.size(); ++i) {
        actual *= vec_expr.get(i,0);
    }
    EXPECT_DOUBLE_EQ(res, actual);
}

} // namespace core
} // namespace ad
