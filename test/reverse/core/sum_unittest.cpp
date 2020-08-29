#include "gtest/gtest.h"
#include <fastad_bits/reverse/core/unary.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/reverse/core/sum.hpp>
#include <testutil/base_fixture.hpp>

namespace ad {
namespace core {

struct sum_fixture : base_fixture
{
protected:
    using unary_t = MockUnary;
    using scl_unary_t = UnaryNode<unary_t, scl_expr_view_t>;
    using vec_unary_t = UnaryNode<unary_t, vec_expr_view_t>;
    using mat_unary_t = UnaryNode<unary_t, mat_expr_view_t>;

    size_t size = 5;
    value_t seed = 3.14;
    aVectorXd vseed;

    std::vector<scl_expr_t> scl_exprs;
    std::vector<vec_expr_t> vec_exprs;
    std::vector<mat_expr_t> mat_exprs;

    Eigen::VectorXd val_buf;
    Eigen::VectorXd adj_buf;

    sum_fixture()
        : base_fixture()
        , vseed(size)
        , scl_exprs(size)
        , vec_exprs(size, vec_expr_t{vec_size})
        , mat_exprs(size, mat_expr_t{mat_rows, mat_cols})
        , val_buf((size+1)*std::max(vec_size, mat_size))
        , adj_buf((size+1)*std::max(vec_size, mat_size))
    {
        vseed << 0.23, 0.1, 3.231, 4.23, 5.;
        for (auto& expr : scl_exprs) this->scl_initialize(expr);
        for (auto& expr : vec_exprs) this->vec_initialize(expr);
        for (auto& expr : mat_exprs) this->mat_initialize(expr);
    }

    // Creates a sum node depending on UnaryType (each expression)
    // Already binds the node to val_buf.
    template <class ShapeType>
    auto make_sum()
    {
        ptr_pack_t ptr_pack(val_buf.data(), adj_buf.data());
        if constexpr (std::is_same_v<ShapeType, ad::scl>) {
            auto sumnode = sum(scl_exprs.begin(), scl_exprs.end(), 
                               [](const auto& x) { return scl_unary_t(x); });
            sumnode.bind_cache(ptr_pack);
            return sumnode;
        } else if constexpr (std::is_same_v<ShapeType, ad::vec>) {
            auto sumnode = sum(vec_exprs.begin(), vec_exprs.end(), 
                               [](const auto& x) { return vec_unary_t(x); });
            sumnode.bind_cache(ptr_pack);
            return sumnode;
        } else {
            auto sumnode = sum(mat_exprs.begin(), mat_exprs.end(), 
                               [](const auto& x) { return mat_unary_t(x); });
            sumnode.bind_cache(ptr_pack);
            return sumnode;
        }
    }
};

// Sum (iter) TEST

TEST_F(sum_fixture, scl_feval)
{
    auto scl_sum = make_sum<ad::scl>();
    value_t res = scl_sum.feval();
    EXPECT_DOUBLE_EQ(res, size*2.*scl_expr.get());
}

TEST_F(sum_fixture, scl_beval)
{
    auto scl_sum = make_sum<ad::scl>();
    scl_sum.beval(seed);
    for (size_t i = 0; i < scl_exprs.size(); ++i) {
        EXPECT_DOUBLE_EQ(scl_exprs[i].get_adj(0,0), 2.*seed);
    }
}

TEST_F(sum_fixture, vec_feval)
{
    auto vec_sum = make_sum<ad::vec>();
    Eigen::VectorXd res = vec_sum.feval();
    for (int i = 0; i < res.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), size*2.*vec_expr.get(i,0));
    }
}

TEST_F(sum_fixture, vec_beval)
{
    auto vec_sum = make_sum<ad::vec>();
    vec_sum.beval(vseed);
    for (size_t k = 0; k < vec_exprs.size(); ++k) {
        for (size_t i = 0; i < vec_size; ++i) {
            EXPECT_DOUBLE_EQ(vec_exprs[k].get_adj(i,0), 2*vseed[i]);
        }        
    }
}

TEST_F(sum_fixture, mat_feval)
{
    auto mat_sum = make_sum<ad::mat>();
    Eigen::MatrixXd res = mat_sum.feval();
    for (int i = 0; i < res.rows(); ++i) {
        for (int j = 0; j < res.cols(); ++j) {
            EXPECT_DOUBLE_EQ(res(i,j), size*2.*mat_expr.get(i,j));
        }
    }
}

TEST_F(sum_fixture, mat_beval)
{
    auto mat_sum = make_sum<ad::mat>();
    mat_sum.beval(seed);
    for (size_t k = 0; k < mat_exprs.size(); ++k) {
        for (size_t i = 0; i < mat_rows; ++i) {
            for (size_t j = 0; j < mat_cols; ++j) {
                EXPECT_DOUBLE_EQ(mat_exprs[k].get_adj(i,j), 2*seed);
            }
        }        
    }
}

TEST_F(sum_fixture, scl_constant)
{
    auto sumnode = sum(scl_exprs.begin(), scl_exprs.end(),
                       [](const auto& x) { return ad::constant(x.get()); });
    static_assert(std::is_same_v<
            std::decay_t<decltype(sumnode)>,
            Constant<double, ad::scl> >);
    EXPECT_DOUBLE_EQ(sumnode.feval(), size*scl_expr.get());
}

TEST_F(sum_fixture, vec_constant)
{
    auto sumnode = sum(vec_exprs.begin(), vec_exprs.end(),
                       [](const auto& x) { 
                            Eigen::VectorXd res = x.get();
                            return ad::constant(res); 
                        });
    static_assert(std::is_same_v<
            std::decay_t<decltype(sumnode)>,
            Constant<double, ad::vec> >);
    auto res = sumnode.feval();
    for (int i = 0; i < res.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), size*vec_expr.get(i,0));
    }
}

// Sum (expr) TEST

TEST_F(sum_fixture, scl_expr_feval)
{
    auto scl_sum = ad::sum(scl_unary_t(scl_expr));
    val_buf.resize(scl_sum.bind_cache_size()(0));
    adj_buf.resize(scl_sum.bind_cache_size()(1));
    scl_sum.bind_cache({val_buf.data(), adj_buf.data()});
    value_t res = scl_sum.feval();
    EXPECT_DOUBLE_EQ(res, 2.*scl_expr.get());
}

TEST_F(sum_fixture, scl_expr_beval)
{
    auto scl_sum = ad::sum(scl_unary_t(scl_expr));
    val_buf.resize(scl_sum.bind_cache_size()(0));
    adj_buf.resize(scl_sum.bind_cache_size()(1));
    scl_sum.bind_cache({val_buf.data(), adj_buf.data()});
    scl_sum.beval(seed);
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(), 2.*seed);
}

TEST_F(sum_fixture, vec_expr_feval)
{
    auto vec_sum = ad::sum(vec_unary_t(vec_expr));
    val_buf.resize(vec_sum.bind_cache_size()(0));
    adj_buf.resize(vec_sum.bind_cache_size()(1));
    vec_sum.bind_cache({val_buf.data(), adj_buf.data()});
    value_t res = vec_sum.feval();
    value_t actual = 0;
    for (size_t i = 0; i < vec_expr.size(); ++i) {
        actual += 2.*vec_expr.get(i,0);
    }
    EXPECT_DOUBLE_EQ(res, actual);
}

TEST_F(sum_fixture, vec_expr_beval)
{
    auto vec_sum = ad::sum(vec_unary_t(vec_expr));
    val_buf.resize(vec_sum.bind_cache_size()(0));
    adj_buf.resize(vec_sum.bind_cache_size()(1));
    vec_sum.bind_cache({val_buf.data(), adj_buf.data()});
    vec_sum.beval(seed);
    for (size_t i = 0; i < vec_expr.size(); ++i) {
        EXPECT_DOUBLE_EQ(vec_expr.get_adj(i,0), 2.*seed);
    }
}

TEST_F(sum_fixture, mat_expr_feval)
{
    auto mat_sum = ad::sum(mat_unary_t(mat_expr));
    val_buf.resize(mat_sum.bind_cache_size()(0));
    adj_buf.resize(mat_sum.bind_cache_size()(1));
    mat_sum.bind_cache({val_buf.data(), adj_buf.data()});
    value_t res = mat_sum.feval();
    value_t actual = 0;
    for (size_t i = 0; i < mat_expr.rows(); ++i) {
        for (size_t j = 0; j < mat_expr.cols(); ++j) {
            actual += 2.*mat_expr.get(i,j);
        }
    }
    EXPECT_DOUBLE_EQ(res, actual);
}

TEST_F(sum_fixture, mat_expr_beval)
{
    auto mat_sum = ad::sum(mat_unary_t(mat_expr));
    val_buf.resize(mat_sum.bind_cache_size()(0));
    adj_buf.resize(mat_sum.bind_cache_size()(1));
    mat_sum.bind_cache({val_buf.data(), adj_buf.data()});
    mat_sum.beval(seed);
    for (size_t i = 0; i < mat_expr.rows(); ++i) {
        for (size_t j = 0; j < mat_expr.cols(); ++j) {
            EXPECT_DOUBLE_EQ(mat_expr.get_adj(i,j), 2.*seed);
        }
    }
}

TEST_F(sum_fixture, scl_expr_constant)
{
    auto sumnode = ad::sum(ad::constant(scl_expr.get()));
    static_assert(std::is_same_v<
            std::decay_t<decltype(sumnode)>,
            Constant<double, ad::scl> >);
    EXPECT_DOUBLE_EQ(sumnode.feval(), scl_expr.get());
}

TEST_F(sum_fixture, vec_expr_constant)
{
    auto sumnode = ad::sum(ad::constant(vec_expr.get()));
    static_assert(std::is_same_v<
            std::decay_t<decltype(sumnode)>,
            Constant<double, ad::scl> >);
    auto res = sumnode.feval();
    value_t actual = 0;
    for (size_t i = 0; i < vec_expr.size(); ++i) {
        actual += vec_expr.get(i,0);
    }
    EXPECT_DOUBLE_EQ(res, actual);
}

} // namespace core
} // namespace ad
