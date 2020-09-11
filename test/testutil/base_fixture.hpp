#pragma once
#include "gtest/gtest.h"
#include <array>
#include <fastad_bits/reverse/core/var.hpp>

namespace ad {

// Represents f(x) = 2*x
struct MockUnary
{
    template <class T>
    static auto fmap(T x) { return 2.*x; }

    template <class S, class T, class U>
    static auto bmap(S seed, T, U) { return seed * 2.; }
};

// Represents f(x, y) = x - 2*y
struct MockBinary
{
    static constexpr bool is_comparison = false;

    template <class T, class U>
    static auto fmap(const T& x, const U& y)
    { return x - 2.*y; }

    template <class S, class T, class U, class F>
    static auto blmap(const S& seed, const T&, const U&, const F&) 
    { 
        if constexpr (!util::is_eigen_v<T> && util::is_eigen_v<U>) {
            return seed.sum();
        } else {
            return seed; 
        }
    }

    template <class S, class T, class U, class F>
    static auto brmap(const S& seed, const T&, const U&, const F&) 
    { 
        if constexpr (util::is_eigen_v<T> && !util::is_eigen_v<U>) {
            return -2. * seed.sum();
        } else {
            return -2. * seed; 
        }
    }
};

struct base_fixture : ::testing::Test
{
protected:
    using value_t = double;
    using aVectorXd = Eigen::Array<value_t, Eigen::Dynamic, 1>;
    using ptr_pack_t = util::PtrPack<value_t>;

    using scl_expr_t = Var<value_t, ad::scl>;
    using vec_expr_t = Var<value_t, ad::vec>;
    using mat_expr_t = Var<value_t, ad::mat>;
    using scl_expr_view_t = VarView<value_t, ad::scl>;
    using vec_expr_view_t = VarView<value_t, ad::vec>;
    using mat_expr_view_t = VarView<value_t, ad::mat>;

    size_t vec_size;
    size_t mat_rows;
    size_t mat_cols;
    size_t mat_size;

    scl_expr_t scl_expr;
    vec_expr_t vec_expr;
    mat_expr_t mat_expr;

    Eigen::VectorXd val_buf;
    Eigen::VectorXd adj_buf;

    base_fixture(size_t vec_size=5,
                 size_t mat_rows=2,
                 size_t mat_cols=3)
        : vec_size(vec_size)
        , mat_rows(mat_rows)
        , mat_cols(mat_cols)
        , mat_size(mat_rows * mat_cols)
        , scl_expr()
        , vec_expr(vec_size)
        , mat_expr(mat_rows, mat_cols)
        , val_buf()
        , adj_buf()
    {
        // if default setting, initialize
        scl_initialize(scl_expr);
        if (vec_size == 5) {
            vec_initialize(vec_expr);
        }
        if (mat_rows == 2 &&
            mat_cols == 3) {
            mat_initialize(mat_expr);
        }
    }

    // random (fixed) initialization
    // only makes sense when default constructed base fixture
    template <class T>
    void scl_initialize(T& scl_expr)
    {
        scl_expr.get() = 2.31;
    }

    template <class T>
    void vec_initialize(T& vec_expr)
    {
        auto& vec_raw = vec_expr.get();
        vec_raw(0) = 3.1;
        vec_raw(1) = -2.3;
        vec_raw(2) = 1.3;
        vec_raw(3) = 0.;
        vec_raw(4) = 5.1;
    }

    template <class T>
    void mat_initialize(T& mat_expr)
    {
        auto& mat_raw = mat_expr.get();
        mat_raw(0,0) = 3.1;
        mat_raw(0,1) = -2.3;
        mat_raw(0,2) = 1.3;
        mat_raw(1,0) = 0.2;
        mat_raw(1,1) = 5.1;
        mat_raw(1,2) = -0.9;
    }

    template <class T>
    void bind(T& expr)
    {
        auto buf_size = expr.bind_cache_size();
        val_buf.resize(buf_size(0));
        adj_buf.resize(buf_size(1));
        expr.bind_cache({val_buf.data(), adj_buf.data()});
    }

    void check_eq(value_t x, value_t y)
    {
        EXPECT_DOUBLE_EQ(x, y);
    }

    template <class T, class U>
    void check_eq(const Eigen::DenseBase<T>& x,
                  const Eigen::DenseBase<U>& y)
    {
        EXPECT_EQ(x.rows(), y.rows());
        EXPECT_EQ(x.cols(), y.cols());
        for (int i = 0; i < x.rows(); ++i) {
            for (int j = 0; j < x.cols(); ++j) {
                EXPECT_DOUBLE_EQ(x(i,j), y(i,j));
            }
        }
    }

    template <class T, class U>
    void check_near(const Eigen::DenseBase<T>& x,
                    const Eigen::DenseBase<U>& y,
                    double tol = 1e-15)
    {
        EXPECT_EQ(x.rows(), y.rows());
        EXPECT_EQ(x.cols(), y.cols());
        for (int i = 0; i < x.rows(); ++i) {
            for (int j = 0; j < x.cols(); ++j) {
                EXPECT_NEAR(x(i,j), y(i,j), tol);
            }
        }
    }
};

} // namespace ad
