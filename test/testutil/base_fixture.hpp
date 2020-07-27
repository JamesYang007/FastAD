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

    template <class T>
    static auto bmap(T) { return 2.; }
};

// Represents f(x, y) = x - 2*y
struct MockBinary
{
    template <class T, class U>
    static auto fmap(T x, U y)
    { return x - 2*y; }

    template <class T, class U>
    static auto blmap(T, U) 
    { return 1.; }

    template <class T, class U>
    static auto brmap(T, U)
    { return -2.; }
};

struct base_fixture : ::testing::Test
{
protected:
    using value_t = double;
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

    std::vector<value_t> val_buf;

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
        size_t buf_size = expr.bind_size();
        val_buf.resize(buf_size);
        expr.bind(val_buf.data());
    }
};

} // namespace ad
