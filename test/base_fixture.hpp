#pragma once
#include "gtest/gtest.h"
#include <array>
#include <fastad_bits/var.hpp>

namespace ad {

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
    {
        initialize();
    }

    // random (fixed) initialization
    // only makes sense when default constructed base fixture
    void initialize()
    {
        scl_expr.get() = 2.31;

        auto& vec_raw = vec_expr.get();
        vec_raw(0) = 3.1;
        vec_raw(1) = -2.3;
        vec_raw(2) = 1.3;
        vec_raw(3) = 0.2;
        vec_raw(4) = 5.1;

        auto& mat_raw = mat_expr.get();
        mat_raw(0,0) = 3.1;
        mat_raw(0,1) = -2.3;
        mat_raw(0,2) = 1.3;
        mat_raw(1,0) = 0.2;
        mat_raw(1,1) = 5.1;
        mat_raw(1,2) = -0.9;
    }
};

} // namespace ad
