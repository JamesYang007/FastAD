#pragma once
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_view.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/util/type_traits.hpp>

namespace ad {
namespace core {

/**
 * MatVecDotNode represents a matrix multiplication with a (column) vector.
 * Indeed, the left expression must be a matrix shape, and the right be a vector.
 * No other shapes are permitted for this node.
 * At construction, the actual sizes of the two are checked -
 * specifically, the number of columns for matrix must equal the number of rows for vector.
 *
 * We assert that the value type be the same for the two expressions.
 * The output shape is always a (column) vector.
 *
 * @tparam  MatExprType     type of matrix expression
 * @tparam  VecExprType     type of vector expression
 */

template <class MatExprType
        , class VecExprType>
struct MatVecDotNode:
    ValueView<typename util::expr_traits<MatExprType>::value_t,
              ad::vec>,
    ExprBase<MatVecDotNode<MatExprType, VecExprType>>
{
private:
    using mat_expr_t = MatExprType;
    using vec_expr_t = VecExprType;
    using mat_value_t = typename 
        util::expr_traits<mat_expr_t>::value_t;

    // assert that both expressions have same value type
    static_assert(std::is_same_v<
            typename util::expr_traits<mat_expr_t>::value_t,
            typename util::expr_traits<vec_expr_t>::value_t>);

    // assert correct shapes
    static_assert(util::is_mat_v<mat_expr_t>);
    static_assert(util::is_vec_v<vec_expr_t>);

public:
    using value_view_t = ValueView<mat_value_t, ad::vec>;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::bind;

    MatVecDotNode(const mat_expr_t& mat_expr,
                  const vec_expr_t& vec_expr)
        : value_view_t(nullptr, mat_expr.rows(), 1)
        , mat_expr_{mat_expr}
        , vec_expr_{vec_expr}
    {
        assert(mat_expr.cols() == vec_expr.rows());
    }

    const var_t& feval()
    {
        auto&& mat_val = mat_expr_.feval();
        auto&& vec_val = vec_expr_.feval();
        return this->get() = mat_val * vec_val;
    }

    void beval(value_t seed, size_t i, size_t, util::beval_policy pol)
    {
        if (seed == 0) return;
        for (size_t j = 0; j < vec_expr_.size(); ++j) {
            vec_expr_.beval(seed * mat_expr_.get(i,j), j, 0, pol);
        }
        for (size_t j = 0; j < vec_expr_.size(); ++j) {
            mat_expr_.beval(seed * vec_expr_.get(j,0), i, j, pol);
        }
    }

    value_t* bind(value_t* begin) 
    {
        value_t* next = begin;
        if constexpr (!util::is_var_view_v<mat_expr_t>) {
            next = mat_expr_.bind(next);
        }
        if constexpr (!util::is_var_view_v<vec_expr_t>) {
            next = vec_expr_.bind(next);
        }
        return value_view_t::bind(next);
    }

    size_t bind_size() const 
    { 
        return single_bind_size() + 
                mat_expr_.bind_size() + 
                vec_expr_.bind_size();
    }

    size_t single_bind_size() const
    {
        return this->size();
    }


private:
    mat_expr_t mat_expr_;
    vec_expr_t vec_expr_;
};

} // namespace core

template <class MatExprType
        , class VecExprType
        , class = std::enable_if_t<
            util::is_mat_v<MatExprType> &&
            util::is_vec_v<VecExprType>
        >
    >
inline auto dot(const core::ExprBase<MatExprType>& mat_expr,
                const core::ExprBase<VecExprType>& vec_expr)
{
    using mat_expr_t = MatExprType;
    using vec_expr_t = VecExprType;
    using mat_value_t = typename util::expr_traits<mat_expr_t>::value_t;
    using vec_value_t = typename util::expr_traits<vec_expr_t>::value_t;

    // optimization for when both expressions are constant
    if constexpr (util::is_constant_v<mat_expr_t> &&
                  util::is_constant_v<vec_expr_t>) {
        static_assert(std::is_same_v<mat_value_t, vec_value_t>);

        using var_t = core::details::constant_var_t<vec_value_t, ad::vec>;
        var_t out = mat_expr.self().feval() * vec_expr.self().feval();
        return ad::constant(out);
    } else {
        return core::MatVecDotNode<mat_expr_t, vec_expr_t>(
                mat_expr.self(), vec_expr.self());
    }
}

} // namespace ad
