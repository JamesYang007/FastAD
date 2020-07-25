#pragma once
#include <fastad_bits/expr_base.hpp>
#include <fastad_bits/shape_traits.hpp>
#include <fastad_bits/value_view.hpp>
#include <fastad_bits/type_traits.hpp>
#include <fastad_bits/constant.hpp>

namespace ad {
namespace core {

/**
 * IfElseNode represents an if-else statement.
 * During AD computation, forward evaluation evaluates the condition
 * expression and if true, evaluates the IfExprType object.
 * Otherwise, evaluates ElseExprType object.
 *
 * Both if and else expressions must have same value and shape type.
 * Currently, condition expression can only be a scalar.
 *
 * @tparam  CondExprType    type of condition expression
 * @tparam  IfExprType      type of expression in if-statement
 * @tparam  ElseExprType    type of expression in else-statement
 */

template <class CondExprType
        , class IfExprType
        , class ElseExprType>
struct IfElseNode : 
    ValueView<typename util::expr_traits<IfExprType>::value_t,
              typename util::shape_traits<IfExprType>::shape_t>,
    ExprBase<IfElseNode<CondExprType, IfExprType, ElseExprType>>
{
private:
    using cond_t = CondExprType;
    using if_t = IfExprType;
    using else_t = ElseExprType;
    using if_value_t = typename util::expr_traits<if_t>::value_t;
    using else_value_t = typename util::expr_traits<else_t>::value_t;
    using if_shape_t = typename util::shape_traits<if_t>::shape_t;
    using else_shape_t = typename util::shape_traits<else_t>::shape_t;

    // all expr types must AD expressions
    static_assert(util::is_expr_v<if_t> &&
                  util::is_expr_v<else_t> &&
                  util::is_expr_v<cond_t>);
    
    // restrict shape combinations
    static_assert(
        util::is_scl_v<cond_t> &&
        std::is_same_v<if_shape_t, else_shape_t>
        );

    // restrict value combinations
    static_assert(std::is_same_v<if_value_t, else_value_t>);

public:
    using value_view_t = ValueView<if_value_t, if_shape_t>;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::bind;

    IfElseNode(const cond_t& cond_expr,
               const if_t& if_expr,
               const else_t& else_expr)
        : value_view_t(nullptr, 
                       if_expr.rows(), 
                       if_expr.cols())
        , cond_expr_{cond_expr}
        , if_expr_{if_expr}
        , else_expr_{else_expr}
    {
        // assert same size of if and else expressions
        assert(if_expr.rows() == else_expr.rows());
        assert(if_expr.cols() == else_expr.cols());
    }

    const var_t& feval()
    {
        if (cond_expr_.feval()) {
            return this->get() = if_expr_.feval();
        } else {
            return this->get() = else_expr_.feval();
        }
    }

    void beval(value_t seed, size_t i, size_t j, util::beval_policy pol)
    {
        if (seed == 0) return;
        if (cond_expr_.get()) {
            if_expr_.beval(seed, i, j, pol);
        } else {
            else_expr_.beval(seed, i, j, pol);
        } 
    }

    value_t* bind(value_t* begin) 
    {
        value_t* next = begin;
        if constexpr (!util::is_var_view_v<cond_t>) {
            next = cond_expr_.bind(next);
        }
        if constexpr (!util::is_var_view_v<if_t>) {
            next = if_expr_.bind(next);
        }
        if constexpr (!util::is_var_view_v<else_t>) {
            next = else_expr_.bind(next);
        }
        return value_view_t::bind(next);
    }

    size_t bind_size() const 
    { 
        return single_bind_size() + 
                cond_expr_.bind_size() +
                if_expr_.bind_size() + 
                else_expr_.bind_size();
    }

    size_t single_bind_size() const
    {
        return this->size();
    }

private:
    cond_t cond_expr_;
    if_t if_expr_;
    else_t else_expr_;
};

} // namespace core

template <class CondExprType
        , class IfExprType
        , class ElseExprType>
inline constexpr auto if_else(const core::ExprBase<CondExprType>& cond_expr,
                              const core::ExprBase<IfExprType>& if_expr,
                              const core::ExprBase<ElseExprType>& else_expr)
{
    using if_t = IfExprType;
    using if_value_t = typename util::expr_traits<if_t>::value_t;
    using if_shape_t = typename util::shape_traits<if_t>::shape_t;

    // optimized if every expression type is constant
    if constexpr (util::is_constant_v<CondExprType> &&
                  util::is_constant_v<IfExprType> &&
                  util::is_constant_v<ElseExprType>) {

        using var_t = core::details::constant_var_t<if_value_t, if_shape_t>;

        var_t if_out = if_expr.self().feval();
        var_t else_out = else_expr.self().feval();

        assert(if_expr.self().rows() == else_expr.self().rows());
        assert(if_expr.self().cols() == else_expr.self().cols());

        return cond_expr.self().feval() ? 
            ad::constant(if_out) :
            ad::constant(else_out);

    } else {
        return core::IfElseNode(cond_expr.self(), 
                                if_expr.self(), 
                                else_expr.self());
    }
}

} // namespace ad
