#pragma once
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_adj_view.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/util/shape_traits.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/util/size_pack.hpp>
#include <fastad_bits/util/value.hpp>

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
    ValueAdjView<typename util::expr_traits<IfExprType>::value_t,
                 typename util::shape_traits<IfExprType>::shape_t >,
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
    using value_adj_view_t = ValueAdjView<if_value_t, if_shape_t>;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    IfElseNode(const cond_t& cond_expr,
               const if_t& if_expr,
               const else_t& else_expr)
        : value_adj_view_t(nullptr, nullptr, if_expr.rows(), if_expr.cols())
        , cond_expr_{cond_expr}
        , if_expr_{if_expr}
        , else_expr_{else_expr}
    {
        // assert same size of if and else expressions
        assert(if_expr.rows() == else_expr.rows());
        assert(if_expr.cols() == else_expr.cols());
    }

    const auto& feval()
    {
        return cond_expr_.feval() ? 
                if_expr_.feval() : else_expr_.feval();
    }

    template <class T>
    void beval(const T& seed)
    {
        if (cond_expr_.get()) {
            if_expr_.beval(seed);
        } else {
            else_expr_.beval(seed);
        } 
    }

    ptr_pack_t bind_cache(ptr_pack_t begin)
    {
        begin = cond_expr_.bind_cache(begin);
        begin = if_expr_.bind_cache(begin);
        return else_expr_.bind_cache(begin);
    }

    util::SizePack bind_cache_size() const 
    { 
        return cond_expr_.bind_cache_size() +
                if_expr_.bind_cache_size() + 
                else_expr_.bind_cache_size();
    }

    util::SizePack single_bind_cache_size() const
    { return {0,0}; }

private:
    cond_t cond_expr_;
    if_t if_expr_;
    else_t else_expr_;
};

} // namespace core

template <class CondType
        , class IfType
        , class ElseType
        , class = std::enable_if_t<
            util::is_convertible_to_ad_v<CondType> &&
            util::is_convertible_to_ad_v<IfType> && 
            util::is_convertible_to_ad_v<ElseType> &&
            util::any_ad_v<CondType, IfType, ElseType> >>
inline constexpr auto if_else(const CondType& c,
                              const IfType& i,
                              const ElseType& e)
{
    using cond_t = util::convert_to_ad_t<CondType>;
    using if_t = util::convert_to_ad_t<IfType>;
    using else_t = util::convert_to_ad_t<ElseType>;
    using if_value_t = typename util::expr_traits<if_t>::value_t;
    using if_shape_t = typename util::shape_traits<if_t>::shape_t;

    cond_t cond_expr = c;
    if_t if_expr = i;
    else_t else_expr = e;

    // optimized if every expression type is constant
    if constexpr (util::is_constant_v<cond_t> &&
                  util::is_constant_v<if_t> &&
                  util::is_constant_v<else_t>) {

        using var_t = util::constant_var_t<if_value_t, if_shape_t>;

        var_t if_out = if_expr.feval();
        var_t else_out = else_expr.feval();

        assert(if_expr.rows() == else_expr.rows());
        assert(if_expr.cols() == else_expr.cols());

        return cond_expr.feval() ? 
            ad::constant(if_out) :
            ad::constant(else_out);

    } else {
        return core::IfElseNode(cond_expr, 
                                if_expr, 
                                else_expr);
    }
}

} // namespace ad
