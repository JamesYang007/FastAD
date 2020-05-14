#pragma once
#include <fastad_bits/node.hpp>

namespace ad {
namespace core {
namespace details {

// The result is expected to be DualNum<ValueType>
// for some reasonable ValueType (like double).
template <class T1, class T2, class T3>
using common_data_t = std::common_type_t<
    typename T1::data_t,
    typename T2::data_t,
    typename T3::data_t
        >;

} // namespace details

/*
 * An AD expression to represent if-else statement.
 * During AD computation, forward evaluation evaluates the condition
 * expression and if true, evaluates the IfExprType object.
 * Otherwise, evaluates ElseExprType object.
 */
template <class CondExprType
        , class IfExprType
        , class ElseExprType>
struct IfElseNode : 
    public details::common_data_t<
        CondExprType, IfExprType, ElseExprType
    >, 
    ADNodeExpr<IfElseNode<CondExprType, IfExprType, ElseExprType>>
{
    using data_t = details::common_data_t<
        CondExprType, IfExprType, ElseExprType
    >;
    using value_type = typename data_t::value_type;

    IfElseNode(const CondExprType& cond,
               const IfExprType& if_expr,
               const ElseExprType& else_expr)
        : data_t(0., 0.)
        , cond_{cond}, if_expr_{if_expr}, else_expr_{else_expr}
    {}

    // Forward evaluation.
    value_type feval()
    {
        if (cond_.feval()) {
            return this->set_value(if_expr_.feval());
        } else {
            return this->set_value(else_expr_.feval());
        }
    }

    // Backward evaluation.
    // Assumes forward evaluated before calling beval.
    void beval(value_type seed)
    {
        this->set_adjoint(seed);
        if (cond_.get_value()) {
            if_expr_.beval(seed);
        } else {
            else_expr_.beval(seed);
        } 
    }

private:

    CondExprType cond_;
    IfExprType if_expr_;
    ElseExprType else_expr_;
};

} // namespace core

template <class CondExprType
        , class IfExprType
        , class ElseExprType>
inline constexpr auto if_else(const core::ADNodeExpr<CondExprType>& cond,
                              const core::ADNodeExpr<IfExprType>& if_expr,
                              const core::ADNodeExpr<ElseExprType>& else_expr)
{
    using value_t = typename core::details::common_data_t<
        CondExprType, IfExprType, ElseExprType>::value_type;

    // optimized if every expression type is constant
    if constexpr (std::is_same_v<CondExprType, core::ConstNode<value_t>> &&
                  std::is_same_v<IfExprType, core::ConstNode<value_t>> &&
                  std::is_same_v<ElseExprType, core::ConstNode<value_t>>) {
        return cond.self().get_value() ? 
            ad::constant(if_expr.self().get_value()) :
            ad::constant(else_expr.self().get_value());
    } else {
        return core::IfElseNode(cond.self(), if_expr.self(), else_expr.self());
    }
}

} // namespace ad
