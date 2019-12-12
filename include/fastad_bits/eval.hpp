#pragma once
#include <cstdlib>
#include <type_traits>
#include <tuple>

namespace ad {

// Evaluates expression in the forward direction of reverse-mode AD.
// @tparam ExprType expression type
// @param expr  expression to forward evaluate
// @return the expression value
template <class ExprType>
inline auto evaluate(ExprType&& expr)
{
    return expr.feval();
}

// Evaluates expression in the backward direction of reverse-mode AD.
// @tparam ExprType expression type
// @param expr  expression to backward evaluate
template <class ExprType>
inline void evaluate_adj(ExprType&& expr)
{
    expr.beval(1);
}

// Evaluates expression both in the forward and backward direction of reverse-mode AD.
// @tparam ExprType expression type
// @param expr  expression to forward and backward evaluate
// Returns the forward expression value
template <class ExprType>
inline auto autodiff(ExprType&& expr)
{
    auto t = evaluate(expr);
    evaluate_adj(expr);
    return t;
}

namespace details {

///////////////////////////////////////////////////////
// Sequential autodiff
///////////////////////////////////////////////////////

// This function is the ending condition when number of expressions is equal to I.
// @tparam I    index of first expression to auto-differentiate
// @tparam ExprTypes expression types
template <size_t I, class... ExprTypes>
inline typename std::enable_if<I == sizeof...(ExprTypes)>::type
autodiff(std::tuple<ExprTypes...>&) 
{}

// This function calls ad::autodiff from the Ith expression to the last expression in tup.
// @tparam I    index of first expression to auto-differentiate
// @tparam ExprTypes    expression types
// @param tup   the tuple of expressions to auto-differentiate
template <size_t I, class... ExprTypes>
inline typename std::enable_if < I < sizeof...(ExprTypes)>::type
autodiff(std::tuple<ExprTypes...>& tup)
{
    ad::autodiff(std::get<I>(tup)); 
    autodiff<I + 1>(tup);
}

} // namespace details 

// Auto-differentiator for lvalue reference of tuple of expressions.
// Always processes sequentially.
// @tparam  ExprTypes   expression types
// @param   tup tuple of expressions to auto-differentiate
template <class... ExprTypes>
inline void autodiff(std::tuple<ExprTypes...>& tup)
{
    details::autodiff<0>(tup);
}

// Auto-differentiator for rvalue reference of tuple of expressions
// Always processes sequentially.
// @tparam  ExprTypes   expression types
// @param   tup tuple of expressions to auto-differentiate
template <class... ExprTypes>
inline void autodiff(std::tuple<ExprTypes...>&& tup)
{
    details::autodiff<0>(tup);
}

} // namespace ad
