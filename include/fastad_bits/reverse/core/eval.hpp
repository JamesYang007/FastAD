#pragma once
#include <cstdlib>
#include <type_traits>
#include <tuple>
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/bind.hpp>
#include <fastad_bits/util/value.hpp>

namespace ad {

/*
 * Evaluates expression in the forward direction of reverse-mode AD.
 * @tparam ExprType expression type
 * @param expr  expression to forward evaluate
 * @return the expression value
 */

template <class ExprType>
inline auto evaluate(ExprType&& expr)
{
    return expr.feval();
}

template <class ExprType>
inline auto evaluate(core::ExprBind<ExprType>& expr)
{
    return expr.get().feval();
}

template <class ExprType>
inline auto evaluate(core::ExprBind<ExprType>&& expr)
{
    return expr.get().feval();
}

/* 
 * Evaluates expression in the backward direction of reverse-mode AD.
 * Default parameter should fail exactly when expression is multi-dimensional.
 *
 * @tparam ExprType expression type
 * @param expr  expression to backward evaluate
 */
template <class ExprType>
inline std::enable_if_t<util::is_scl_v<std::decay_t<ExprType>>> 
evaluate_adj(ExprType&& expr, 
             typename util::expr_traits<std::decay_t<ExprType>>::value_t seed = 1.)
{
    expr.beval(seed);
}

template <class ExprType, class T>
inline std::enable_if_t<!util::is_scl_v<std::decay_t<ExprType>>> 
evaluate_adj(ExprType&& expr, 
             const Eigen::ArrayBase<T>& seed)
{
    expr.beval(seed);
}

template <class ExprType>
inline std::enable_if_t<util::is_scl_v<std::decay_t<ExprType>>> 
evaluate_adj(core::ExprBind<ExprType>& expr, 
             typename util::expr_traits<std::decay_t<ExprType>>::value_t seed = 1.)
{
    evaluate_adj(expr.get(), seed);
}

template <class ExprType, class T>
inline std::enable_if_t<!util::is_scl_v<std::decay_t<ExprType>>> 
evaluate_adj(core::ExprBind<ExprType>&& expr, 
             const Eigen::ArrayBase<T>& seed)
{
    evaluate_adj(expr.get(), seed);
}

/* 
 * Evaluates expression both in the forward and backward direction of reverse-mode AD.
 * @tparam ExprType expression type
 * @param expr  expression to forward and backward evaluate
 * Returns the forward expression value
 */

template <class ExprType
        , class = std::enable_if_t<util::is_scl_v<std::decay_t<ExprType>>> 
        >
inline auto autodiff(ExprType&& expr,
                     typename util::expr_traits<
                        std::decay_t<ExprType>>::value_t seed = 1.)
{
    auto t = evaluate(expr);
    evaluate_adj(expr, seed);
    return t;
}

template <class ExprType
        , class T
        , class = std::enable_if_t<!util::is_scl_v<std::decay_t<ExprType>>> 
        >
inline auto autodiff(ExprType&& expr,
                     const Eigen::ArrayBase<T>& seed)
{
    auto t = evaluate(expr);
    evaluate_adj(expr, seed);
    return t;
}

/** 
 * Evaluates expression both in the forward and backward direction of reverse-mode AD.
 * Overload for ExprBind helper class.
 *
 * @tparam ExprType expression type
 * @param expr  expression to forward and backward evaluate
 * Returns the forward expression value
 */

template <class ExprType
        , class = std::enable_if_t<util::is_scl_v<std::decay_t<ExprType>>> 
        >
inline auto autodiff(core::ExprBind<ExprType>& expr,
                     typename util::expr_traits<
                        std::decay_t<ExprType>>::value_t seed = 1.)
{
    return autodiff(expr.get(), seed);
}

template <class ExprType
        , class T
        , class = std::enable_if_t<!util::is_scl_v<std::decay_t<ExprType>>> 
        >
inline auto autodiff(core::ExprBind<ExprType>& expr,
                     const Eigen::ArrayBase<T>& seed)
{
    return autodiff(expr.get(), seed);
}

template <class ExprType
        , class = std::enable_if_t<util::is_scl_v<std::decay_t<ExprType>>> 
        >
inline auto autodiff(core::ExprBind<ExprType>&& expr,
                     typename util::expr_traits<
                        std::decay_t<ExprType>>::value_t seed = 1.)
{
    return autodiff(expr.get(), seed);
}

template <class ExprType
        , class T
        , class = std::enable_if_t<!util::is_scl_v<std::decay_t<ExprType>>> 
        >
inline auto autodiff(core::ExprBind<ExprType>&& expr,
                     const Eigen::ArrayBase<T>& seed)
{
    return autodiff(expr.get(), seed);
}

} // namespace ad
