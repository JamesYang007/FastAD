#pragma once
#include <iterator>
#include <fastad_bits/expr_base.hpp>
#include <fastad_bits/value_view.hpp>
#include <fastad_bits/constant.hpp>
#include <fastad_bits/type_traits.hpp>

namespace ad {
namespace core {

/** 
 * SumIterNode represents a summation of an arbitrary function on many expressions.
 * Ex.
 * f(x1) + f(x2) + ... + f(xn)
 * Mathematically, the derivative of this expression can be optimized
 * since the partial derivative w.r.t. xi is simply f'(xi)
 * and does not depend on other xj values.
 *
 * @tparam  Iter        type of iterator of values.
 * @tparam  Lmda        type of functor to apply on each iterated value returning an expression.
 */

template <class Iter, class Lmda>
struct SumIterNode:
    ValueView<typename util::expr_traits< std::decay_t< std::invoke_result_t<
                Lmda, typename std::iterator_traits<Iter>::value_type>
                      > >::value_t,
              typename util::shape_traits< std::decay_t< std::invoke_result_t<
                Lmda, typename std::iterator_traits<Iter>::value_type>
                      > >::shape_t >,
    ExprBase<SumIterNode<Iter, Lmda>>
{
private:
    using iter_value_t = typename std::iterator_traits<Iter>::value_type;
    using vec_elem_t = std::decay_t< std::invoke_result_t<
            Lmda, iter_value_t> >;
    using elem_value_t = typename util::expr_traits<vec_elem_t>::value_t;
    using elem_shape_t = typename util::shape_traits<vec_elem_t>::shape_t;
    
public:
    using value_view_t = ValueView<elem_value_t, elem_shape_t>;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::bind;

    SumIterNode(Iter start, Iter end, Lmda f)
        : value_view_t(nullptr,
                       (start == end) ? 0 : f(*start).rows(),
                       (start == end) ? 0 : f(*start).cols())
        , exprs_{}
    {
        exprs_.reserve(std::distance(start, end));
        std::for_each(start, end,
            [&](const auto& val) {
                exprs_.emplace_back(f(val));
            });
    }

    /** 
     * Forward evaluate by evaluating every expression left to right
     * and accumulating the results.
     *
     * @return forward evaluation of sum of functor on every expr.
     */
    const var_t& feval()
    {
        this->zero();
        for (auto& expr : exprs_) {
            this->get() += expr.feval();
        }
        return this->get();
    }

    /** 
     * Backward evaluate from right to left by seeding the same seed.
     */
    void beval(value_t seed, size_t i, size_t j)
    {
        std::for_each(exprs_.rbegin(), exprs_.rend(),
            [=](auto& expr) {
                expr.beval(seed, i, j);
            });
    }

    /**
     * Bind ever expression from left to right then bind itself.
     *
     * @return  the next pointer not bound by any of the expressions and itself.
     */
    value_t* bind(value_t* begin)
    {
        if constexpr (!util::is_var_view_v<vec_elem_t>) {
            for (auto& expr : exprs_) begin = expr.bind(begin);
        }
        return value_view_t::bind(begin);
    }

private:
    std::vector<vec_elem_t> exprs_;
};

} // namespace core

/**
 * Helper function to create a SumIterNode.
 * If there are no expressions to iterate over, 
 * and if each expression types are constant,
 * then returns a constant of 0 scalar, or empty Eigen vector/matrix.
 * If not constant, then SumIterNode is returned that will effectively be a noop.
 * Otherwise, if there is at least one expression to iterate over, 
 * returns a SumIterNode that will not be a noop.
 */
template <class Iter, class Lmda>
inline auto sum(Iter begin, Iter end, Lmda&& f)
{
    using node_t = std::decay_t<decltype(f(*begin))>;
    using value_t = typename util::expr_traits<node_t>::value_t;
    using shape_t = typename util::shape_traits<node_t>::shape_t;
    using var_t = core::details::constant_var_t<value_t, shape_t>;

    // optimized for f that returns a constant node
    if constexpr (util::is_constant_v<node_t>) {
        if (std::distance(begin, end) <= 0) return ad::constant(var_t(0));
        var_t sum = f(*begin).feval(); 
        std::for_each(std::next(begin), end, 
                [&](const auto& x) 
                { sum += f(x).feval(); });
        return ad::constant(sum);
    } else {
        return core::SumIterNode<Iter, Lmda>(
                begin, end, std::forward<Lmda>(f));
    }
}

} // namespace ad
