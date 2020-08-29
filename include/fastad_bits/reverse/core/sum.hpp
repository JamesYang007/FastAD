#pragma once
#include <iterator>
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_adj_view.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/util/size_pack.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/util/value.hpp>

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
 * @tparam  VecType     type of vector of expressions to sum over 
 */

template <class VecType>
struct SumIterNode:
    ValueAdjView<typename util::expr_traits< 
                    typename VecType::value_type >::value_t,
                 typename util::shape_traits< 
                    typename VecType::value_type >::shape_t >,
    ExprBase<SumIterNode<VecType>>
{
private:
    using vec_elem_t = typename VecType::value_type;
    using elem_value_t = typename util::expr_traits<vec_elem_t>::value_t;
    using elem_shape_t = typename util::shape_traits<vec_elem_t>::shape_t;
    
public:
    using value_adj_view_t = ValueAdjView<elem_value_t, elem_shape_t>;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    SumIterNode(const VecType& exprs)
        : value_adj_view_t(nullptr, nullptr,
                       (exprs.size() == 0) ? 0 : exprs[0].rows(),
                       (exprs.size() == 0) ? 0 : exprs[0].cols())
        , exprs_{exprs}
    {}

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
     * Simply reuse last expression's adjoint since it is guaranteed to be the same as seed.
     * The point is that seed may be an Eigen expression and should be evaluated first.
     * Then we reuse the evaluated values, rather than reusing the expression, 
     * which will lead to multiple evaluations.
     *
     * Note: this may break some cases I'm not sure...
     */
    template <class T>
    void beval(const T& seed)
    {
        if (exprs_.empty()) return;
        auto&& a_adj = util::to_array(this->get_adj());
        a_adj = seed;
        std::for_each(exprs_.rbegin(), exprs_.rend(),
            [&](auto& expr) {
                expr.beval(a_adj);
            });
    }

    /**
     * Bind every expression from left to right then bind itself.
     *
     * @return  the next pointer not bound by any of the expressions and itself.
     */
    ptr_pack_t bind_cache(ptr_pack_t begin)
    {
        for (auto& expr : exprs_) {
            begin = expr.bind_cache(begin);
        }
        return value_adj_view_t::bind(begin);
    }

    util::SizePack bind_cache_size() const 
    { 
        util::SizePack out = util::SizePack::Zero();
        for (const auto& expr : exprs_) {
            out += expr.bind_cache_size();
        }
        return out + single_bind_cache_size();
    }

    util::SizePack single_bind_cache_size() const
    { 
        return {this->size(), this->size()}; 
    }

private:
    std::vector<vec_elem_t> exprs_;
};

/** 
 * SumElemNode represents a summation of all elements of an expression.
 * Ex. \sum_{i,j=1}^{m,n} e_{ij}
 *
 * Mathematically, the derivative of this expression can be optimized
 * since the partial derivative w.r.t. e_ij is simply e'_ij
 * and does not depend on other e_ij values.
 *
 * @tparam  VecType     type of vector of expressions to sum over 
 */

template <class ExprType>
struct SumElemNode:
    ValueAdjView<typename util::expr_traits<ExprType>::value_t,
                 ad::scl>,
    ExprBase<SumElemNode<ExprType>>
{
private:
    using expr_t = ExprType;
    using expr_value_t = typename util::expr_traits<expr_t>::value_t;
    
public:
    using value_adj_view_t = ValueAdjView<expr_value_t, ad::scl>;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    SumElemNode(const expr_t& expr)
        : value_adj_view_t(nullptr, nullptr, 1, 1)
        , expr_{expr}
    {}

    /** 
     * Forward evaluate by evaluating the expression and accumulating the elements.
     *
     * @return forward evaluation of sum of functor on every expr.
     */
    const var_t& feval()
    {
        auto&& res = expr_.feval();
        if constexpr (util::is_scl_v<expr_t>) {
            return this->get() = res;
        } else {
            return this->get() = res.sum();
        }
    }

    /** 
     * Backward evaluate every element of expression with same seed.
     * Note that since this node is always scalar, beval does not have to be templated.
     */
    void beval(value_t seed)
    {
        expr_.beval(seed);
    }

    /**
     * Bind the expression then itself to a scalar.
     *
     * @return  the next pointer pack not bound by any of the expressions and itself.
     */
    ptr_pack_t bind_cache(ptr_pack_t begin)
    {
        begin = expr_.bind_cache(begin);
        auto adj = begin.adj;
        begin.adj = nullptr;
        begin = value_adj_view_t::bind(begin);
        begin.adj = adj;
        return begin;
    }

    util::SizePack bind_cache_size() const 
    { 
        return expr_.bind_cache_size() + 
                single_bind_cache_size();
    }

    util::SizePack single_bind_cache_size() const
    { 
        return {this->size(), 0}; 
    }

private:
    expr_t expr_;
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
    using expr_t = std::decay_t<decltype(f(*begin))>;
    using value_t = typename util::expr_traits<expr_t>::value_t;
    using shape_t = typename util::shape_traits<expr_t>::shape_t;
    using var_t = util::constant_var_t<value_t, shape_t>;

    // optimized for f that returns a constant node
    if constexpr (util::is_constant_v<expr_t>) {
        if (std::distance(begin, end) <= 0) return ad::constant(var_t(0));
        var_t sum = f(*begin).feval(); 
        std::for_each(std::next(begin), end, 
                [&](const auto& x) 
                { sum += f(x).feval(); });
        return ad::constant(sum);
    } else {
        std::vector<expr_t> exprs;
        exprs.reserve(std::distance(begin, end));
        std::for_each(begin, end, 
                [&](const auto& x) {
                    exprs.emplace_back(f(x));
                });
        return core::SumIterNode<std::vector<expr_t>>(exprs);
    }
}

template <class Derived
        , class = std::enable_if_t<
            util::is_convertible_to_ad_v<Derived> &&
            util::any_ad_v<Derived> > >
inline auto sum(const Derived& x)
{
    using expr_t = util::convert_to_ad_t<Derived>;
    expr_t expr = x;

    // optimized when expr is constant
    if constexpr (util::is_constant_v<expr_t>) {
        if constexpr (util::is_scl_v<expr_t>) return expr;
        else {
            return ad::constant(expr.feval().array().sum());
        }
    } else {
        return core::SumElemNode<expr_t>(expr);
    }
}

} // namespace ad
