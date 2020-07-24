#pragma once
#include <fastad_bits/expr_base.hpp>
#include <fastad_bits/value_view.hpp>
#include <fastad_bits/type_traits.hpp>

namespace ad {
namespace core {

/** 
 * ForEachIterNode represents collection of expressions to evaluate.
 * It can be thought of as a generalization of GlueNode.
 * Applies functor on every iterated values as an expression.
 *
 * @tparam  VecType     type of vector of expressions to for-each over 
 */

template <class VecType>
struct ForEachIterNode: 
    ValueView<typename util::expr_traits< 
                typename VecType::value_type >::value_t,
              typename util::shape_traits< 
                typename VecType::value_type >::shape_t >,
    ExprBase<ForEachIterNode<VecType>>
{
private:
    using vec_elem_t = typename VecType::value_type;
    using elem_value_t = typename util::expr_traits<vec_elem_t>::value_t;
    using elem_shape_t = typename util::shape_traits<vec_elem_t>::shape_t;

public:
    using value_view_t = ValueView<elem_value_t, elem_shape_t>;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::bind;

    ForEachIterNode(const VecType& vec)
        : value_view_t(nullptr,
                       (vec.size() == 0) ? 0 : vec[0].rows(),
                       (vec.size() == 0) ? 0 : vec[0].cols())
        , vec_(vec)
    {}

    /** 
     * Forward evaluation on every functored expressions.
     * Note that if there are no expressions to evaluate,
     * it is undefined behavior doing any computations with the return value.
     * If there is at least one expression, then it will return
     * the last cached forward evaluation.
     *
     * @return  last functored expression forward evaluation value
     */
    const var_t& feval()
    {
        if (vec_.size() == 0) { return this->get(); }
        std::for_each(vec_.begin(), vec_.end(), 
                [](auto& expr) { expr.feval(); }
        );
        return this->get() = vec_.back().get();
    }

    /**
     * Backward evaluation seeds the last functored expression with seed,
     * and backward evaluates every expression in reverse order with 0 seed.
     * See GlueNode::beval for reasons for this design choice.
     */
    void beval(value_t seed, size_t i, size_t j)
    {
        if (vec_.size() == 0) return;
        auto it = vec_.rbegin();
        it->beval(seed, i, j);
        std::for_each(std::next(it), vec_.rend(), 
                [=](auto& expr) {
                    expr.beval(0, i, j); 
                }
        );
    }

    /**
     * Bind every expression from left to right then bind itself
     * to the last expression.
     *
     * @return  the next pointer not bound by any of the expressions and itself.
     */
    value_t* bind(value_t* begin)
    {
        if (vec_.size() == 0) return begin;
        if constexpr (!util::is_var_view_v<vec_elem_t>) {
            for (auto& expr : vec_) begin = expr.bind(begin);
        }
        value_view_t::bind(vec_.back().data());
        return begin;
    }

    size_t bind_size() const 
    { 
        size_t out = 0;
        for (const auto& expr : vec_) out += expr.bind_size();
        return out;
    }

    constexpr size_t single_bind_size() const { return 0; }

private:
    std::vector<vec_elem_t> vec_;
};

} // namespace core

/**
 * Helper function to create a ForEachIterNode.
 */
template <class Iter, class Lmda>
inline auto for_each(Iter begin, Iter end, Lmda f)
{
    using expr_t = std::decay_t<decltype(f(*begin))>;
    std::vector<expr_t> exprs;
    exprs.reserve(std::distance(begin, end));
    std::for_each(begin, end, 
            [&](const auto& x) {
                exprs.emplace_back(f(x));
            });
    return core::ForEachIterNode<std::vector<expr_t>>(exprs);
}

} // namespace ad
