#pragma once
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_view.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/reverse/core/constant.hpp>

namespace ad {
namespace core {

/**
 * VecNormNode represents the (squared) norm of a vector.
 * Currently, we do not support the feature for row vectors.
 * Indeed, the expression a vector shape.
 * No other shapes are permitted for this node.
 *
 * The node assumes the same value type as that of the vector expression.
 * It is always a scalar shape.
 *
 * We assert that the value type be the same for the two expressions.
 * The output shape is always a (column) vector.
 *
 * @tparam  VecExprType     type of vector expression
 */

template <class VecExprType>
struct VecNormNode:
    ValueView<typename util::expr_traits<VecExprType>::value_t,
              ad::scl>,
    ExprBase<VecNormNode<VecExprType>>
{
private:
    using expr_t = VecExprType;
    using expr_value_t = typename util::expr_traits<expr_t>::value_t;

    static_assert(util::is_vec_v<VecExprType>);

public:
    using value_view_t = ValueView<expr_value_t, ad::scl>;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::bind;

    VecNormNode(const expr_t& expr)
        : value_view_t(nullptr, 1, 1)
        , expr_{expr}
    {}

    const var_t& feval()
    {
        auto&& res = expr_.feval();
        return this->get() = res.squaredNorm();
    }

    // current implementation assumes that expr_ is a column vector
    void beval(value_t seed, size_t, size_t, util::beval_policy pol)
    {
        if (seed == 0) return;
        for (size_t k = 0; k < expr_.rows(); ++k) {
            expr_.beval(seed * 2. * expr_.get(k,0), k, 0, pol);
        }
    }

    value_t* bind(value_t* begin)
    {
        if constexpr (!util::is_var_view_v<expr_t>) {
            begin = expr_.bind(begin);
        }
        return value_view_t::bind(begin);
    }

    size_t bind_size() const 
    { 
        return expr_.bind_size() + single_bind_size();
    }

    constexpr size_t single_bind_size() const 
    { 
        return this->size(); 
    }

private:
    expr_t expr_;
};

} // namespace core

template <class T
        , class = std::enable_if_t<
            util::is_convertible_to_ad_v<T> &&
            util::any_ad_v<T> > >
inline auto norm(const T& x)
{
    using expr_t = util::convert_to_ad_t<T>;
    using value_t = typename util::expr_traits<expr_t>::value_t;

    expr_t expr = x;

    // optimization for when expression is constant
    if constexpr (util::is_constant_v<expr_t>) {
        static_assert(util::is_vec_v<expr_t>);
        using var_t = core::details::constant_var_t<value_t, ad::scl>;
        var_t out = expr.feval().squaredNorm();
        return ad::constant(out);
    } else {
        return core::VecNormNode<expr_t>(expr);
    }
}

} // namespace ad
