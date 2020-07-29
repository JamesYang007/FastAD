#pragma once
#include <Eigen/Core>
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/util/shape_traits.hpp>
#include <fastad_bits/reverse/core/value_view.hpp>
#include <fastad_bits/forward/core/forward.hpp>

namespace ad {
namespace core {

template <class ExprType>
struct SqrtNode:
    ValueView<typename util::expr_traits<ExprType>::value_t,
              typename util::shape_traits<ExprType>::shape_t>,
    ExprBase<SqrtNode<ExprType>>
{
private:
    using expr_t = ExprType;
    static_assert(util::is_expr_v<expr_t>);

public:
    using value_view_t = ValueView<
        typename util::expr_traits<expr_t>::value_t, 
        typename util::shape_traits<expr_t>::shape_t >;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::bind;

    SqrtNode(const expr_t& expr)
        : value_view_t(nullptr, expr.rows(), expr.cols())
        , expr_(expr)
    {}

    const var_t& feval()
    {
        if constexpr (util::is_scl_v<expr_t>) {
            return this->get() = std::sqrt(expr_.feval());
        } else {
            return this->get() = Eigen::sqrt(expr_.feval().array()).matrix();
        }
    }

    void beval(value_t seed, size_t i, size_t j, util::beval_policy pol)
    {
        if (seed == 0) return;
        expr_.beval(seed / (2. * this->get(i,j)), i, j, pol);
    }

    value_t* bind(value_t* begin)
    { 
        value_t* next = begin;
        if constexpr (!util::is_var_view_v<expr_t>) {
            next = expr_.bind(next);
        }
        return value_view_t::bind(next);
    }

    size_t bind_size() const 
    { 
        return single_bind_size() + expr_.bind_size();
    }

    size_t single_bind_size() const
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
            util::any_ad_v<T> >>
inline auto sqrt(const T& x)
{
    using expr_t = util::convert_to_ad_t<T>;
    using value_t = typename util::expr_traits<expr_t>::value_t;
    using shape_t = typename util::shape_traits<expr_t>::shape_t;
    using var_t = core::details::constant_var_t<value_t, shape_t>;

    expr_t expr = x;

    if constexpr (util::is_constant_v<expr_t>) {
        if constexpr (util::is_scl_v<expr_t>) {
            return ad::constant(std::sqrt(expr.get()));
        } else {
            var_t v = expr.get().array().sqrt();
            return ad::constant(v);
        }
    }
    return core::SqrtNode<expr_t>(expr);
}

} // namespace ad
