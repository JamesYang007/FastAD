#pragma once
#include <cstdint>
#include <cmath>
#include <limits>
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/reverse/core/value_adj_view.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/util/size_pack.hpp>
#include <fastad_bits/util/value.hpp>

namespace ad {
namespace core {

/*
 * Utility pow function for scalars.
 */

// Primary definition: exp > 0
template <int64_t exp, bool = (exp >= 0)>
struct PowFunc
{
    template <class BaseType>
    static auto evaluate(const BaseType& base)
    {
        if constexpr (exp % 2 == 0) {
            auto tmp = PowFunc<exp/2>::evaluate(base);
            return tmp * tmp;
        } else {
            return base * PowFunc<exp-1>::evaluate(base);
        }
    }
};

// Specialization: exp == 0
// Note by convention, we take 0^0 to be 1.
template <>
struct PowFunc<0, true>
{
    template <class BaseType>
    constexpr static auto evaluate(const BaseType&)
    { return 1.; }
};

// Specialization: exp < 0
template <int64_t n>
struct PowFunc<n, false>
{
    template <class BaseType>
    static auto evaluate(const BaseType& base)
    {
        return (base == 0) ? 
            std::numeric_limits<BaseType>::infinity() : 
            PowFunc<-n>::evaluate(1./base);
    }
};

/**
 * PowNode represents a power function on an expression with exponent exp.
 * This exp must be known at compile-time.
 * It is a vectorized operation.
 *
 * @tparam  exp         exponent to raise by
 * @tparam  ExprType    expression to exponentiate 
 */

template <int64_t exp, class ExprType>
struct PowNode : 
    ValueAdjView<typename util::expr_traits<ExprType>::value_t, 
                 typename util::shape_traits<ExprType>::shape_t>,
    ExprBase<PowNode<exp, ExprType>>
{
private:
    using expr_t = ExprType;
    static_assert(util::is_expr_v<expr_t>);

public:
    using value_adj_view_t = ValueAdjView<
        typename util::expr_traits<expr_t>::value_t, 
        typename util::shape_traits<expr_t>::shape_t >;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    PowNode(const expr_t& expr)
        : value_adj_view_t(nullptr, nullptr, expr.rows(), expr.cols())
        , expr_{expr}
    {}

    const var_t& feval()
    {
        if constexpr (util::is_scl_v<expr_t>) {
            return this->get() =
                    PowFunc<exp_>::evaluate(expr_.feval()); 
        } else {
            return this->get() = expr_.feval().array().pow(exp_); 
        }
    }

    template <class T>
    void beval(const T& seed)
    {
        static_cast<void>(seed);

        // derivative of x^0 = c is 0
        if constexpr (exp == 0) {
            expr_.beval(0.);

        // derivative of x^1 is 1
        } else if constexpr (exp == 1) {
            expr_.beval(seed);

        // derivative of x^n when n < 0 is nx^(n-1)
        // we only need to be careful when x == 0
        } else {
            auto&& a_val = util::to_array(this->get());
            auto&& a_adj = util::to_array(this->get_adj());
            auto&& a_expr = util::to_array(expr_.get());

            a_adj = seed;

            using a_expr_t = std::decay_t<decltype(a_expr)>;

            if constexpr (exp > 1) {

                auto correct_seed = [&]() {
                    if constexpr (util::is_eigen_v<a_expr_t>) {
                        return a_expr_t::NullaryExpr(a_expr.rows(), a_expr.cols(),
                                [&](size_t i, size_t j) { 
                                    return a_expr(i,j) == 0 ? 
                                        0 : a_val(i,j) / a_expr(i,j);
                                });
                    } else {
                        return a_expr == 0 ? 0 : a_val / a_expr;
                    }
                };
                auto corrected_seed = exp * a_adj * correct_seed();

                expr_.beval(corrected_seed);

            } else {

                auto correct_seed = [&]() {
                    if constexpr (util::is_eigen_v<a_expr_t>) {
                        return a_expr_t::NullaryExpr(a_expr.rows(), a_expr.cols(),
                                [&](size_t i, size_t j) {
                                    return a_expr(i,j) == 0 ? 
                                        -std::numeric_limits<value_t>::infinity():
                                        exp * a_adj(i,j) * a_val(i,j) / a_expr(i,j);
                                });
                    } else {
                        return a_expr == 0 ? 
                                -std::numeric_limits<value_t>::infinity():
                                exp * a_adj * a_val / a_expr;
                    }
                };
                expr_.beval(correct_seed());

            }
        } 
    }

    ptr_pack_t bind_cache(ptr_pack_t begin)
    { 
        begin = expr_.bind_cache(begin);
        if constexpr (exp == 0 || exp == 1) {
            auto adj = begin.adj;
            begin.adj = nullptr;
            begin = value_adj_view_t::bind(begin);
            begin.adj = adj;
            return begin;
        } else {
            return value_adj_view_t::bind(begin);
        }
    }

    util::SizePack bind_cache_size() const 
    { 
        return single_bind_cache_size() + 
                expr_.bind_cache_size();
    }

    util::SizePack single_bind_cache_size() const
    {
        if constexpr (exp == 0 || exp == 1) {
            return {this->size(), 0};
        } else {
            return {this->size(), this->size()};
        }
    }

private:
    expr_t expr_;
    static constexpr int64_t exp_ = exp;
};

} // namespace core

/**
 * Helper function to generate PowNode of an expression.
 * If expression evaluates to 0 during back-evaluation,
 * and exp is less than 0, the seed will pass -infinity.
 */
template <int64_t exp
        , class Derived
        , class = std::enable_if_t<
            util::is_convertible_to_ad_v<Derived> &&
            util::any_ad_v<Derived> > >
inline constexpr auto pow(const Derived& x)
{
    using expr_t = util::convert_to_ad_t<Derived>;
    using value_t = typename util::expr_traits<expr_t>::value_t;
    using shape_t = typename util::shape_traits<expr_t>::shape_t;
    using var_t = util::constant_var_t<value_t, shape_t>;
    
    expr_t expr = x;

    if constexpr (util::is_constant_v<expr_t>) {
        if constexpr (util::is_scl_v<expr_t>) {
            return ad::constant(
                    core::PowFunc<exp>::evaluate(expr.feval())
                    ); 
        } else {
            var_t out = expr.feval().array().pow(exp);
            return ad::constant(out); 
        }
    } else {
        return core::PowNode<exp, expr_t>(expr);
    }
}

} // namespace ad
