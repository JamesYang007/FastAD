#pragma once
#include <cstdint>
#include <cmath>
#include <limits>
#include <fastad_bits/expr_base.hpp>
#include <fastad_bits/type_traits.hpp>
#include <fastad_bits/constant.hpp>
#include <fastad_bits/value_view.hpp>

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
    ValueView<typename util::expr_traits<ExprType>::value_t, 
              typename util::shape_traits<ExprType>::shape_t>,
    ExprBase<PowNode<exp, ExprType>>
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

    PowNode(const expr_t& expr)
        : value_view_t(nullptr, expr.rows(), expr.cols())
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

    void beval(value_t seed, size_t i, size_t j, util::beval_policy pol)
    {
        if (seed == 0) return;

        // derivative of x^0 = c is 0
        if constexpr (exp == 0) {
            expr_.beval(0, i, j, pol);

        // derivative of x^1 is 1
        } else if constexpr (exp == 1) {
            expr_.beval(seed, i, j, pol);

        // derivative of x^n when n < 0 is nx^(n-1)
        // we only need to be careful when x == 0
        } else if constexpr (exp < 0) {
            if (expr_.get(i,j) == 0) {
                value_t next_seed =
                    -std::numeric_limits<value_t>::infinity();
                expr_.beval(next_seed, i, j, pol);
            } else {
                value_t next_seed = seed * exp * 
                    this->get(i,j) / expr_.get(i,j);
                expr_.beval(next_seed, i, j, pol);
            }

        // derivative of x^n when n > 1 is nx^(n-1)
        // we only need to be careful when x == 0
        } else {
            if (expr_.get(i,j) == 0) {
                expr_.beval(0, i, j, pol);
            } else {
                value_t next_seed = seed * exp * 
                    this->get(i,j) / expr_.get(i,j);
                expr_.beval(next_seed, i, j, pol);
            }
        }
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
    static constexpr int64_t exp_ = exp;
};

} // namespace core

/**
 * Helper function to generate PowNode of an expression.
 * If expression evaluates to 0 during back-evaluation,
 * and exp is less than 0, the seed will pass -infinity.
 */
template <int64_t exp, class Derived>
inline constexpr auto pow(const core::ExprBase<Derived>& expr)
{
    using expr_t = Derived;
    using value_t = typename util::expr_traits<expr_t>::value_t;
    using shape_t = typename util::shape_traits<expr_t>::shape_t;
    using var_t = core::details::constant_var_t<value_t, shape_t>;

    if constexpr (util::is_constant_v<Derived>) {
        if constexpr (util::is_scl_v<Derived>) {
            return ad::constant(
                    core::PowFunc<exp>::evaluate(expr.self().feval())
                    ); 
        } else {
            var_t out = expr.self().feval().array().pow(exp);
            return ad::constant(out); 
        }
    } else {
        return core::PowNode<exp, Derived>(expr.self());
    }
}

} // namespace ad
