#pragma once
#include <cmath>
#include <limits>
#include <fastad_bits/node.hpp>

namespace ad {
namespace core {

/*
 * Utility pow function
 */

// Primary definition: exp > 0
template <int64_t exp, bool = (exp >= 0)>
struct PowFunc
{
    template <class BaseType>
    static auto evaluate(BaseType base)
    {
        if constexpr (exp % 2 == 0) {
            double tmp = PowFunc<exp/2>::evaluate(base);
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
    static auto evaluate(BaseType)
    {
        return 1.;
    }
};

// Specialization: exp < 0
template <int64_t n>
struct PowFunc<n, false>
{
    template <class BaseType>
    static auto evaluate(BaseType base)
    {
        return (base == 0) ? 
            std::numeric_limits<BaseType>::max() : 
            PowFunc<-n>::evaluate(1./base);
    }
};

/*
 * An AD expression to represent integral powers
 */
template <int64_t exp, class ADExprType>
struct PowNode : 
    ADExprType::data_t,
    ADNodeExpr<PowNode<exp, ADExprType>>
{
    using data_t = typename ADExprType::data_t;
    using value_type = typename data_t::value_type;

    PowNode(const ADExprType& expr)
        : data_t{0., 0.}
        ,  expr_{expr}
    {}

    value_type feval()
    {
        return this->set_value(
                PowFunc<exp_>::evaluate(expr_.feval()) ); 
    }

    void beval(value_type seed)
    {
        this->set_adjoint(seed);
        if (expr_.get_value() == 0) {
            expr_.beval(0);
        } else {
            expr_.beval(seed * exp_ * this->get_value() / expr_.get_value());
        }
    }

private:
    ADExprType expr_;
    static constexpr int64_t exp_ = exp;
};

} // namespace core

// Undefined behavior if expression evaluates to 0
// and backward-evaluates during AD.
template <int64_t exp, class ADExprType>
inline constexpr auto pow(const ADExprType& expr)
{
    return core::PowNode<exp, ADExprType>(expr);
}

} // namespace ad
