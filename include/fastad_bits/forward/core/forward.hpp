#pragma once
#include <cmath>
#include <fastad_bits/forward/core/dualnum.hpp>

// Forward-mode Automatic Differentiation

// Unary function definition with function name "f" that operates on ADForward<T> variable x.
// "first" represents code that computes the unary function on x value.
// "second" represents code that computes the directional derivative of unary function on x value.
// The variadic arguments are optional and represent code to be placed before executing
// "first" and "second" for optimization purposes.
// @tparam  T   underlying data type for x.
// @param   x   variable to apply unary function to
// @return a new ADForward<T> with value and adjoint as the mathematical values for f(x), f'(x) * x'.
//
// Example generation with no variadic arguments:
//
// FORWARD_UNARY_FUNC(sin, std::sin(x.get_value()), std::cos(x.get_value()) * x.get_adjoint())
// =>
// template <class T> 
// inline auto sin(const ad::core::ADForward<T>& x) 
// { 
//      return ad::core::ADForward<T>(std::sin(x.get_value()), std::cos(x.get_value()) * x.get_adjoint()); 
// } 
//
// Example generation with variadic arguments:
//
// FORWARD_UNARY_FUNC(exp, tmp, tmp * x.get_adjoint(), auto tmp = std::exp(x.get_value());)
// =>
// template <class T> 
// inline auto exp(const ad::core::ADForward<T>& x) 
// { 
//      auto tmp = std::exp(x.get_value());
//      return ad::core::ADForward<T>(tmp, tmp * x.get_adjoint()); 
// } 
//
// Note that we only compute std::exp(x.get_value()) once and reuse to compute both "first" and "second".
#define FORWARD_UNARY_FUNC(f, first, second, ...) \
template <class T> \
inline auto f(const ad::core::ADForward<T>& x) \
{ \
	__VA_ARGS__ \
	return ad::core::ADForward<T>(first, second); \
} \

// Binary function definition with function name "f" that operates on ADForward<T> variables x, y.
// "first" represents code that computes the binary function on x, y values.
// "second" represents code that computes the directional derivative of binary function on x, y values
// in the direction of (x.get_adjoint(), y.get_adjoint()).
// The variadic arguments are optional and represent code to be placed before executing
// "first" and "second" for optimization purposes.
// @tparam  T   underlying data type for x.
// @param   x   one of the variables to apply binary function to
// @param   y   other variable to apply binary function to
// @return a new ADForward<T> with value and adjoint as the mathematical values for f(x, y), f'(x, y) * (x', y').
//
// Example generation with no variadic arguments:
//
// FORWARD_BINARY_FUNC(operator+, x.get_value() + y.get_value(), x.get_adjoint() + y.get_adjoint())
// =>
// template <class T>
// inline auto operator+(const ad::core::ADForward<T>& x, ad::core::ADForward<T>& y)
// {
//      return ad::core::ADForward<T>(x.get_value() + y.get_value(), x.get_adjoint() + y.get_adjoint());
// }
#define FORWARD_BINARY_FUNC(f, first, second) \
template <class T> \
inline auto f(const ad::core::ADForward<T>& x, const ad::core::ADForward<T>& y) \
{ \
	return ad::core::ADForward<T>(first, second); \
} \

namespace ad {
namespace core {

// Forward variable to store value and adjoint.
// If x is an ADForward variable that is a result of composing functions of ADForward variables x1,...,xn
// x.get_value() is the value of the function on these variables and x.get_adjoint() is the adjoint, i.e.
// directional (total) derivative of the composed functions in the direction of x1.get_adjoint(),...,xn.get_adjoint()
template <class T>
struct ADForward : public core::DualNum<T>
{
    using data_t = core::DualNum<T>;

    ADForward()
        : data_t(0, 0)
    {}

    ADForward(T w, T df = 0)
        : data_t(w, df)
    {}

    ADForward& operator+=(const ADForward& x);
};

} // namespace core

// user-exposed forward variable alias 
template <class T>
using ForwardVar = core::ADForward<T>;

//================================================================================

// Unary functions 

// Negate forward variable
FORWARD_UNARY_FUNC(operator-, -x.get_value(), -x.get_adjoint())
// ad::sin(core::ADForward)
FORWARD_UNARY_FUNC(sin, std::sin(x.get_value()), 
        std::cos(x.get_value())*x.get_adjoint())
// ad::cos(core::ADForward)
FORWARD_UNARY_FUNC(cos, std::cos(x.get_value()), 
        -std::sin(x.get_value())*x.get_adjoint())
// ad::tan(core::ADForward)
FORWARD_UNARY_FUNC(tan, std::tan(x.get_value()), 
        tmp*tmp * x.get_adjoint(), auto tmp = 1 / std::cos(x.get_value());)
// ad::asin(core::ADForward)
FORWARD_UNARY_FUNC(asin, std::asin(x.get_value()), 
        x.get_adjoint() / (std::sqrt(1 - x.get_value()*x.get_value())))
// ad::acos(core::ADForward)
FORWARD_UNARY_FUNC(acos, std::acos(x.get_value()), 
        -x.get_adjoint() / (std::sqrt(1 - x.get_value()*x.get_value())))
// ad::atan(core::ADForward)
FORWARD_UNARY_FUNC(atan, std::atan(x.get_value()), 
        x.get_adjoint() / (1 + x.get_value()*x.get_value()))
// ad::exp(core::ADForward)
FORWARD_UNARY_FUNC(exp, tmp, tmp * x.get_adjoint(), 
        auto tmp = std::exp(x.get_value());)
// ad::log(core::ADForward)
FORWARD_UNARY_FUNC(log, std::log(x.get_value()), 
        x.get_adjoint() / x.get_value())
// ad::sqrt(core::ADForward)
FORWARD_UNARY_FUNC(sqrt, tmp, x.get_adjoint() / (2 * tmp), 
        auto tmp = std::sqrt(x.get_value());)
// ad::erf(core::ADForward)
FORWARD_UNARY_FUNC(erf, std::erf(x.get_value()), 
        two_over_sqrt_pi * std::exp(-t_sq), 
        static constexpr double two_over_sqrt_pi =
                1.1283791670955126;
        auto t_sq = x.get_value() * x.get_value();)

//================================================================================

// Binary operators 

namespace core {

// Add forward variables
FORWARD_BINARY_FUNC(operator+, x.get_value() + y.get_value(), 
                    x.get_adjoint() + y.get_adjoint())
// Subtract forward variables
FORWARD_BINARY_FUNC(operator-, x.get_value() - y.get_value(), 
                    x.get_adjoint() - y.get_adjoint())
// Multiply forward variables
FORWARD_BINARY_FUNC(operator*, x.get_value() * y.get_value(), 
        x.get_value() * y.get_adjoint() + x.get_adjoint() * y.get_value())
// Divide forward variables
FORWARD_BINARY_FUNC(operator/, x.get_value() / y.get_value(), 
        (x.get_adjoint() * y.get_value() - x.get_value() * y.get_adjoint()) / (y.get_value() * y.get_value()))

// Add current forward variable with x and update current variable with the result.
template <class T>
inline ADForward<T>& ADForward<T>::operator+=(const ADForward<T>& x)
{
    return *this = *this + x;
}

} // namespace core
} // namespace ad
