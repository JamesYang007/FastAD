#pragma once

// Forward-mode Automatic Differentiation

#include <cmath>
#include "dualnum.hpp"

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
// FORWARD_UNARY_FUNC(sin, std::sin(x.w), std::cos(x.w) * x.df)
// =>
// template <class T> 
// inline auto sin(ad::core::ADForward<T> const& x) 
// { 
//      return ad::core::ADForward<T>(std::sin(x.w), std::cos(x.w) * x.df); 
// } 
//
// Example generation with variadic arguments:
//
// FORWARD_UNARY_FUNC(exp, tmp, tmp * x.df, auto tmp = std::exp(x.w);)
// =>
// template <class T> 
// inline auto exp(ad::core::ADForward<T> const& x) 
// { 
//      auto tmp = std::exp(x.w);
//      return ad::core::ADForward<T>(tmp, tmp * x.df); 
// } 
//
// Note that we only compute std::exp(x.w) once and reuse to compute both "first" and "second".
#define FORWARD_UNARY_FUNC(f, first, second, ...) \
template <class T> \
inline auto f(ad::core::ADForward<T> const& x) \
{ \
	__VA_ARGS__ \
	return ad::core::ADForward<T>(first, second); \
} \

// Binary function definition with function name "f" that operates on ADForward<T> variables x, y.
// "first" represents code that computes the binary function on x, y values.
// "second" represents code that computes the directional derivative of binary function on x, y values
// in the direction of (x.df, y.df).
// The variadic arguments are optional and represent code to be placed before executing
// "first" and "second" for optimization purposes.
// @tparam  T   underlying data type for x.
// @param   x   one of the variables to apply binary function to
// @param   y   other variable to apply binary function to
// @return a new ADForward<T> with value and adjoint as the mathematical values for f(x, y), f'(x, y) * (x', y').
//
// Example generation with no variadic arguments:
//
// FORWARD_BINARY_FUNC(operator+, x.w + y.w, x.df + y.df)
// =>
// template <class T>
// inline auto operator+(ADForward<T> const& x, ADForward<T> const& y)
// {
//      return ad::core::ADForward<T>(x.w + y.w, x.df + y.df);
// }
#define FORWARD_BINARY_FUNC(f, first, second) \
template <class T> \
inline auto f(ADForward<T> const& x, ADForward<T> const& y) \
{ \
	return ad::core::ADForward<T>(first, second); \
} \

namespace ad {

namespace core {

// Forward variable to store value and adjoint.
// If x is an ADForward variable that is a result of composing functions of ADForward variables x1,...,xn
// x.w is the value of the function on these variables and x.df is the adjoint, i.e.
// directional (total) derivative of the composed functions in the direction of x1.df,...,xn.df
template <class T>
struct ADForward : core::DualNum<T>
{
    using datatype = core::DualNum<T>;

    ADForward()
        : datatype(0, 0)
    {}

    ADForward(T w, T df = 0)
        : datatype(w, df)
    {}

    ADForward& operator+=(ADForward const& x);
};

} // namespace core

// user-exposed forward variable alias 
template <class T>
using ForwardVar = core::ADForward<T>;

//================================================================================

// Unary functions 

// Negate forward variable
FORWARD_UNARY_FUNC(operator-, -x.w, -x.df)
// ad::sin(core::ADForward)
FORWARD_UNARY_FUNC(sin, std::sin(x.w), std::cos(x.w)*x.df)
// ad::cos(core::ADForward)
FORWARD_UNARY_FUNC(cos, std::cos(x.w), -std::sin(x.w)*x.df)
// ad::tan(core::ADForward)
FORWARD_UNARY_FUNC(tan, std::tan(x.w), tmp*tmp * x.df, auto tmp = 1 / std::cos(x.w);)
// ad::asin(core::ADForward)
FORWARD_UNARY_FUNC(asin, std::asin(x.w), x.df / (std::sqrt(1 - x.w*x.w)))
// ad::acos(core::ADForward)
FORWARD_UNARY_FUNC(acos, std::acos(x.w), -x.df / (std::sqrt(1 - x.w*x.w)))
// ad::atan(core::ADForward)
FORWARD_UNARY_FUNC(atan, std::atan(x.w), x.df / (1 + x.w*x.w))
// ad::exp(core::ADForward)
FORWARD_UNARY_FUNC(exp, tmp, tmp * x.df, auto tmp = std::exp(x.w);)
// ad::log(core::ADForward)
FORWARD_UNARY_FUNC(log, std::log(x.w), x.df / x.w)
// ad::sqrt(core::ADForward)
FORWARD_UNARY_FUNC(sqrt, tmp, x.df / (2 * tmp), auto tmp = std::sqrt(x.w);)

//================================================================================

// Binary operators 

namespace core {

// Add forward variables
FORWARD_BINARY_FUNC(operator+, x.w + y.w, x.df + y.df)
// Subtract forward variables
FORWARD_BINARY_FUNC(operator-, x.w - y.w, x.df - y.df)
// Multiply forward variables
FORWARD_BINARY_FUNC(operator*, x.w * y.w, x.w * y.df + x.df * y.w)
// Divide forward variables
FORWARD_BINARY_FUNC(operator/, x.w / y.w, (x.df * y.w - x.w * y.df) / (y.w * y.w))

// Add current forward variable with x and update current variable with the result.
template <class T>
inline ADForward<T>& ADForward<T>::operator+=(ADForward<T> const& x)
{
    return *this = std::move(*this + x);
}

} // namespace core

} // namespace ad
