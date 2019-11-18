#pragma once
#include "dualnum.hpp"
#include <cmath>

// FORWARD_UNARY_FUNC(f, f(x), f'(x), optimization using temporary (tmp))
#define FORWARD_UNARY_FUNC_DECL(f) \
template <class T> \
inline auto f(ad::core::ADForward<T> const& x) 

#define FORWARD_UNARY_FUNC_DEF(f, first, second, ...) \
FORWARD_UNARY_FUNC_DECL(f) \
{ \
	__VA_ARGS__ \
	return ad::core::ADForward<T>(first, second); \
} \

#define FORWARD_BINARY_FUNC(f, first, second) \
template <class T> \
inline auto f(ADForward<T> const& x, ADForward<T> const& y) \
{ \
	return ad::core::ADForward<T>(first, second); \
} \

// Forward-mode Automatic Differentiation

namespace ad {

namespace core {
// Forward Variable

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

// Unary minus
FORWARD_UNARY_FUNC_DEF(operator-, -x.w, -x.df)

} // end core

// USER-FRIENDLY
template <class T>
using ForwardVar = core::ADForward<T>;

//================================================================================

// Unary functions 
// ad::sin(core::ADForward)
FORWARD_UNARY_FUNC_DEF(sin, std::sin(x.w), std::cos(x.w)*x.df)
// ad::cos(core::ADForward)
FORWARD_UNARY_FUNC_DEF(cos, std::cos(x.w), -std::sin(x.w)*x.df)
// ad::tan(core::ADForward)
FORWARD_UNARY_FUNC_DEF(tan, std::tan(x.w), tmp*tmp * x.df, auto tmp = 1 / std::cos(x.w);)
// ad::asin(core::ADForward)
FORWARD_UNARY_FUNC_DEF(asin, std::asin(x.w), x.df / (std::sqrt(1 - x.w*x.w)))
// ad::acos(core::ADForward)
FORWARD_UNARY_FUNC_DEF(acos, std::acos(x.w), -x.df / (std::sqrt(1 - x.w*x.w)))
// ad::atan(core::ADForward)
FORWARD_UNARY_FUNC_DEF(atan, std::atan(x.w), x.df / (1 + x.w*x.w))
// ad::exp(core::ADForward)
FORWARD_UNARY_FUNC_DEF(exp, tmp, tmp * x.df, auto tmp = std::exp(x.w);)
// ad::log(core::ADForward)
FORWARD_UNARY_FUNC_DEF(log, std::log(x.w), x.df / x.w)
// ad::sqrt(core::ADForward)
FORWARD_UNARY_FUNC_DEF(sqrt, tmp, x.df / (2 * tmp), auto tmp = std::sqrt(x.w);)

//================================================================================

// Binary operator functions 

namespace core {

// ad::core::operator+(ADForward<T> const&, ADForward<T> const&)
FORWARD_BINARY_FUNC(operator+, x.w + y.w, x.df + y.df)
// ad::core::operator-(ADForward<T> const&, ADForward<T> const&)
FORWARD_BINARY_FUNC(operator-, x.w - y.w, x.df - y.df)
// ad::core::operator*(ADForward<T> const&, ADForward<T> const&)
FORWARD_BINARY_FUNC(operator*, x.w * y.w, x.w * y.df + x.df * y.w)
// ad::core::operator/(ADForward<T> const&, ADForward<T> const&)
FORWARD_BINARY_FUNC(operator/, x.w / y.w, (x.df * y.w - x.w * y.df) / (y.w * y.w))

template <class T>
inline ADForward<T>& ADForward<T>::operator+=(ADForward<T> const& x)
{
    return *this = std::move(*this + x);
}

} // end core

} // end ad
