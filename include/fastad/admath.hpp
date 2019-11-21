#pragma once
#include "adnode.hpp"
#include "adforward.hpp"    // USING_STD_AD requires all overloads of ad::fname
                            // adforward defines these overloads
#include <cmath>
#include <utility>

// Allow compiler to choose from namespace std or ad
#define USING_STD_AD(fname) \
using std::fname;\
using ad::fname;

// Unary Struct
// fmap:	evaluate the function
// bmap:	evaluate the derivative
#define UNARY_STRUCT(name, fmap_body, bmap_body) \
template <class T> \
struct name \
{ \
	static T fmap(T x) \
	{ \
		fmap_body \
	} \
	static T bmap(T x) \
	{ \
		bmap_body \
	} \
}; 

// Unary function
// Declares function with name associated with struct_name
#define ADNODE_UNARY_FUNC_DECL(name) \
template <class Derived> \
inline auto name(core::ADNodeExpr<Derived> const& node) 

// Defines function with name associated with struct_name
#define ADNODE_UNARY_FUNC_DEF(name, struct_name) \
ADNODE_UNARY_FUNC_DECL(name) \
{return ad::core::ADNode< \
        typename Derived::value_type \
        , typename ad::math::struct_name<typename Derived::value_type> \
        , Derived>(node.self()) \
        ;} 

// Binary Struct
// fmap:	evaluate binary operation
// blmap:	evaluate partial derivative w.r.t. lhs argument
// brmap:	evaluate partial derivative w.r.t. rhs argument
#define BINARY_STRUCT(name, fmap_body, blmap_body, brmap_body) \
template <class T> \
struct name \
{ \
	static T fmap(T x, T y) \
	{fmap_body} \
	static T blmap(T x, T y) \
	{blmap_body} \
	static T brmap(T x, T y) \
	{brmap_body} \
}; 

#define ADNODE_BINARY_FUNC(name, struct_name) \
template < \
    class Derived1 \
    , class Derived2 \
    , typename value_type =  \
        typename std::common_type< \
            typename Derived1::value_type, typename Derived2::value_type \
            >::type \
    > \
inline auto name( \
        ADNodeExpr<Derived1> const& node1 \
        , ADNodeExpr<Derived2> const& node2) \
{return make_node<value_type, typename ad::math::struct_name<value_type>>( \
            node1.self(), node2.self());} 

namespace ad {

// Unary Operators

ADNODE_UNARY_FUNC_DECL(sin);
ADNODE_UNARY_FUNC_DECL(cos);
ADNODE_UNARY_FUNC_DECL(tan);
ADNODE_UNARY_FUNC_DECL(asin);
ADNODE_UNARY_FUNC_DECL(acos);
ADNODE_UNARY_FUNC_DECL(atan);
ADNODE_UNARY_FUNC_DECL(exp);
ADNODE_UNARY_FUNC_DECL(log);
ADNODE_UNARY_FUNC_DECL(id);

namespace math {

// UnaryMinus struct
UNARY_STRUCT(UnaryMinus, return -x;, return -1;)
// Sin struct
UNARY_STRUCT(Sin, USING_STD_AD(sin) return sin(x);, USING_STD_AD(cos)return cos(x);)
// Cos struct
UNARY_STRUCT(Cos, return Sin<T>::bmap(x);, return -Sin<T>::fmap(x);)
// Tan struct
UNARY_STRUCT(Tan, USING_STD_AD(tan) return tan(x);, auto tmp = Cos<T>::fmap(x); return T(1) / (tmp * tmp);)
// Arcsin (degrees)
UNARY_STRUCT(Arcsin, USING_STD_AD(asin) return asin(x);, return 1 / sqrt(1 - x * x);)
// Arccos (degrees)
UNARY_STRUCT(Arccos, USING_STD_AD(acos) return acos(x);, return -Arcsin<T>::bmap(x);)
// Arctan (degrees)
UNARY_STRUCT(Arctan, USING_STD_AD(atan) return atan(x);, return 1 / (1 + x * x);)
// Exp struct
UNARY_STRUCT(Exp, USING_STD_AD(exp) return exp(x); , return fmap(x);)
// Log struct
UNARY_STRUCT(Log, USING_STD_AD(log) return log(x);, return T(1) / x;)
// Identity struct
UNARY_STRUCT(Id, return x; , return T(1);)

//================================================================================
// Binary Operators

// Add
BINARY_STRUCT(Add, return x + y;, return 1;, return 1;)
// Subtract
BINARY_STRUCT(Sub, return x - y;, return 1;, return -1;)
// Multiply
BINARY_STRUCT(Mul, return x * y;, return y;, return x;)
// Divide
BINARY_STRUCT(Div, return x / y;, return T(1) / y;, return T(-1)*x / (y*y);)

} // namespace math

//================================================================================
// ADNodeExpr ONLY

// Unary functions 

// Unary minus
ADNODE_UNARY_FUNC_DEF(operator-, UnaryMinus)
// ad::sin(ADNode)
ADNODE_UNARY_FUNC_DEF(sin, Sin)
// ad::cos(ADNode)
ADNODE_UNARY_FUNC_DEF(cos, Cos)
// ad::tan(ADNode)
ADNODE_UNARY_FUNC_DEF(tan, Tan)
// ad::asin(ADNode)
ADNODE_UNARY_FUNC_DEF(asin, Arcsin)
// ad::acos(ADNode)
ADNODE_UNARY_FUNC_DEF(acos, Arccos)
// ad::atan(ADNode)
ADNODE_UNARY_FUNC_DEF(atan, Arctan)
// ad::exp(ADNode)
ADNODE_UNARY_FUNC_DEF(exp, Exp)
// ad::log(ADNode)
ADNODE_UNARY_FUNC_DEF(log, Log)
// ad::id(ADNode)
ADNODE_UNARY_FUNC_DEF(id, Id)

//================================================================================

// Binary operator functions 
namespace core {

// ad::core::operator+(ADNode)
ADNODE_BINARY_FUNC(operator+, Add)
// ad::core::operator-(ADNode)
ADNODE_BINARY_FUNC(operator-, Sub)
// ad::core::operator*(ADNode)
ADNODE_BINARY_FUNC(operator*, Mul)
// ad::core::operator/(ADNode)
ADNODE_BINARY_FUNC(operator/, Div)

} // namespace core

} // namespace ad
