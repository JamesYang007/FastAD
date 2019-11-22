#pragma once
#include <cmath>
#include <utility>
#include "adnode.hpp"
#include "adforward.hpp"    // USING_STD_AD requires all overloads of ad::fname for various fname's
                            // Ex. ad::sin, ad::cos, ad::tan.
                            // adforward.hpp defines some of these overloads.
                            // Rest of the overloads are in this header

// Expose fname from namespace std and ad for look-up.
#define USING_STD_AD(fname) \
using std::fname;\
using ad::fname;

// Defines a unary struct with name "name".
// Unary struct contains two static functions: fmap and bmap.
// fmap evaluates the function that the unary struct represents 
// in the forward direction of reverse-mode AD.
// fmap definition is given by "fmap_body".
// bmap evaluates the directional derivative of function that struct represents
// in the backward direction of reverse-mode AD.
// bmap definition is given by "bmap_body".
//
// This struct is acts as a functor that will be passed as a type to Unary Nodes.
// @tparam T    underlying data type (ex. double, float, ADForward)
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

// Defines function with name associated with struct_name.
// @tparam  Derived     the actual type of node in CRTP
// @return  Unary Node that will evaluate forward and backward direction 
//          defined by "struct_name"'s fmap and bmap acting on "node"
#define ADNODE_UNARY_FUNC(name, struct_name) \
template <class Derived> \
inline auto name(ad::core::ADNodeExpr<Derived> const& node) \
{ \
    return ad::core::ADNode< \
        typename Derived::value_type \
        , typename ad::math::struct_name<typename Derived::value_type> \
        , Derived>(node.self()); \
} 

// Defines a binary struct with name "name".
// Binary struct contains three static functions: fmap, blmap, brmap.
// fmap evaluates the binary function on x and y in the forward direction.
// The definition is provided by "fmap_body".
// blmap evaluates the partial derivative w.r.t. x. 
// The definition is provided by "blmap_body".
// brmap evaluates the partial derivative w.r.t. y.
// The definition is provided by "brmap_body".
// @tparam T    underlying data type (ex. double, float, ADForward)
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

// Defines function with name associated with struct_name.
// @tparam  Derived1    the actual type of node1 in CRTP
// @tparam  Derived2    the actual type of node2 in CRTP
// @tparam  value_type  the underlying data type.
//                      By default, it is the common value_type of Derived1 and Derived2.
// @return  Binary Node that will evaluate forward and backward direction
//          defined by "struct_name"'s fmap, blmap, and brmap acting on node1 and node2
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
        const ADNodeExpr<Derived1>& node1 \
        , const ADNodeExpr<Derived2>& node2) \
{return make_node<value_type, typename ad::math::struct_name<value_type>>( \
            node1.self(), node2.self());} 

namespace ad {
namespace math {

// Unary struct definitions 

// UnaryMinus struct
UNARY_STRUCT(UnaryMinus, return -x;, return -1;)
// Sin struct
UNARY_STRUCT(Sin, USING_STD_AD(sin) return sin(x);, USING_STD_AD(cos) return cos(x);)
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

// Binary struct definitions

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
// Unary function definitions

// Unary minus
ADNODE_UNARY_FUNC(operator-, UnaryMinus)
// ad::sin(ADNode)
ADNODE_UNARY_FUNC(sin, Sin)
// ad::cos(ADNode)
ADNODE_UNARY_FUNC(cos, Cos)
// ad::tan(ADNode)
ADNODE_UNARY_FUNC(tan, Tan)
// ad::asin(ADNode)
ADNODE_UNARY_FUNC(asin, Arcsin)
// ad::acos(ADNode)
ADNODE_UNARY_FUNC(acos, Arccos)
// ad::atan(ADNode)
ADNODE_UNARY_FUNC(atan, Arctan)
// ad::exp(ADNode)
ADNODE_UNARY_FUNC(exp, Exp)
// ad::log(ADNode)
ADNODE_UNARY_FUNC(log, Log)
// ad::id(ADNode)
ADNODE_UNARY_FUNC(id, Id)

//================================================================================

namespace core {

// Binary operators 

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
