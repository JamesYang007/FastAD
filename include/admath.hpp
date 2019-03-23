#pragma once
#include "adnode.hpp"
#include <cmath>
#include <utility>

#define USING_STD_AD(fname) \
using std::fname;\
using ad::fname;

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

#define ADNODE_UNARY_FUNC(name, struct_name) \
template <class Derived> \
inline auto name(core::ADNodeExpr<Derived> const& node) \
{return core::ADNode< \
        typename Derived::value_type \
        , typename math::struct_name<typename Derived::value_type> \
        , Derived>(node.self()) \
        ;} 

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
	namespace math {

		// Unary Operators

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

	//========================================================================================
	// ad::sum(Iter start, Iter end, lmda fn)
	template <class Iter, class Lmda>
	inline auto sum(Iter start, Iter end, Lmda&& f)
	{
		return core::SumNode<
			typename decltype(f(*start))::value_type
			, Iter
			, Lmda
		>(start, end, std::forward<Lmda>(f));
	}

	// ad::prod(Iter start, Iter end, lmda fn)
	template <
		class Iter
		, class Lmda
	>
		inline auto prod(Iter start, Iter end, Lmda&& f)
	{
		return core::ProdNode<
			typename decltype(f(*start))::value_type
			, Iter
			, Lmda
		>(start, end, std::forward<Lmda>(f));
	}

} // namespace ad
