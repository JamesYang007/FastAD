#pragma once
#include <cmath>
#include <utility>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/reverse/core/unary.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/reverse/core/binary.hpp>
#include <fastad_bits/forward/core/forward.hpp>    

// USING_STD_AD_EIGEN requires all overloads of ad::fname for various fname's
// Ex. ad::sin, ad::cos, ad::tan.
// adforward.hpp defines some of these overloads.
// Rest of the overloads are in this header

// Expose fname from namespace std and ad for look-up.
#define USING_STD_AD_EIGEN(fname) \
using std::fname;\
using ad::fname;\
using Eigen::fname;

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
//
// Example generation:
//
// UNARY_STRUCT(UnaryMinus, return -x;, return -1.;)
// =>
// struct UnaryMinus
// {
//      template <class T>
// 	    static auto fmap(T x)
// 	    {
// 	        return -x;
// 	    }
//
//      template <class T>
// 	    static auto bmap(T x)
// 	    {
// 	        return -1.;
// 	    }
// }; 
#define UNARY_STRUCT(name, fmap_body, bmap_body) \
struct name \
{ \
    template <class T> \
	static auto fmap(T x) \
	{ \
		fmap_body \
	} \
\
    template <class T> \
	static auto bmap(T x) \
	{ \
		bmap_body \
	} \
}; 

// Defines function with name associated with struct_name.
// Overloaded for constant nodes to be eager-evaluated.
// @tparam  Derived     the actual type of node in CRTP
// @return  Unary Node that will evaluate forward and backward direction 
//          defined by "struct_name"'s fmap and bmap acting on "node"
//
// Example generation:
//
// ADNODE_UNARY_FUNC(operator-, UnaryMinus)
// =>
// template <class Derived>
// inline auto operator-(const core::ExprBase<Derived>& node)
// {
//     return core::UnaryNode<math::UnaryMinus
//         , Derived>(node.self());
// } 
// template <class ValueType, class ShapeType> 
// inline auto operator-(const ad::core::ConstantView<
//                              ValueType, ShapeType>& node)
// { 
//     return ad::constant(
//             math::UnaryMinus::fmap(node.get())
//             );
// }
#define ADNODE_UNARY_FUNC(name, struct_name) \
template <class Derived> \
inline auto name(const core::ExprBase<Derived>& node) \
{ \
    return core::UnaryNode<math::struct_name,\
        Derived>(node.self()); \
} \
\
template <class Derived> \
inline auto name(const core::ConstantBase<Derived>& node) \
{ \
    if constexpr (util::is_scl_v<Derived>) { \
        return ad::constant( \
                math::struct_name::fmap(node.self().feval()) \
                ); \
    } else if constexpr (util::is_vec_v<Derived>){ \
        using value_t = std::decay_t<decltype(\
            math::struct_name::fmap(node.self().feval().array())(0) \
                )>; \
        Eigen::Matrix<value_t, Eigen::Dynamic, 1> res = \
                math::struct_name::fmap(node.self().feval().array()); \
        return ad::constant(res); \
    } else { \
        using value_t = std::decay_t<decltype(\
            math::struct_name::fmap(node.self().feval().array())(0,0) \
                )>; \
        Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic> res = \
                math::struct_name::fmap(node.self().feval().array()); \
        return ad::constant(res); \
    }\
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
//
// BINARY_STRUCT(Add, return x + y;, return 1.;, return 1.;)
// =>
// struct Add
// {
//      template <class T, class U>
// 	    static auto fmap(T x, U y)
// 	    {
// 	        return x + y;
// 	    }
//      template <class T, class U>
// 	    static auto blmap(T x, U y)
// 	    {
// 	        return 1.;
// 	    }
//      template <class T, class U>
// 	    static auto brmap(T x, U y)
// 	    {
// 	        return 1.;
// 	    }
// }; 

#define BINARY_STRUCT(name, fmap_body, blmap_body, brmap_body) \
struct name \
{ \
    template <class T, class U> \
	static auto fmap(T x, U y) \
	{ \
        fmap_body \
    } \
    template <class T, class U> \
	static auto blmap(T x, U y) \
	{ \
        blmap_body \
    } \
    template <class T, class U> \
	static auto brmap(T x, U y) \
	{ \
        brmap_body \
    } \
}; 

// Defines function with name associated with struct_name.
// Overload for constant for eager evaluation.
// @tparam  Derived1    the actual type of node1 in CRTP
// @tparam  Derived2    the actual type of node2 in CRTP
// @tparam  value_type  the underlying data type.
//                      By default, it is the common value_type of Derived1 and Derived2.
// @return  Binary Node that will evaluate forward and backward direction
//          defined by "struct_name"'s fmap, blmap, and brmap acting on node1 and node2
//
// ADNODE_BINARY_FUNC(operator+, Add)
// =>
// 
// template <class Derived1
//         , class Derived2>
// inline auto operator+(const core::ExprBase<Derived1>& node1
//                     , const core::ExprBase<Derived2>& node2)
// {
//      return BinaryNode<math::Add, Derived1, Derived2>(
//          node1.self(), node2.self());
// } 
// template <class ValueType, class ShapeType>
// inline auto operator+(const core::ConstantView<ValueType, ShapeType>& node1, 
//                       const core::ConstantView<ValueType, ShapeType>& node2)
// {
//     return ad::constant(math::Add::fmap(
//                 node1.get(), node2.get()
//                 ));
// }

#define ADNODE_BINARY_FUNC(name, struct_name) \
template <class Derived1 \
        , class Derived2> \
inline auto name(const ExprBase<Derived1>& node1, \
                 const ExprBase<Derived2>& node2) \
{ \
    return BinaryNode<math::struct_name, \
                      Derived1, \
                      Derived2>(\
            node1.self(), node2.self()); \
}\
\
template <class Derived1, class Derived2> \
inline auto name(const core::ConstantBase<Derived1>& node1, \
                 const core::ConstantBase<Derived2>& node2) \
{ \
    if constexpr (util::is_scl_v<Derived1> && \
                  util::is_scl_v<Derived2>) { \
        return ad::constant(math::struct_name::fmap( \
                    node1.self().feval(), node2.self().feval() \
                    )); \
    } else if constexpr (util::is_vec_v<Derived1> &&  \
                         util::is_scl_v<Derived2>) {\
        using value_t = std::decay_t<decltype( \
            math::struct_name::fmap( \
                    node1.self().feval().array(), node2.self().feval())(0) \
                )>; \
        Eigen::Matrix<value_t, Eigen::Dynamic, 1> res = \
            math::struct_name::fmap( \
                    node1.self().feval().array(), node2.self().feval()); \
        return ad::constant(res); \
    } else if constexpr (util::is_mat_v<Derived1> &&  \
                         util::is_scl_v<Derived2>) {\
        using value_t = std::decay_t<decltype( \
            math::struct_name::fmap( \
                    node1.self().feval().array(), node2.self().feval())(0,0) \
                )>; \
        Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic> res = \
            math::struct_name::fmap( \
                    node1.self().feval().array(), node2.self().feval()); \
        return ad::constant(res); \
    } else if constexpr (util::is_scl_v<Derived1> &&  \
                         util::is_vec_v<Derived2>) {\
        using value_t = std::decay_t<decltype( \
            math::struct_name::fmap( \
                    node1.self().feval(), node2.self().feval().array())(0) \
                )>; \
        Eigen::Matrix<value_t, Eigen::Dynamic, 1> res = \
            math::struct_name::fmap( \
                    node1.self().feval(), node2.self().feval().array()); \
        return ad::constant(res); \
    } else if constexpr (util::is_scl_v<Derived1> &&  \
                         util::is_mat_v<Derived2>) {\
        using value_t = std::decay_t<decltype( \
            math::struct_name::fmap( \
                    node1.self().feval(), node2.self().feval().array())(0,0) \
                )>; \
        Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic> res = \
            math::struct_name::fmap( \
                    node1.self().feval(), node2.self().feval().array()); \
        return ad::constant(res); \
    } else if constexpr (util::is_vec_v<Derived1>){    \
        using value_t = std::decay_t<decltype( \
            math::struct_name::fmap( \
                    node1.self().feval().array(), node2.self().feval().array())(0) \
                )>; \
        Eigen::Matrix<value_t, Eigen::Dynamic, 1> res = \
            math::struct_name::fmap( \
                    node1.self().feval().array(), node2.self().feval().array()); \
        return ad::constant(res); \
    } else { \
        using value_t = std::decay_t<decltype( \
            math::struct_name::fmap( \
                    node1.self().feval().array(), node2.self().feval().array())(0,0) \
                )>; \
        Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic> res = \
            math::struct_name::fmap( \
                    node1.self().feval().array(), node2.self().feval().array()); \
        return ad::constant(res); \
    }\
}

namespace ad {
namespace math {

// Unary struct definitions 

// UnaryMinus struct
UNARY_STRUCT(UnaryMinus, 
             return -x;, 
             static_cast<void>(x); return -1.;)

// Sin struct
UNARY_STRUCT(Sin, 
             USING_STD_AD_EIGEN(sin) 
             return sin(x);, 
             USING_STD_AD_EIGEN(cos) return cos(x);)
// Cos struct
UNARY_STRUCT(Cos, 
             USING_STD_AD_EIGEN(cos) 
             return cos(x);, 
             return -Sin::fmap(x);)
// Tan struct
UNARY_STRUCT(Tan, 
             USING_STD_AD_EIGEN(tan) 
             return tan(x);, 
             auto tmp = Cos::fmap(x); return T(1.) / (tmp * tmp);)
// Arcsin (degrees)
UNARY_STRUCT(Arcsin, 
             USING_STD_AD_EIGEN(asin) 
             return asin(x);, 
             USING_STD_AD_EIGEN(sqrt) return 1. / sqrt(1. - x * x);)
// Arccos (degrees)
UNARY_STRUCT(Arccos, 
             USING_STD_AD_EIGEN(acos) 
             return acos(x);, 
             return -Arcsin::bmap(x);)
// Arctan (degrees)
UNARY_STRUCT(Arctan, 
             USING_STD_AD_EIGEN(atan) 
             return atan(x);, 
             return 1. / (1. + x * x);)
// Exp struct
UNARY_STRUCT(Exp, 
             USING_STD_AD_EIGEN(exp) 
             return exp(x);, 
             return fmap(x);)
// Log struct
UNARY_STRUCT(Log, 
             USING_STD_AD_EIGEN(log) 
             return log(x);, 
             return T(1.) / x;)
// Identity struct
UNARY_STRUCT(Id, 
             return x;, 
             static_cast<void>(x); return T(1.);)

// Binary struct definitions

// Add
BINARY_STRUCT(Add, 
        return x + y;, 
        static_cast<void>(x); static_cast<void>(y); return 1.;, 
        static_cast<void>(x); static_cast<void>(y); return 1.;)
// Subtract
BINARY_STRUCT(Sub, 
        return x - y;,
        static_cast<void>(x); static_cast<void>(y); return 1.;, 
        static_cast<void>(x); static_cast<void>(y); return -1.;)
// Multiply
BINARY_STRUCT(Mul, 
        return x * y;, 
        static_cast<void>(x); return y;, 
        static_cast<void>(y); return x;)
// Divide
BINARY_STRUCT(Div, 
        return x / y;, 
        static_cast<void>(x); return T(1.) / y;, 
        return T(-1.)*x / (y*y);)

/* 
 * Comparison operators
 * By convention, derivatives always return 0.
 * Backward evaluation should not be called for such binary operators.
 */

// LessThan
BINARY_STRUCT(LessThan, 
        return x < y;,
        static_cast<void>(x); static_cast<void>(y); return 0;,
        static_cast<void>(x); static_cast<void>(y); return 0;)

// LessThanEq
BINARY_STRUCT(LessThanEq, 
        return x <= y;,
        static_cast<void>(x); static_cast<void>(y); return 0;,
        static_cast<void>(x); static_cast<void>(y); return 0;)

// GreaterThan
BINARY_STRUCT(GreaterThan, 
        return x > y;,
        static_cast<void>(x); static_cast<void>(y); return 0;,
        static_cast<void>(x); static_cast<void>(y); return 0;)

// GreaterThanEq
BINARY_STRUCT(GreaterThanEq, 
        return x >= y;,
        static_cast<void>(x); static_cast<void>(y); return 0;,
        static_cast<void>(x); static_cast<void>(y); return 0;)

// Equal
BINARY_STRUCT(Equal, 
        return x == y;,
        static_cast<void>(x); static_cast<void>(y); return 0;,
        static_cast<void>(x); static_cast<void>(y); return 0;)

// NotEqual
BINARY_STRUCT(NotEqual, 
        return x != y;,
        static_cast<void>(x); static_cast<void>(y); return 0;,
        static_cast<void>(x); static_cast<void>(y); return 0;)

// Logical AND
// Note: only well-defined when inputs are of boolean context
BINARY_STRUCT(LogicalAnd, 
        if constexpr (util::is_eigen_matrix_v<T> &&
                      !util::is_eigen_matrix_v<U>) return (x.min(y));
        else if constexpr (!util::is_eigen_matrix_v<T> &&
                           util::is_eigen_matrix_v<U>) return (y.min(x));
        else if constexpr (util::is_eigen_matrix_v<T> &&
                           util::is_eigen_matrix_v<U>) return (x.min(y));
        else return x && y;,
        static_cast<void>(x); static_cast<void>(y); return 0;,
        static_cast<void>(x); static_cast<void>(y); return 0;)

// Logical OR
// Note: only well-defined when inputs are of boolean context
BINARY_STRUCT(LogicalOr, 
        if constexpr (util::is_eigen_matrix_v<T> &&
                      !util::is_eigen_matrix_v<U>) return (x.max(y));
        else if constexpr (!util::is_eigen_matrix_v<T> &&
                           util::is_eigen_matrix_v<U>) return (y.max(x));
        else if constexpr (util::is_eigen_matrix_v<T> &&
                           util::is_eigen_matrix_v<U>) return (x.max(y));
        else return x || y;,
        static_cast<void>(x); static_cast<void>(y); return 0;,
        static_cast<void>(x); static_cast<void>(y); return 0;)

} // namespace math

//================================================================================
// Unary function definitions

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
// NOTE: ALL OPERATOR OVERLOADS MUST BE IN namespace core

namespace core {

// Unary operators

// operator-
ADNODE_UNARY_FUNC(operator-, UnaryMinus)

// Binary operators 

// ad::core::operator+(ADNode)
ADNODE_BINARY_FUNC(operator+, Add)
// ad::core::operator-(ADNode)
ADNODE_BINARY_FUNC(operator-, Sub)
// ad::core::operator*(ADNode)
ADNODE_BINARY_FUNC(operator*, Mul)
// ad::core::operator/(ADNode)
ADNODE_BINARY_FUNC(operator/, Div)

// ad::core::operator<(ADNode)
ADNODE_BINARY_FUNC(operator<, LessThan)
// ad::core::operator<=(ADNode)
ADNODE_BINARY_FUNC(operator<=, LessThanEq)
// ad::core::operator>(ADNode)
ADNODE_BINARY_FUNC(operator>, GreaterThan)
// ad::core::operator>=(ADNode)
ADNODE_BINARY_FUNC(operator>=, GreaterThanEq)
// ad::core::operator==(ADNode)
ADNODE_BINARY_FUNC(operator==, Equal)
// ad::core::operator!=(ADNode)
ADNODE_BINARY_FUNC(operator!=, NotEqual)
// ad::core::operator&&(ADNode)
ADNODE_BINARY_FUNC(operator&&, LogicalAnd)
// ad::core::operator||(ADNode)
ADNODE_BINARY_FUNC(operator||, LogicalOr)

} // namespace core
} // namespace ad
