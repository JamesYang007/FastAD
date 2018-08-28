#pragma once
#include "adnode.h"
#include <cmath>

namespace ad {
namespace math {

    // Unary Operators
   
    // Identity struct
    template <class T>
    struct Identity
    {
        static inline T fmap(T x)
        {return x;}
        static inline T bmap(T x)
        {return 1;}
    };

    // Sin struct
    template <class T>
    struct Sin
    {
        static inline T fmap(T x)
        {return std::sin(x);}
        static inline T bmap(T x)
        {return std::cos(x);}
    };

    // Cos struct
    template <class T>
    struct Cos
    {
        static inline T fmap(T x)
        {return std::cos(x);}
        static inline T bmap(T x)
        {return -std::sin(x);}
    };

    // Exp struct
    template <class T>
    struct Exp
    {
        static inline T fmap(T x)
        {return std::exp(x);}
        static inline T bmap(T x)
        {return fmap(x);}
    };

    // Binary Operators

    // Add
    template <class T>
    struct Add
    {
        static inline T fmap(T x, T y)
        {return x + y;}
        static inline T blmap(T x, T y)
        {return 1;}
        static inline T brmap(T x, T y)
        {return 1;}
    };

    // Subtract
    template <class T>
    struct Sub
    {
        static inline T fmap(T x, T y)
        {return x - y;}
        static inline T blmap(T x, T y)
        {return 1;}
        static inline T brmap(T x, T y)
        {return -1;}
    };

    // Multiply
    template <class T>
    struct Mul
    {
        static inline T fmap(T x, T y)
        {return x * y;}
        static inline T blmap(T x, T y)
        {return y;}
        static inline T brmap(T x, T y)
        {return x;}
    };

    // Divide
    template <class T>
    struct Div
    {
        static inline T fmap(T x, T y)
        {return x / y;}
        static inline T blmap(T x, T y)
        {return static_cast<T>(1) / y;}
        static inline T brmap(T x, T y)
        {return static_cast<T>(-1) * x / (y*y);}
    };

} // namespace math

//================================================================================

// Easy unary functions 
// 
// ad::sin(ADNode)
// Returns a UnaryNode containing Sin as Unary operation
// sin(expr)
template <class Derived>
inline auto sin(core::ADNodeExpr<Derived> const& node)
    -> core::ADNode<
        typename Derived::valuetype
        , typename math::Sin<typename Derived::valuetype>
        , Derived> 
{return core::ADNode<
        typename Derived::valuetype
        , typename math::Sin<typename Derived::valuetype>
        , Derived>(node.self())
        ;}

// ad::cos(ADNode)
// Returns a UnaryNode containing Cos as Unary operation
template <class Derived>
inline auto cos(core::ADNodeExpr<Derived> const& node)
    -> core::ADNode<
        typename Derived::valuetype
        , typename math::Cos<typename Derived::valuetype>
        , Derived> 
{return core::ADNode<
        typename Derived::valuetype
        , typename math::Cos<typename Derived::valuetype>
        , Derived>(node.self())
        ;}

// ad::exp(ADNode)
// Returns a UnaryNode containing Exp as Unary operation
template <class Derived>
inline auto exp(core::ADNodeExpr<Derived> const& node)
    -> core::ADNode<
        typename Derived::valuetype
        , typename math::Exp<typename Derived::valuetype>
        , Derived> 
{return core::ADNode<
        typename Derived::valuetype
        , typename math::Exp<typename Derived::valuetype>
        , Derived>(node.self())
        ;}

//================================================================================

// Binary operator functions 
namespace core {

// ad::core::operator+(ADNode)
// expr + expr
template <class Derived1, class Derived2>
inline auto operator+(
        ADNodeExpr<Derived1> const& node1
        , ADNodeExpr<Derived2> const& node2)
    -> ADNode<
        typename std::common_type<
            typename Derived1::valuetype, typename Derived2::valuetype
            >::type
        , typename ad::math::Add<
            typename std::common_type<
                typename Derived1::valuetype, typename Derived2::valuetype
                >::type
            >
        , Derived1
        , Derived2>
{return make_node<
        typename std::common_type<
            typename Derived1::valuetype, typename Derived2::valuetype
            >::type
        , typename ad::math::Add<
            typename std::common_type<
                typename Derived1::valuetype, typename Derived2::valuetype
                >::type
            >
        >(
            node1.self(), node2.self()
        );}

//========================================================================================
// ad::core::operator-(ADNode)
// expr - expr
template <class Derived1, class Derived2>
inline auto operator-(
        ADNodeExpr<Derived1> const& node1
        , ADNodeExpr<Derived2> const& node2)
    -> ADNode<
        typename std::common_type<
            typename Derived1::valuetype, typename Derived2::valuetype
            >::type
        , typename ad::math::Sub<
            typename std::common_type<
                typename Derived1::valuetype, typename Derived2::valuetype
                >::type
            >
        , Derived1
        , Derived2>
{return make_node<
        typename std::common_type<
            typename Derived1::valuetype, typename Derived2::valuetype
            >::type
        , typename ad::math::Sub<
            typename std::common_type<
                typename Derived1::valuetype, typename Derived2::valuetype
                >::type
            >
        >(
            node1.self(), node2.self()
        );}

//========================================================================================
// ad::core::operator*(ADNode)
// expr * expr
template <class Derived1, class Derived2>
inline auto operator*(
        ADNodeExpr<Derived1> const& node1
        , ADNodeExpr<Derived2> const& node2)
    -> ADNode<
        typename std::common_type<
            typename Derived1::valuetype, typename Derived2::valuetype
            >::type
        , typename ad::math::Mul<
            typename std::common_type<
                typename Derived1::valuetype, typename Derived2::valuetype
                >::type
            >
        , Derived1
        , Derived2>
{return make_node<
        typename std::common_type<
            typename Derived1::valuetype, typename Derived2::valuetype
            >::type
        , typename ad::math::Mul<
            typename std::common_type<
                typename Derived1::valuetype, typename Derived2::valuetype
                >::type
            >
        >(
            node1.self(), node2.self()
        );}

//========================================================================================
// ad::core::operator/(ADNode)
// expr / expr
template <class Derived1, class Derived2>
inline auto operator/(
        ADNodeExpr<Derived1> const& node1
        , ADNodeExpr<Derived2> const& node2)
    -> ADNode<
        typename std::common_type<
            typename Derived1::valuetype, typename Derived2::valuetype
            >::type
        , typename ad::math::Div<
            typename std::common_type<
                typename Derived1::valuetype, typename Derived2::valuetype
                >::type
            >
        , Derived1
        , Derived2>
{return make_node<
        typename std::common_type<
            typename Derived1::valuetype, typename Derived2::valuetype
            >::type
        , typename ad::math::Div<
            typename std::common_type<
                typename Derived1::valuetype, typename Derived2::valuetype
                >::type
            >
        >(
            node1.self(), node2.self()
        );}


} // namespace core


//========================================================================================
// ad::sum(Iter start, Iter end, lmda fn)
template <class Iter, class Lmda>
inline void sum(Iter start, Iter end, Lmda f)
{}

} // namespace ad
