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
    // Tan struct
    template <class T>
    struct Tan 
    {
        static inline T fmap(T x)
        {return std::tan(x);}
        static inline T bmap(T x)
        {auto tmp = Cos<T>::fmap(x); return 1/(tmp * tmp);}
    };

    // Arcsin (degrees)
    template <class T>
    struct Arcsin
    {
        static inline T fmap(T x)
        {return std::asin(x);}
        static inline T bmap(T x)
        {return 1 / std::sqrt(1 - x*x);}
    };

    // Arccos (degrees)
    template <class T>
    struct Arccos
    {
        static inline T fmap(T x)
        {return std::acos(x);}
        static inline T bmap(T x)
        {return -Arcsin<T>::bmap(x);}
    };
    
    // Arctan (degrees)
    template <class T>
    struct Arctan
    {
        static inline T fmap(T x)
        {return std::atan(x);}
        static inline T bmap(T x)
        {return 1 / (1 + x*x);}
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

//================================================================================
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

// ad::tan(ADNode)
template <class Derived>
inline auto cos(core::ADNodeExpr<Derived> const& node)
    -> core::ADNode<
        typename Derived::valuetype
        , typename math::Tan<typename Derived::valuetype>
        , Derived> 
{return core::ADNode<
        typename Derived::valuetype
        , typename math::Tan<typename Derived::valuetype>
        , Derived>(node.self())
        ;}

// ad::asin(ADNode)
template <class Derived>
inline auto asin(core::ADNodeExpr<Derived> const& node)
    -> core::ADNode<
        typename Derived::valuetype
        , typename math::Arcsin<typename Derived::valuetype>
        , Derived> 
{return core::ADNode<
        typename Derived::valuetype
        , typename math::Arcsin<typename Derived::valuetype>
        , Derived>(node.self())
        ;}

// ad::acos(ADNode)
template <class Derived>
inline auto acos(core::ADNodeExpr<Derived> const& node)
    -> core::ADNode<
        typename Derived::valuetype
        , typename math::Arccos<typename Derived::valuetype>
        , Derived> 
{return core::ADNode<
        typename Derived::valuetype
        , typename math::Arccos<typename Derived::valuetype>
        , Derived>(node.self())
        ;}

// ad::atan(ADNode)
template <class Derived>
inline auto atan(core::ADNodeExpr<Derived> const& node)
    -> core::ADNode<
        typename Derived::valuetype
        , typename math::Arctan<typename Derived::valuetype>
        , Derived> 
{return core::ADNode<
        typename Derived::valuetype
        , typename math::Arctan<typename Derived::valuetype>
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
template <
    class Derived1
    , class Derived2
    , typename value_type = 
        typename std::common_type<
            typename Derived1::valuetype, typename Derived2::valuetype
            >::type
    >
inline auto operator+(
        ADNodeExpr<Derived1> const& node1
        , ADNodeExpr<Derived2> const& node2)
    -> ADNode<
        value_type
        , typename ad::math::Add<value_type>
        , Derived1
        , Derived2>
{return make_node<value_type, typename ad::math::Add<value_type>>(
            node1.self(), node2.self());}

//========================================================================================
// ad::core::operator-(ADNode)
// expr - expr
template <
    class Derived1
    , class Derived2
    , typename value_type = 
        typename std::common_type<
            typename Derived1::valuetype, typename Derived2::valuetype
            >::type
    >
inline auto operator-(
        ADNodeExpr<Derived1> const& node1
        , ADNodeExpr<Derived2> const& node2)
    -> ADNode<
        value_type
        , typename ad::math::Sub<value_type>
        , Derived1
        , Derived2>
{return make_node<value_type, typename ad::math::Sub<value_type>>(
            node1.self(), node2.self());}

//========================================================================================
// ad::core::operator*(ADNode)
// expr * expr
template <
    class Derived1
    , class Derived2
    , typename value_type = 
        typename std::common_type<
            typename Derived1::valuetype, typename Derived2::valuetype
            >::type
    >
inline auto operator*(
        ADNodeExpr<Derived1> const& node1
        , ADNodeExpr<Derived2> const& node2)
    -> ADNode<
        value_type
        , typename ad::math::Mul<value_type>
        , Derived1
        , Derived2>
{return make_node<value_type, typename ad::math::Mul<value_type>>(
            node1.self(), node2.self());}

//========================================================================================
// ad::core::operator/(ADNode)
// expr / expr
template <
    class Derived1
    , class Derived2
    , typename value_type = 
        typename std::common_type<
            typename Derived1::valuetype, typename Derived2::valuetype
            >::type
    >
inline auto operator/(
        ADNodeExpr<Derived1> const& node1
        , ADNodeExpr<Derived2> const& node2)
    -> ADNode<
        value_type
        , typename ad::math::Div<value_type>
        , Derived1
        , Derived2>
{return make_node<value_type, typename ad::math::Div<value_type>>(
            node1.self(), node2.self());}


} // namespace core


//========================================================================================
// ad::sum(Iter start, Iter end, lmda fn)
template <
    class Iter
    , class Lmda
    >
inline auto sum(Iter start, Iter end, Lmda f)
    -> core::SumNode<
    typename decltype(f(*start))::valuetype
    , Iter
    , Lmda
    >
{return core::SumNode<
    typename decltype(f(*start))::valuetype
    , Iter
    , Lmda
    >(start, end, f);
}

// ad::prod(Iter start, Iter end, lmda fn)
template <
    class Iter
    , class Lmda
    >
inline auto prod(Iter start, Iter end, Lmda f)
    -> core::ProdNode<
    typename decltype(f(*start))::valuetype
    , Iter
    , Lmda
    >
{
    return core::ProdNode<
    typename decltype(f(*start))::valuetype
    , Iter
    , Lmda
    >(start, end, f);
}

} // namespace ad
