#pragma once
#include "adnode.h"
#include <cmath>

namespace ad {
namespace math {

    // Unary Operators
    //
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
    //
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
template <class T, class Derived>
inline auto sin(core::ADNode<T, Derived> const& node)
    -> core::UnaryNode<T
        , typename math::Sin<T>
        , Derived> 
{return core::UnaryNode<T
        , typename math::Sin<T>
        , Derived>(
                const_cast<core::ADNode<T, Derived>&>(node).self())
        ;}

// ad::cos(ADNode)
// Returns a UnaryNode containing Cos as Unary operation
template <class T, class Derived>
inline auto cos(core::ADNode<T, Derived> const& node)
    -> core::UnaryNode<T
        , typename math::Cos<T>
        , Derived> 
{return core::UnaryNode<T
        , typename math::Cos<T>
        , Derived>(
                const_cast<core::ADNode<T, Derived>&>(node).self())
        ;}

// ad::exp(ADNode)
// Returns a UnaryNode containing Exp as Unary operation
template <class T, class Derived>
inline auto exp(core::ADNode<T, Derived> const& node)
    -> core::UnaryNode<T
        , typename math::Exp<T>
        , Derived> 
{return core::UnaryNode<T
        , typename math::Exp<T>
        , Derived>(
                const_cast<core::ADNode<T, Derived>&>(node).self())
        ;}

//================================================================================

// Binary operator functions 
//
// ad::core::operator+(ADNode)
// Returns a BinaryNode containing Operator
namespace core {
template <class T, class Derived1, class Derived2>
inline auto operator+(
        ADNode<T, Derived1> const& node1
        , ADNode<T, Derived2> const& node2)
    -> BinaryNode<T
        , typename ad::math::Add<T>
        , Derived1
        , Derived2>
{return BinaryNode<T
        , typename ad::math::Add<T>
        , Derived1
        , Derived2>(
                const_cast<ADNode<T, Derived1>&>(node1).self()
                , const_cast<ADNode<T, Derived2>&>(node2).self())
        ;}

// ad::core::operator-(ADNode)
template <class T, class Derived1, class Derived2>
inline auto operator-(
        ADNode<T, Derived1> const& node1
        , ADNode<T, Derived2> const& node2)
    -> BinaryNode<T
        , typename ad::math::Sub<T>
        , Derived1
        , Derived2>
{return BinaryNode<T
        , typename ad::math::Sub<T>
        , Derived1
        , Derived2>(
                const_cast<ADNode<T, Derived1>&>(node1).self()
                , const_cast<ADNode<T, Derived2>&>(node2).self())
        ;}

// ad::core::operator*(ADNode)
template <class T, class Derived1, class Derived2>
inline auto operator*(
        ADNode<T, Derived1> const& node1
        , ADNode<T, Derived2> const& node2)
    -> BinaryNode<T
        , typename ad::math::Mul<T>
        , Derived1
        , Derived2>
{return BinaryNode<T
        , typename ad::math::Mul<T>
        , Derived1
        , Derived2>(
                const_cast<ADNode<T, Derived1>&>(node1).self()
                , const_cast<ADNode<T, Derived2>&>(node2).self())
        ;}

// ad::core::operator/(ADNode)
template <class T, class Derived1, class Derived2>
inline auto operator/(
        ADNode<T, Derived1> const& node1
        , ADNode<T, Derived2> const& node2)
    -> BinaryNode<T
        , typename ad::math::Div<T>
        , Derived1
        , Derived2>
{return BinaryNode<T
        , typename ad::math::Div<T>
        , Derived1
        , Derived2>(
                const_cast<ADNode<T, Derived1>&>(node1).self()
                , const_cast<ADNode<T, Derived2>&>(node2).self())
        ;}
} // namespace core
} // namespace ad
