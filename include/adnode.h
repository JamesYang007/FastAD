#pragma once
#include "dualnum.h"
#include <utility> // std::move
#include <iostream> // test

namespace ad {
namespace core {

    // ADNode Expression
    template <class Derived>
    struct ADNodeExpr 
    {
        inline Derived const& self() const
        {return *static_cast<Derived const*>(this);}
        inline Derived& self()
        {return *static_cast<Derived*>(this);}
    };

    // Binary Node
    template <class T, class Binary=void, class TL=void, class TR=void>
    struct ADNode: 
        public DualNum<T>, ADNodeExpr<ADNode<T, Binary, TL, TR>>
    {
        using datatype = DualNum<T>;
        TL const& lhs; 
        TR const& rhs;
        ADNode(TL const& lhs, TR const& rhs)
            : datatype(0,0), lhs(lhs), rhs(rhs)
        {}
        //{std::cout << this << ": BinaryNode constructor" << std::endl;}
        //~ADNode()
        //{std::cout << this << ": BinaryNode destructor" << std::endl;}

        inline T feval()
        {this->w = Binary::fmap(
                const_cast<TL&>(lhs).feval()
                , const_cast<TR&>(rhs).feval()); return this->w;}
        // Takes advantage of fact that feval() is always called before beval()
        inline void beval() const
        {
            const_cast<TL&>(lhs).df += Binary::blmap(lhs.w, rhs.w) * this->df; lhs.beval();
            const_cast<TR&>(rhs).df += Binary::brmap(lhs.w, rhs.w) * this->df; rhs.beval();
        }
    };

    // Unary Node
    template <class T, class Unary, class TL>
    struct ADNode<T, Unary, TL>: 
        public DualNum<T>, ADNodeExpr<ADNode<T, Unary, TL>>
    {
        using datatype = DualNum<T>;
        TL const& lhs; 
        ADNode(TL const& lhs)
            : datatype(0,0), lhs(lhs)
        {}
        //{std::cout << this << ": UnaryNode constructor" << std::endl;}
        //~ADNode()
        //{std::cout << this << ": UnaryNode destructor" << std::endl;}

        inline T feval()
        {this->w = Unary::fmap(
                const_cast<TL&>(lhs).feval()); return this->w;}
        // Takes advantage of fact that feval() is always called before beval()
        inline void beval() const
        {const_cast<TL&>(lhs).df += Unary::bmap(lhs.w) * this->df; lhs.beval();}
    };

    // Leaf node
    template <class T>
    struct ADNode<T>: 
        public DualNum<T>, ADNodeExpr<ADNode<T>>
    {
        using datatype = DualNum<T>;
        T* df_ptr;
        ADNode(): datatype(0,0), df_ptr(nullptr) 
        {std::cout << this << ": LeafNode default constructor" << std::endl;};
        ADNode(T w, T* df_ptr, T df) 
            : datatype(w, df), df_ptr(df_ptr)
        {}
        //{std::cout << this << ": LeafNode constructor" << std::endl;}
        //~ADNode()
        //{std::cout << this << ": LeafNode destructor" << std::endl;}
        
        inline T feval() const
        {return this->w;}
        inline void beval() const
        {*df_ptr = this->df;}
    };
    
} // namespace core

// Make functions
namespace core {
    // BinaryNode
    template <class T, class Op=void, class TL=void, class TR=void>
    inline auto make_node(TL const& lhs, TR const& rhs)
        -> ADNode<T, Op, TL, TR>
    {return ADNode<T, Op, TL, TR>(lhs, rhs);}
    
    // UnaryNode
    template <class T, class Unary, class TL>
    inline auto make_node(TL const& lhs)
        -> ADNode<T, Unary, TL>
    {return ADNode<T, Unary, TL>(lhs);}

    // LeafNode
    template <class T>
    inline auto make_node(T x, T* df_ptr, T df)
        -> ADNode<T>
    {return ADNode<T>(x, df_ptr, df);}

    // Testing already uses these names
    template <class T>
    using LeafNode = ADNode<T>;
    template <class T, class Unary, class TL>
    using UnaryNode = ADNode<T, Unary, TL>;
    template <class T, class Binary, class TL, class TR>
    using BinaryNode = ADNode<T, Binary, TL, TR>;

} // namespace core

template <class T>
constexpr inline auto make_node(T x, T* df_ptr=nullptr, T df=static_cast<T>(0)) 
    -> core::ADNode<T>
{return core::make_node(x, df_ptr, df);}

// Testing already uses it
template <class T>
constexpr inline auto make_leaf(T x, T* df_ptr=nullptr, T df=static_cast<T>(0)) 
    -> core::ADNode<T>
{return make_node(x, df_ptr, df);}

// User-friendly (intuitive)
template <class T>
constexpr inline auto make_var(T x, T* df_ptr=nullptr, T df=static_cast<T>(0))
    -> core::ADNode<T>
{return make_node(x, df_ptr, df);}

} // namespace ad
