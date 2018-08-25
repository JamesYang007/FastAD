#pragma once
#include "expr.h"

namespace ad {
namespace core {

    // Node: holds w and df/dw
    template <class T, class Derived>
    struct ADNode 
    {
        T w, df;   
        ADNode(T w, T df)
            : w(w), df(df)
        {}
        inline Derived const& self() const
        {return *static_cast<Derived const*>(this);}
        inline Derived& self()
        {return *static_cast<Derived*>(this);}
    };

    // Leaf node
    template <class T>
    struct LeafNode: public ADNode<T, LeafNode<T>>//, Expr<LeafNode<T>>
    {
        using basetype = ADNode<T, LeafNode<T>>;
        LeafNode(): basetype(0,0) {};
        LeafNode(T w, T df=static_cast<T>(0)) 
            : basetype(w, df)
        {}
        
        inline T feval() const
        {return this->w;}
        inline void beval() const
        {return;}
    };

    // Unary Node
    template <class T, class Unary, class TL>
    struct UnaryNode: 
        public ADNode<T, UnaryNode<T, Unary, TL>>//, Expr<UnaryNode<T,Unary,TL>>
    {
        using basetype = ADNode<T, UnaryNode<T, Unary, TL>>;
        TL& lhs; 
        UnaryNode(TL& lhs)
            : basetype(0,0), lhs(lhs)
        {};
        inline T feval()
        {this->w = Unary::fmap(lhs.feval()); return this->w;}
        // Takes advantage of fact that feval() is always called before beval()
        inline void beval()
        {lhs.df += Unary::bmap(lhs.w) * this->df; lhs.beval();}
    };

    // Binary Node
    template <class T, class Binary, class TL, class TR>
    struct BinaryNode: 
        public ADNode<T, BinaryNode<T, Binary, TL, TR>>//, Expr<UnaryNode<T,Unary,TL>>
    {
        using basetype = ADNode<T, BinaryNode<T, Binary, TL, TR>>;
        TL& lhs; 
        TR& rhs;
        BinaryNode(TL& lhs, TR& rhs)
            : basetype(0,0), lhs(lhs), rhs(rhs)
        {};
        inline T feval() 
        {this->w = Binary::fmap(lhs.feval(), rhs.feval()); return this->w;}
        // Takes advantage of fact that feval() is always called before beval()
        inline void beval()
        {
            lhs.df += Binary::blmap(lhs.w, rhs.w) * this->df; lhs.beval();
            rhs.df += Binary::brmap(lhs.w, rhs.w) * this->df; rhs.beval();
        }
    };
} // namespace core

// Make functions
template <class T>
constexpr inline auto make_leaf(T x) 
    -> core::LeafNode<T>
{return core::LeafNode<T>(x);}

// User-friendly (intuitive)
template <class T>
constexpr inline auto make_var(T x)
    -> core::LeafNode<T>
{return make_leaf(x);}

} // namespace ad
