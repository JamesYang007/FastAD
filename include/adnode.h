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

    
//====================================================================================================
// Basic Node Structures

    // Binary Node
    template <class T=void, class Binary=void, class TL=void, class TR=void>
    struct ADNode: 
        public DualNum<T>, ADNodeExpr<ADNode<T, Binary, TL, TR>>
    {
        using datatype = DualNum<T>;
        TL lhs; 
        TR rhs;
        ADNode(TL const& lhs, TR const& rhs)
            : datatype(0,0), lhs(lhs), rhs(rhs)
        {}

        inline T feval()
        {return (this->w = Binary::fmap(lhs.feval(), rhs.feval()));}
        // feval() must be called BEFORE beval()
        inline void beval(T seed=static_cast<T>(1))
        {
            this->df = seed;
            lhs.beval(this->df * Binary::blmap(lhs.w, rhs.w));
            rhs.beval(this->df * Binary::brmap(lhs.w, rhs.w));
        }
    };

    // GlueNode (operator,)
    template <class TL, class TR>
    struct ADNode<void, void, TL, TR>: 
        public ADNodeExpr<ADNode<void, void, TL, TR>>
    {
        using valuetype = typename std::common_type<
                typename TL::valuetype
                , typename TR::valuetype
                >::type;
        TL lhs; 
        TR rhs;
        ADNode(TL const& lhs, TR const& rhs)
            : lhs(lhs), rhs(rhs)
        {}

        inline void feval()
        {lhs.feval(); rhs.feval();}
        inline void beval(valuetype seed)
        {rhs.beval(seed); lhs.beval();}
        inline void beval()
        {rhs.beval(); lhs.beval();}
    };


    // Equal (simply for type computation)
    struct Equal
    {};

    // EqNode (operator=)
    template <class T, class TR>
    struct ADNode<void, Equal, ADNode<T>, TR>: 
        public ADNodeExpr<ADNode<void, Equal, ADNode<T>, TR>>
    {
        using valuetype = typename std::common_type<
                T
                , typename TR::valuetype
                >::type;
        ADNode<T> lhs; 
        TR rhs;
        ADNode(ADNode<T> const& lhs, TR const& rhs)
            : lhs(lhs), rhs(rhs)
        {}

        inline void feval()
        {lhs.w = *lhs.w_ptr = rhs.feval();}
        inline void beval(valuetype seed)
        {*lhs.df_ptr += (lhs.df += seed); this->beval();}
        inline void beval()
        {rhs.beval(*lhs.df_ptr);}
    };

    // Unary Node
    template <class T, class Unary, class TL>
    struct ADNode<T, Unary, TL>: 
        public DualNum<T>, ADNodeExpr<ADNode<T, Unary, TL>>
    {
        using datatype = DualNum<T>;
        TL lhs; 
        ADNode(TL const& lhs)
            : datatype(0,0), lhs(lhs)
        {}

        inline T feval()
        {return (this->w = Unary::fmap(lhs.feval()));}
        inline void beval(T seed)
        {this->df = seed; lhs.beval(this->df * Unary::bmap(lhs.w));}
    };

    // Leaf node
    template <class T>
    struct ADNode<T>: 
        public DualNum<T>, ADNodeExpr<ADNode<T>>
    {
        using datatype = DualNum<T>;
        T* w_ptr;
        T* df_ptr;
        ADNode(): datatype(0,0), w_ptr(&(this->w)), df_ptr(&(this->df))
        {}
        ADNode(T w) 
            : datatype(w, 0), w_ptr(&(this->w)), df_ptr(&(this->df))
        {}
        ADNode(T w, T* w_ptr, T* df_ptr, T df) 
            : datatype(w, df), w_ptr(w_ptr), df_ptr(df_ptr)
        {}
        ADNode(T w, T* df_ptr, T df=static_cast<T>(0)) 
            : datatype(w, df), w_ptr(&(this->w)), df_ptr(df_ptr)
        {}

        // leaf = expression returns EqNode
        //          =
        //        /   \
        //      leaf  expr
        // Note: auto leaf = expression; is different!
        template <class Derived>
        inline auto operator=(ADNodeExpr<Derived> const& node)
            -> ADNode<void, Equal, ADNode<T>, Derived>
        {return ADNode<void, Equal, ADNode<T>, Derived>(*this, node.self());}

        inline T feval()
        {return (this->w = *(this->w_ptr));}
        inline void beval(T seed)
        {*df_ptr += (this->df = seed);}
    };

    // Intuitive typedefs
    template <class T>
    using LeafNode = ADNode<T>;
    template <class T, class Unary, class TL>
    using UnaryNode = ADNode<T, Unary, TL>;
    template <class T, class Binary, class TL, class TR>
    using BinaryNode = ADNode<T, Binary, TL, TR>;
    template <class T, class TR>
    using EqNode = ADNode<void, Equal, ADNode<T>, TR>;
    template <class TL, class TR>
    using GlueNode = ADNode<void, void, TL, TR>;

//====================================================================================================
// Advanced Nodes
// General sum expression
    
    template <class T, class Iter, class Lmda>
    struct SumNode:
        public DualNum<T>, ADNodeExpr<SumNode<T, Iter, Lmda>>
    {
        using datatype = DualNum<T>;
        Iter start, end;
        Lmda f;

        SumNode(Iter start, Iter end, Lmda f) 
            : datatype(0,0)
              , start(start)
              , end(end)
              , f(f)
        {}

        inline T feval()
        {this->eval(false); return this->w;}
        
        inline void beval(T seed)
        {this->eval(true, seed);}

    private:
        inline void eval(bool do_grad, T seed=static_cast<T>(0))
        {
            this->w = 0; // reset
            this->df = seed;
            std::for_each(start, end,
                [this, do_grad](typename std::iterator_traits<Iter>::value_type& expr)
                {
                    auto&& f_expr = f(expr);
                    this->w += f_expr.feval();
                    if (do_grad) f_expr.beval(this->df);
                }
                );
        }

    };
    
} // namespace core

// Make functions
namespace core {
     
    // BinaryNode
    template <class T, class Op=void, class TL=void, class TR=void>
    inline auto make_node(TL const& lhs, TR const& rhs)
        -> ADNode<T, Op, TL, TR>
    {return ADNode<T, Op, TL, TR>(lhs, rhs);}

    // GlueNode
    template <class TL, class TR>
    inline auto make_node(TL const& lhs, TR const& rhs)
        -> ADNode<void, void, TL, TR>
    {return ADNode<void, void, TL, TR>(lhs, rhs);}

    // EqNode
    template <class T, class TR>
    inline auto make_node(ADNode<T> const& lhs, TR const& rhs)
        -> ADNode<void, Equal, ADNode<T>, TR>
    {return ADNode<void, Equal, ADNode<T>, TR>(lhs, rhs);}
    
    // UnaryNode
    template <class T, class Unary, class TL>
    inline auto make_node(TL const& lhs)
        -> ADNode<T, Unary, TL>
    {return ADNode<T, Unary, TL>(lhs);}

    // DO NOT MAKE make_node (LeafNode)
    // Pointers will point to garbage

} // namespace core

// Operator overloads
namespace core {
    // ad::core::operator,(ADNodeExpr)
    // expr, expr
    // Glues two expressions together by comma
    template <class Derived1, class Derived2>
    inline auto operator,(
            core::ADNodeExpr<Derived1> const& node1
            , core::ADNodeExpr<Derived2> const& node2)
        -> core::ADNode<void, void, Derived1, Derived2>
    {return make_node<void, void>(node1.self(), node2.self());}

}

//====================================================================================================
// Glue then evaluate
// Forward propagation
template <class ExprType>
inline void Evaluate(ExprType&& expr)
{expr.feval();}

// Backward propagation
template <class ExprType>
inline void EvaluateAdj(ExprType&& expr)
{expr.beval(1);}

// Both forward and backward
template <class ExprType>
inline void autodiff(ExprType&& expr)
{expr.feval(); expr.beval(1);}


//====================================================================================================
// User typedefs
template <class T>
using Var = core::ADNode<T>;



} // namespace ad
