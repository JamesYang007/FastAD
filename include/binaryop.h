#pragma once
#include "dualnum.h"
#include "expr.h"
#include "utils.h"
#include <iostream>

// Pre-defined Binary Operation classes
namespace core {
    // Add
    struct add {
        template <class T>
        inline static T map(T const& lhs, T const& rhs)
        {return lhs + rhs;}
        
        // Specialize for DualNum
        template <class T>
        inline static DualNum<T> map(DualNum<T> const& lhs, DualNum<T> const& rhs)
        {return DualNum<T>(lhs.x + rhs.x, lhs.xdot + rhs.xdot);}
    };

    //// Subtract
    //struct sub {
    //    template <class T>
    //    inline static T map(T lhs, T rhs)
    //    {return lhs - rhs;}

    //    // Specialize for DualNum
    //    template <class T>
    //    inline static DualNum<T> map(DualNum<T> const& lhs, DualNum<T> const& rhs)
    //    {return DualNum<T>(lhs.x - rhs.x, lhs.xdot - rhs.xdot);}
    //};

    //// Multiply
    //struct mul {
    //    template <class T>
    //    inline static T map(T lhs, T rhs)
    //    {return lhs * rhs;}

    //    // Specialize for DualNum
    //    template <class T>
    //    inline static DualNum<T> map(DualNum<T> const& lhs, DualNum<T> const& rhs)
    //    {return DualNum<T>(lhs.x * rhs.x, lhs.xdot * rhs.x + lhs.x * rhs.xdot);}
    //};

    //// Divide
    //struct div {
    //    template <class T>
    //    inline static T map(T lhs, T rhs)
    //    {return lhs / rhs;}
    //};
}

// BinaryOpExpr
namespace core {
    // Binary Op Expression
    template <class Op, class TL, class TR>
    struct BinaryOpExpr : public Expr<BinaryOpExpr<Op, TL, TR>>
    {
        TL const& lhs;
        TR const& rhs;

        BinaryOpExpr(TL const& lhs, TR const& rhs)
            : lhs(lhs), rhs(rhs)
        {}

        // Evaluate binary operation expression
        inline auto eval() const 
            -> decltype(Op::map(lhs.eval(), rhs.eval()))
        {return Op::map(lhs.eval(), rhs.eval());}
    };

    // Make BinaryOpExpr function
    template <class Op, class TL, class TR>
    inline auto make_BinaryOpExpr(TL const& lhs, TR const& rhs)
        -> BinaryOpExpr<Op, TL, TR>
    {return BinaryOpExpr<Op, TL, TR>(lhs, rhs);}

    // operator+
    template <class TL, class TR>
    inline auto operator+(TL const& lhs, TR const& rhs)
        -> BinaryOpExpr<add, TL, TR>
    {return make_BinaryOpExpr<add>(lhs, rhs);}
}
