#pragma once
#include "expr.h"
#include <iostream>

namespace core {

template <class T>
struct DualNum: public Expr<DualNum<T>>
{
    using value_type = T;
    T x;
    T xdot;
    DualNum() =default;
    DualNum(T x, T xdot)
        : x(x), xdot(xdot)
    {}

    template <class ExprType, class Enable=void>
    inline DualNum<T>& operator=(Expr<ExprType> const& expr_) {
        ExprType const& expr = expr_.self();
        *this = expr.eval();
        return *this;
    }

    // Higher precedence!
    inline DualNum<T>& operator=(DualNum<T> const& N) {
        x = N.x; xdot = N.xdot; return *this;
    }
    
    // Equality
    inline bool operator==(DualNum<T> const& n) const {
        return (x == n.x) && (xdot == n.xdot);
    }

    // Evaluate
    inline DualNum<T> const& eval() const 
    {return *this;}
};


} // end core
