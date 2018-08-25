#pragma once
#include "expr.h"

namespace ad {
namespace core {

template <class T>
struct DualNum: public Expr<DualNum<T>>
{
    using value_type = T;
    T w;
    T df;
    DualNum() =default;
    DualNum(T w, T df=0)
        : w(w), df(df)
    {}

    template <class ExprType, class Enable=void>
    inline DualNum<T>& operator=(Expr<ExprType> const& expr_) {
        ExprType const& expr = expr_.self();
        *this = expr.eval();
        return *this;
    }

    // Higher precedence!
    inline DualNum<T>& operator=(DualNum<T> const& N) {
        w = N.w; df = N.df; return *this;
    }
    
    // Equality
    inline bool operator==(DualNum<T> const& n) const {
        return (w == n.w) && (df == n.df);
    }

    // Evaluate
    inline DualNum<T> const& eval() const 
    {return *this;}
};


} // namepsace core
} // namespace ad 
