#pragma once
#include "expr.h"

namespace ad {
namespace core {
    // Expressionify type
    template <class T>
    struct TypeExpr : public Expr<TypeExpr<T>>
    {
        using type = T;
    };

} // namespace core
} // namespace ad
