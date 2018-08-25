#pragma once
#include "binaryop.h"
#include "typeexpr.h"
#include <vector>
#include <array>

namespace ad {
namespace core {

    // T (=double), n (=dimension)
    template <class T, size_t n>
    struct AutoDiff
    {
        using dualtype = DualNum<T>;
        using impl_type = std::array<dualtype, n>;
        // Member variables
        // x1,...,xn
        impl_type duals;
        // function value
        T f;

        AutoDiff() =default;
        AutoDiff(std::initializer_list<T> il) 
            : duals(), f(0)
        {size_t i=0; for (auto xi : il) {duals[i] = dualtype(xi); ++i;}}

        //template <class ExprType>
        //AutoDiff<T, n>& operator=(Expr<ExprType> const& expr_)
        //{
        //    ExprType const& expr = expr_.self();
        //    this->f = expr.feval();
        //    return *this;
        //};
    };

    // Make AutoDiff object
    template <class... T>
    constexpr inline auto make_AutoDiff(T... xi)
        -> AutoDiff<typename std::common_type<T...>::type, sizeof...(xi)>
    {return AutoDiff<typename std::common_type<T...>::type, sizeof...(xi)>({xi...});}

} // namespace core
} // namespace ad
