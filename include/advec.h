#pragma once
#include "adnode.h"
#include <vector>
#include <initializer_list>

namespace ad {
    template <class T, size_t N>
    struct Vec
    {
        template <class U>
        using impl_type = std::vector<U>;
        T dfs[N];
        impl_type<core::ADNode<T>> leaves;
        T f;

        Vec(std::initializer_list<T> il)
            : leaves(il.size()), f(0)
        {
            size_t i=0;
            std::transform(
                il.begin()
                , il.end()
                , leaves.begin()
                , [this, &i](T x)
                -> core::ADNode<T> {return make_node(x, dfs+(i++));}
                );
        }

        inline core::ADNode<T>& operator[](size_t i)
        {return leaves[i];}

        template <class Derived>
        inline Vec<T,N>& operator=(core::ADNodeExpr<Derived> const& expr_)
        {
            Derived& expr = const_cast<Derived&>(expr_.self());
            f = expr.feval();
            expr.df = 1; // seed
            expr.beval();
            return *this;
        }
    };

    template <class... Ts>
    inline auto make_vec(Ts ...xi)
        -> Vec<typename std::common_type<Ts...>::type, sizeof...(xi)>
    {return Vec<typename std::common_type<Ts...>::type, sizeof...(xi)>({xi...});}

} // namespace ad
