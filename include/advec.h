#pragma once
#include "adnode.h"
#include <array>
#include <initializer_list>

namespace ad {
    template <class T, size_t N>
    struct Vec
    {
        template <class U>
        using impl_type = std::array<U, N>;
        impl_type<core::ADNode<T>> xvec;

        // memory ptr (array) for adjoint, x1,...,xn
        Vec(T* memptr, std::initializer_list<T> il)
        {
            size_t i=0;
            std::transform(
                il.begin()
                , il.end()
                , xvec.begin()
                , [this, &i, memptr](T x)
                -> core::ADNode<T> {++i; return make_node(x, &xvec[i-1].w, memptr+i-1);}
                );
        }

        inline core::ADNode<T>& operator[](size_t i)
        {return xvec[i];}
    };

    template <class T, class... Ts>
    inline auto make_vec(T* memptr, Ts ...xi)
        -> Vec<typename std::common_type<T, Ts...>::type, sizeof...(xi)>
    {return Vec<typename std::common_type<T, Ts...>::type, sizeof...(xi)>(memptr, {xi...});}

} // namespace ad
