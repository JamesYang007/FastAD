#pragma once
#include "adnode.h"

//====================================================================================================
namespace ad {

// NO GOOD
template <class T>
constexpr inline auto make_node(T x, T* df_ptr=nullptr, T df=static_cast<T>(0)) 
    -> core::ADNode<T>
{return core::make_node(x, df_ptr, df);}

template <class T>
constexpr inline auto make_node(T x, T* x_ptr, T* df_ptr=nullptr, T df=static_cast<T>(0)) 
    -> core::ADNode<T>
{return core::make_node(x, x_ptr, df_ptr, df);}
// end no good

// Testing already uses it
template <class T>
constexpr inline auto make_leaf(T x, T* df_ptr=nullptr, T df=static_cast<T>(0)) 
    -> core::ADNode<T>
{return make_node(x, df_ptr, df);}

// User-friendly (intuitive)
template <class T>
constexpr inline auto make_var(T x, T* df_ptr, T df=static_cast<T>(0))
    -> core::ADNode<T>
{return make_node(x, df_ptr, df);}

template <class T>
constexpr inline auto make_var(T x=static_cast<T>(0))
    -> core::ADNode<T>
{return make_node(x);}

} // namespace ad
