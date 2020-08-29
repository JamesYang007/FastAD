#pragma once
#include <utility>
#include <fastad_bits/util/type_traits.hpp>
#include <Eigen/Dense>

namespace ad {
namespace util {

template <class T>
constexpr inline 
std::enable_if_t<!util::is_eigen_v<std::decay_t<T>>, T&&>
to_array(T&& x)
{ return std::forward<T>(x); }

template <class T>
constexpr inline 
auto to_array(const Eigen::MatrixBase<T>& x)
{ return x.array(); }

template <class T>
constexpr inline 
auto to_array(Eigen::MatrixBase<T>& x)
{ return x.array(); }

template <class T, class XType>
constexpr inline
auto cast_to(const XType& x) 
{
    if constexpr (util::is_eigen_v<XType>) {
        return x.template cast<T>();
    } else {
        return x;
    }
};

template <class T>
constexpr inline void ones(T& x) 
{
    using x_t = std::decay_t<T>;
    if constexpr (util::is_eigen_v<x_t>) {
        x.setOnes();
    } else {
        x = 1;
    }
};

} // namespace util
} // namespace ad
