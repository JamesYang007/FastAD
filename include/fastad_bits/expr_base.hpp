#pragma once
#include <type_traits>

namespace ad {
namespace core {

/* 
 * AD node expression base class.
 * Any ExprBase can be thought of as a node when graphically representing
 * Nodes (Derived) can be defined with different names and template arguments
 * like the Basic Nodes as well as ForEach, etc.
 * This is mainly used for type-checking when operator overloading and defining
 * user-friendly functions for math operations.
 */

template <class Derived>
struct ExprBase
{
    const Derived& self() const
    { return *static_cast<const Derived*>(this); }
    Derived& self()
    { return *static_cast<Derived*>(this); }
};

} // namespace core

namespace util {

template <class T>
struct expr_traits
{
    using value_t = typename T::value_t;
    using shape_t = typename T::shape_t;
    using var_t = typename T::var_t;
};

template <class T>
inline constexpr bool is_expr_v =
    std::is_base_of_v<core::ExprBase<T>, T>;

} // namespace util
} // namespace ad
