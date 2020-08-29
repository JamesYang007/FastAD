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
} // namespace ad
