#pragma once

namespace ad {
namespace util {

/**
 * Pointer pack to wrap the binding material (see reverse/core).
 * This is just for abstraction purposes to minimize 
 * API changes if more pointers need to be added 
 * to member function "bind" for AD expressions.
 */

template <class ValueType>
struct PtrPack
{
    using value_t = ValueType;

    PtrPack(value_t* v,
            value_t* a)
        : val(v), adj(a)
    {}

    value_t* val;
    value_t* adj;
};

} // namespace util
} // namespace ad
