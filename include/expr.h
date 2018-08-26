#pragma once

namespace ad {
namespace core {
namespace test {

template <class SubType>
struct Expr
{
    // returns (casted) SubType const& of itself
    inline SubType const& self() const
    {return *static_cast<SubType const*>(this);}
};

} // namespace test

} // end core
} // namespace ad
