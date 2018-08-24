#pragma once

namespace core {

template <class SubType>
struct Expr
{
    // returns (casted) SubType const& of itself
    inline SubType const& self() const
    {return *static_cast<SubType const*>(this);}
};

} // end core
