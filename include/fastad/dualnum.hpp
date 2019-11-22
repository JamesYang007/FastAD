#pragma once

namespace ad {
namespace core {

// Underlying data structure containing value and adjoint.
// See ADForward or ADNode class template.
// Represent value as "w" and adjoint as "df".
// @tparam T    underlying data type (ex. double)
template <class T>
struct DualNum
{
    using value_type = T;
    T w, df;   
    DualNum(T w, T df)
        : w(w), df(df)
    {}
};

} // namepsace core
} // namespace ad 
