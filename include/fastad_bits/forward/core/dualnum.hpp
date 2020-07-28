#pragma once

namespace ad {
namespace core {

// Underlying data structure containing value and adjoint.
// Represent value as "w" and adjoint as "df".
// @tparam T    underlying data type (ex. double)
template <class T>
struct DualNum
{
    using value_type = T;

    DualNum(T w, T df)
        : w_(w), df_(df)
    {}

    value_type& get_value() 
    {
        return w_;
    }

    const value_type& get_value() const
    {
        return w_;
    }

    value_type& set_value(value_type x) 
    {
        return w_ = x;
    }

    value_type& get_adjoint() 
    {
        return df_;
    }

    const value_type& get_adjoint() const
    {
        return df_;
    }

    value_type& set_adjoint(value_type x) 
    {
        return df_ = x;
    }

private:
    T w_;
    T df_;   
};

} // namepsace core
} // namespace ad 
