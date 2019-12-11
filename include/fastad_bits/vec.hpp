#pragma once
#include <deque>
#include "node.hpp"

namespace ad {

// Vector of ADNodes
// Implemented by inheriting from std::deque.
// We use deque since ADNode contains pointer values that may become garbage
// if ADNodes get copied or moved.
// @tparam  T   underlying data type
template <class T>
struct Vec : std::deque<Var<T>>
{
    using base_t = std::deque<Var<T>>;
    Vec() =default;

    // Construct with n number of ADNodes
    Vec(size_t n)
        : base_t(n)
    {}

    // Construct ADNodes with initial values in il
    Vec(const std::initializer_list<T>& il)
        : base_t()
    {
        for (auto x : il) {
            this->emplace_back(x);
        }
    }

    // Construct ADNodes with initial values in il and memptr
    // It is assumed that memptr points to an array of T
    // where each element corresponds to the adjoint value.
    Vec(const std::initializer_list<T>& il, T* memptr)
        : base_t()
    {
        for (auto x : il) {
            this->emplace_back(x, memptr);
            ++memptr;
        }
    }

    // Construct LeafNodes with initial values between begin and end.
    template <class Iter>
    Vec(Iter begin, Iter end) 
        : base_t(std::distance(begin, end))
    {
        using iter_value_type = typename std::iterator_traits<Iter>::value_type;
        static_assert(std::is_convertible_v<iter_value_type, T>, 
                "Iterator must point to a value that is convertible to T");
        for (auto& var : *this) {
            var.set_value(*begin);
            ++begin;
        }
    }

    // Resets every variable current adjoint and adjoint destination to 0.
    void reset_adjoint()
    {
        for (auto& var : *this) {
            var.reset_adjoint();
        }
    }
};

} // namespace ad
