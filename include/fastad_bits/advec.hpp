#pragma once
#include <deque>
#include <stdexcept>
#include "adnode.hpp"

namespace ad {

// Vector of ADNodes
// Implemented by inheriting from std::deque.
// We use deque since ADNode contains pointer values that may become garbage
// if ADNodes get copied or moved.
// @tparam  T   underlying data type
template <class T>
struct Vec : std::deque<core::ADNode<T>>
{
    using base_t = std::deque<core::ADNode<T>>;
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
            this->emplace_back(x, memptr++);
        }
    }
};

} // namespace ad
