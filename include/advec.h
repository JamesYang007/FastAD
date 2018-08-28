#pragma once
#include "adnode.h"
#include <vector>
#include <boost/range/irange.hpp>
#include <boost/range/combine.hpp>
#include <stdexcept>

namespace ad {
    template <class T>
    struct Vec
    {
        using impl_type = std::vector<core::ADNode<T>>;
        impl_type vec;
        size_t currentN;
        size_t maxcap;

        // Construct with initial values
        Vec(std::initializer_list<T> il, size_t capacity=1e6)
            : vec(), currentN(il.size()), maxcap(capacity)
        {if (currentN > maxcap) throw std::length_error("maximum capacity reached.");
            this->vec.reserve(maxcap);
            this->init(il);}

        // Construct with initial values and memory ptr to adjoints
        Vec(std::initializer_list<T> il, T* memptr, size_t capacity=1e6)
            : vec(), currentN(il.size()), maxcap(capacity)
        {if (this->currentN > this->maxcap) throw std::length_error("maximum capacity reached.");
            this->vec.reserve(this->maxcap);
            this->init(il, memptr);}

        // [] operator
        inline core::ADNode<T>& operator[](size_t i)
        {return this->vec[i];}
        inline core::ADNode<T> const& operator[](size_t i) const
        {return this->vec[i];}

        // push_back
        // Note: w/df_ptr will be copied from node.w/df_ptr
        void push_back(core::ADNode<T> const& node)
        {if (this->currentN < this->maxcap) this->vec.push_back(node);
            else throw std::length_error("maximum capacity reached.");}

        // emplace_back
        template <class ...Args>
        void emplace_back(Args&&... args)
        {if (this->currentN < this->maxcap) this->vec.emplace_back(args...);
            else throw std::length_error("maximum capacity reached.");}

    private:
        // Init functions
        template <
            class InitType
            >
        inline void init(InitType const& init) 
        {
            for (auto x : init) {
                this->vec.emplace_back(x);
            }
        }

        template <class InitType>
        inline void init(InitType const& init, T* memptr) 
        {
            for (auto x : init) {
                this->vec.emplace_back(x, memptr++);
            }
        }

    };

} // namespace ad
