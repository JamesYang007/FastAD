#pragma once
#include "adnode.h"
#include <deque>
#include <boost/range/irange.hpp>
#include <boost/range/combine.hpp>
#include <stdexcept>

namespace ad {
    template <class T>
    struct Vec
    {
        using impl_type = std::deque<core::ADNode<T>>;
        impl_type vec;

        Vec() : vec() {}

        // Construct (default) with preset number of items
        Vec(size_t n)
            : vec(n)
        {}

        // Construct with initial values
        Vec(std::initializer_list<T> il)
            : vec()
        {this->init(il);}

        // Construct with initial values and memory ptr to adjoints
        Vec(std::initializer_list<T> il, T* memptr)
            : vec()
        {this->init(il, memptr);}

        // [] operator
        inline core::ADNode<T>& operator[](size_t i)
        {return this->vec[i];}
        inline core::ADNode<T> const& operator[](size_t i) const
        {return this->vec[i];}

        // push_back
        // Note: w/df_ptr will be copied from node.w/df_ptr
        void push_back(core::ADNode<T> const& node)
        {this->vec.push_back(node);}

        // emplace_back
        template <class ...Args>
        void emplace_back(Args&&... args)
        {this->vec.emplace_back(args...);}

        // size wrapper
        inline size_t size() const
        {return vec.size();}

        // begin wrapper
        inline auto begin() const
            -> decltype(vec.begin())
        {return vec.begin();}

        inline auto begin() 
            -> decltype(vec.begin())
        {return vec.begin();}

        // end wrapper
        inline auto end() const
            -> decltype(vec.end())
        {return vec.end();}
        
        inline auto end() 
            -> decltype(vec.end())
        {return vec.end();}

        // clear
        inline void clear()
        {vec.clear();}

        // reserve
        inline void reserve(size_t n)
        {vec.reserve(n);}

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
