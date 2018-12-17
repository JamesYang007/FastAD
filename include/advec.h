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

        // default constructor
        Vec() : vec()
        {this->vec.reserve(1e6);}
        // Construct (default) with preset number of items
        Vec(size_t n, size_t capacity=1e6)
            : vec()
        {
            this->vec.reserve(capacity);
            auto&& ir = boost::irange<size_t>(0, n);
            std::for_each(ir.begin(), ir.end(), [this](size_t i) {this->vec.emplace_back();});
        }

        // Construct with initial values
        Vec(std::initializer_list<T> il, size_t capacity=1e6)
            : vec()
        {if (il.size() > capacity) throw std::length_error("initializer list longer than maximum capacity.");
            this->vec.reserve(capacity);
            this->init(il);}

        // Construct with initial values and memory ptr to adjoints
        Vec(std::initializer_list<T> il, T* memptr, size_t capacity=1e6)
            : vec()
        {if (il.size() > capacity) throw std::length_error("initializer list longer than maximum capacity.");
            this->vec.reserve(capacity);
            this->init(il, memptr);}

        // [] operator
        inline core::ADNode<T>& operator[](size_t i)
        {return this->vec[i];}
        inline core::ADNode<T> const& operator[](size_t i) const
        {return this->vec[i];}

        // resets the adjoints
        inline void reset() {
            std::for_each(this->vec.begin(), this->vec.end(),
                    [](core::ADNode<T>& v)
                    {v.df = 0;});
        }

        // push_back
        // Note: w/df_ptr will be copied from node.w/df_ptr
        void push_back(core::ADNode<T> const& node)
        {if (this->vec.size() < this->vec.capacity()) this->vec.push_back(node);
            else throw std::length_error("maximum capacity reached.");}

        // emplace_back
        template <class ...Args>
        void emplace_back(Args&&... args)
        {if (this->vec.size() < this->vec.capacity()) this->vec.emplace_back(args...);
            else throw std::length_error("maximum capacity reached.");}

        // size wrapper
        inline size_t size() const
        {return vec.size();}

        // capacity wrapper
        inline size_t capacity() const
        {return vec.capacity();}

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
