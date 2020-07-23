#pragma once
#include <vector>
#include <algorithm>
#include "dualnum.hpp"

namespace ad {
namespace core {

//====================================================================================================
// Basic Node Structures 

//template <class ValueType>
//struct LeafNode:
//{
//
//    // This function sets the value in value dest as well current value.
//    // It is crucial for ForEach::feval to work properly that the value dest be provided.
//    // Note that calling set_value on any copy of current LeafNode will also overwrite
//    // current LeafNode's value in its value dest.
//    // It is best practice to create only expressions on LeafNodes, 
//    // rather than calling this function.
//    // @param   w   the new value to set current value and value dest to
//    // @return w
//    ValueType& set_value(ValueType w)
//    {
//        return *w_ptr_ = data_t::set_value(w);
//    }
//
//    // This function sets both current adjoint and adjoint dest to df.
//    // @param   df  the new adjoint to set current adjoint and adjoint dest to
//    // @return  df
//    ValueType& set_adjoint(ValueType df)
//    {
//        return *df_ptr_ = data_t::set_adjoint(df);
//    }
//
//private:
//    // explicitly declare some DualNum member functions private
//    using data_t::get_value;
//    using data_t::set_value;
//    using data_t::get_adjoint;
//    using data_t::set_adjoint;
//
//    ValueType* w_ptr_;
//    ValueType* df_ptr_; 
//};


//======================================================================================
// Advanced Nodes

namespace core {

// ForEach represents collection of expressions to evaluate.
// It can be thought of as a generalization of GlueNode.
// Applies functor on every iterated values as an expression.
// @tparam  ValueType   underlying data type
// @tparam  Iter        type of iterator on any values
// @tparam  Lmda        type of functor to apply on iterated values that returns an expression
template <class ValueType, class Iter, class Lmda>
struct ForEach: 
    public DualNum<ValueType>, ADNodeExpr<ForEach<ValueType, Iter, Lmda>>
{
    using data_t = DualNum<ValueType>;
    using expr_t = decltype(std::declval<Lmda>()(*std::declval<Iter>()));

    ForEach(Iter start, Iter end, Lmda f)
        : data_t(0, 0), start_(start), end_(end), f_(f), vec_()
    {
        // Store return of functored value as a new expression in vec_.
        std::for_each(start_, end_, 
                [this](const typename std::iterator_traits<Iter>::value_type& i) {
                    this->vec_.emplace_back(this->f_(i)); 
                }
        );
    }

    // Forward evaluation on every functored expressions.
    // @return  last functored expression forward evaluation value
    ValueType feval()
    {
        if (vec_.size() == 0) return this->set_value(0);
        std::for_each(vec_.begin(), vec_.end(), 
                [](expr_t& expr) {
                    expr.feval(); 
                }
        );
        return this->set_value(vec_[vec_.size() - 1].get_value());
    }

    // Backward evaluation first sets current adjoint to seed,
    // seeds the last functored expression with seed,
    // and backward evaluates every expression in reverse order with 0 seed.
    // See GlueNode::beval for reasons for this design choice.
    void beval(ValueType seed)
    {
        this->set_adjoint(seed);
        if (vec_.size() == 0) return;
        auto it = vec_.rbegin();
        it->beval(seed);
        std::for_each(std::next(it), vec_.rend(), 
                [](expr_t& expr) {
                    expr.beval(0); 
                }
        );
    }

private:
    Iter start_, end_;
    Lmda f_;
    std::vector<expr_t> vec_;
};

} // namespace core

// ad::for_each(Iter, Iter, Lmda&&)
template <class Iter, class Lmda>
inline auto for_each(Iter start, Iter end, const Lmda& f)
{
    using value_t = typename std::decay_t<decltype(f(*start))>::value_type;
    return core::ForEach<value_t, Iter, Lmda>(start, end, f);
}

} // namespace ad
