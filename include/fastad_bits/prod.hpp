#pragma once
#include "node.hpp"
#include "math.hpp"
#include "vec.hpp"
#include "dualnum.hpp"

namespace ad {
namespace core {

// ProdNode represents a product of functor on expressions.
// Ex.
// f(e1) * f(e2) * ... * f(en)
// @tparam  ValueType    underlying data type
// @tparam  Iter        type of iterator of values
// @tparam  Lmda        type of functor to apply on each value returning an expression
template <class ValueType, class Iter, class Lmda>
struct ProdNode :
    public DualNum<ValueType>, ADNodeExpr<ProdNode<ValueType, Iter, Lmda>>
{
    using data_t = DualNum<ValueType>;

    ProdNode(Iter start, Iter end, const Lmda& f)
        : data_t(0, 0), start_(start), end_(end), f_(f)
        , vec_(static_cast<size_t>(std::distance(start_, end_)))
    {}

    ProdNode(const ProdNode& node)
        : data_t(node), start_(node.start_), end_(node.end_), f_(node.f_)
        , vec_(node.vec_)
    {
        // Reset all adjoint destinations to point to current adjoints
        // and value destinations to point to current values.
        vec_.reset_value_ptr();
        vec_.reset_adjoint_ptr(); 
    }

    ProdNode(ProdNode&& node)
        : data_t(node)
        , start_(std::move(node.start_))
        , end_(std::move(node.end_))
        , f_(std::move(node.f_))
        , vec_(std::move(node.vec_))
    {
        vec_.reset_value_ptr();
        vec_.reset_adjoint_ptr(); 
    }

    ProdNode& operator=(const ProdNode& node)
    {
        if (this == &node) {
            return *this;
        }

        *this = node;
        start_ = node.start_;
        end_ = node.end_;
        f_ = node.f_;
        vec_ = node.vec_;
        vec_.reset_value_ptr();
        vec_.reset_adjoint_ptr(); 

        return *this;
    }

    // Forward evaluation.
    // See eval function below for more details.
    ValueType feval()
    {
        eval(false); 
        return this->get_value();
    }

    // Backward evaluation.
    // See eval function below for more details.
    void beval(ValueType seed)
    {
        eval(true, seed);
    }

private:
    // TODO: can feval and beval be separated?
    // If do_grad is false, only forward evaluates.
    // Otherwise, forward and backward evaluation is done.
    // Forward evaluation is simply a multiplication of all the functored expression forward values.
    // Backward evaluation is applied on every running product.
    void eval(bool do_grad, ValueType seed = static_cast<ValueType>(0))
    {
        auto&& first = make_eq(vec_[0], this->f_(*start_)); // create EqNode
        this->set_value(first.feval());
        if (std::distance(start_, end_) == 1 && !do_grad) return;
        else if (std::distance(start_, end_) == 1) {
            first.beval(seed);
            return;
        }
        auto it = start_;
        auto it_prev = vec_.begin();
        auto foreach = ad::for_each(std::next(vec_.begin()), vec_.end(), 
                [&, this](const typename Vec<ValueType>::value_type& x) {
                    return x = *(it_prev++) * (this->f_)(*(++it)); // returns an EqNode
                });
        this->set_value(foreach.feval());
        if (do_grad) {
            this->set_adjoint(seed);
            foreach.beval(this->get_adjoint()); 
            first.beval();
        }
    }

    Iter start_, end_;
    Lmda f_;
    Vec<ValueType> vec_;
};

} // namespace core

// ad::prod(Iter, Iter, Lmda&&)
// It is undefined behavaior if the lambda function returns
// an expression other than a LeafNode.
template <class Iter, class Lmda>
inline auto prod(Iter start, Iter end, Lmda&& f)
{
    using value_t = typename decltype(f(*start))::value_type;
    return core::ProdNode<value_t, Iter, Lmda>(start, end, std::forward<Lmda>(f));
}

} // namespace ad
