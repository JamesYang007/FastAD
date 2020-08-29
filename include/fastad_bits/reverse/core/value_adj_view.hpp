#pragma once
#include <fastad_bits/reverse/core/value_view.hpp>
#include <fastad_bits/util/ptr_pack.hpp>

namespace ad {
namespace core {

/**
 * This class wraps the more fundamental value viewer
 * to view both value and adjoints.
 * The same value viewer can be used to view both.
 * Only extra API is exposed for the user.
 * It is assumed that whatever entity views values and its corresponding adjoints,
 * they are of the same shape and size.
 * This is because our system assumes that the final expression to evaluate will
 * always be a scalar function (if vector-function, then differentiate one at a time),
 * hence the adjoints will always be of the form 
 *
 * \partial f / \partial w
 *
 * where it has the same shape and size as w.
 */

template <class ValueType, class ShapeType>
struct ValueAdjView
    : ValueView<ValueType, ShapeType>
{
    using base_t = ValueView<ValueType, ShapeType>;
    using typename base_t::value_t;
    using typename base_t::shape_t;
    using typename base_t::var_t;
    using ptr_pack_t = util::PtrPack<value_t>;

    ValueAdjView(value_t* val, 
                 value_t* adj,
                 size_t rows=1, 
                 size_t cols=1)
        : base_t(val, rows, cols)
        , adj_view_(adj, rows, cols)
    {}
     
    var_t& get_adj() { return adj_view_.get(); }
    const var_t& get_adj() const { return adj_view_.get(); }
    value_t& get_adj(size_t i, size_t j) { return adj_view_.get(i,j); }
    const value_t& get_adj(size_t i, size_t j) const { return adj_view_.get(i,j); }

    ptr_pack_t bind(ptr_pack_t begin)
    { 
        begin.val = base_t::bind(begin.val);
        begin.adj = adj_view_.bind(begin.adj);
        return begin;
    }

    value_t* data_adj() { return adj_view_.data(); }
    const value_t* data_adj() const { return adj_view_.data(); }
    void zero_adj() { adj_view_.zero(); }
    void ones_adj() { adj_view_.ones(); }
    void reset_adj() { adj_view_.zero(); }

private:
    base_t adj_view_;
};

} // namespace core
} // namespace ad
