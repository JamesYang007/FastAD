#pragma once
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/util/shape_traits.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/reverse/core/value_view.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <Eigen/Core>

namespace ad {

// forward declaration
namespace core {

template <class VarViewType, class ExprType>
struct EqNode;
template <class Op, class VarViewType, class ExprType>
struct OpEqNode;
struct AddEq;
struct SubEq;
struct MulEq;
struct DivEq;

} // namespace core

/* 
 * VarView views a variable, which could be a scalar, vector, or matrix.
 * VarView objects are precisely the leaves of the computation tree.
 * VarView objects view the variable value(s) and partial derivative(s), or adjoint(s).
 *
 * ShapeType must be one of scl, vec, or mat.
 * All other specializations are disabled.
 *
 * @tparam ValueType    underlying data type
 * @tparam ShapeType    shape of variable (one of scl, vec, mat).
 *                      Default is scl.
 */

template <class ValueType
        , class ShapeType = scl>
struct VarView;

template <class ValueType>
struct VarView<ValueType, scl>: 
    core::ValueView<ValueType, scl>,
    core::ExprBase<VarView<ValueType, scl>>
{
    using value_view_t = core::ValueView<ValueType, scl>;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::get;
    using value_view_t::bind;
    using value_view_t::size;
    using value_view_t::rows;
    using value_view_t::cols;
    using value_view_t::data;

    VarView() : VarView(nullptr, nullptr) {}

    VarView(value_t* val_begin,
            value_t* adj_begin,
            size_t=1,
            size_t=1)
        : value_view_t{val_begin}
        , adj_{adj_begin}
    {}

    /* 
     * (leaf = non-leaf expression) returns EqNode
     * Optimized for constant expressions to simply copy into value.
     */
    template <class Derived
            , class = std::enable_if_t<
                util::is_convertible_to_ad_v<Derived>> >
    inline auto operator=(const Derived& x) const
    {
        using expr_t = util::convert_to_ad_t<Derived>;
        expr_t expr = x;
        return core::EqNode<VarView, expr_t>(*this, expr);
    }

    /** 
     * Forward-evaluation simply returns the underlying value.
     * @return  value destination value
     */
    const var_t& feval() const { return this->get(); }

    /**
     * Backward-evaluation increments the adjoint by seed.
     * Mathematically, seed is precisely a component of partial derivative
     * that was computed and passed down from the root of computation tree.
     *
     * The last two parameters indicate the i,jth function to backward evaluate.
     * If scalar, ignores both, if column vector, ignores second,
     * if row vector, ignores first, and if matrix, does not ignore either.
     */
    void beval(value_t seed, size_t, size_t, util::beval_policy) { adj_.get() += seed; }

    /**
     * Get underlying (full) adjoint.
     * @return  const reference to underlying adjoint.
     */
    const var_t& get_adj() const { return adj_.get(); }
    value_t& get_adj(size_t, size_t) { return adj_.get(); }
    const value_t& get_adj(size_t, size_t) const { return adj_.get(); }

    /**
     * Binds adjoint pointer to view the same adjoint that adj_begin points to.
     * @return  the next pointer from adj_begin that is not viewed by current object.
     */
    value_t* bind_adj(value_t* begin) { return adj_.bind(begin); }

    value_t* data_adj() { return adj_.data(); }
    const value_t* data_adj() const { return adj_.data(); }
    
    /**
     * Resets adjoints to all zeros.
     */
    void reset_adj() { adj_.get() = 0; }

    /**
     * Bind size is 0 since it will never get rebound once an expression is constructed.
     */
    constexpr size_t bind_size() const { return 0; }
    constexpr size_t single_bind_size() const { return 0; }

private:
    /* 
     * The current adjoint is only a component of the total derivative value.
     * Consider the following:
     *
     * f(g(x)) : R -> R^n -> R 
     * fog(x)' = grad(f)(g(x)) * g'(x) = sum_i df/dx_i * dg_i/dx
     *
     * The current adjoint will simply be df/dx_i * dg_i/dx for some i.
     * df_ptr_ points to the unique location that will accumulate all such components.
     * A single VarView will be copied in different parts of the expression and hence
     * will have to update different sub-adjoints.
     * However, every copy will have adj_ pointing to the same final adjoint location.
     */
    value_view_t adj_;
};

template <class ValueType>
struct VarView<ValueType, vec>: 
    core::ValueView<ValueType, vec>,
    core::ExprBase<VarView<ValueType, vec>>
{
    using value_view_t = core::ValueView<ValueType, vec>;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::get;
    using value_view_t::bind;
    using value_view_t::size;
    using value_view_t::rows;
    using value_view_t::cols;
    using value_view_t::data;

    VarView(size_t rows) 
        : VarView(nullptr, nullptr, rows, 0) {}

    VarView(value_t* val_begin,
            value_t* adj_begin,
            size_t rows,
            size_t = 1)
        : value_view_t(val_begin, rows)
        , adj_(adj_begin, rows)
    {}

    template <class Derived
            , class = std::enable_if_t<
                util::is_convertible_to_ad_v<Derived>> >
    inline auto operator=(const Derived& x) const
    {
        using expr_t = util::convert_to_ad_t<Derived>;
        expr_t expr = x;
        return core::EqNode<VarView, expr_t>(*this, expr);
    }

    const var_t& feval() const { return this->get(); }
    void beval(value_t seed, size_t i, size_t, util::beval_policy) { adj_.get()(i) += seed; }
    const var_t& get_adj() const { return adj_.get(); }
    value_t& get_adj(size_t i, size_t) { return adj_.get()(i); }
    const value_t& get_adj(size_t i, size_t) const { return adj_.get()(i); }
    value_t* bind_adj(value_t* begin) { return adj_.bind(begin); }
    value_t* data_adj() { return adj_.data(); }
    const value_t* data_adj() const { return adj_.data(); }

    size_t size() const {
        assert(value_view_t::size() == adj_.size());
        return value_view_t::size();
    }

    size_t rows() const {
        assert(value_view_t::rows() == adj_.rows());
        return value_view_t::rows();
    }

    void reset_adj() { adj_.get().setZero(); }
    constexpr size_t bind_size() const { return 0; }
    constexpr size_t single_bind_size() const { return 0; }

    // subviews
    auto operator()(size_t i, size_t=0) {
        assert(i < size());
        return VarView<value_t, ad::scl>(data() + i, data_adj() + i);
    }
    auto operator[](size_t i) {
        return operator()(i);
    }
    auto head(size_t n) {
        assert(n <= size());
        return VarView<value_t, ad::vec>(data(), data_adj(), n);
    }
    auto tail(size_t n) {
        assert(n <= size());
        size_t offset = size() - n;
        return VarView<value_t, ad::vec>(data() + offset, 
                                         data_adj() + offset,
                                         n);
    }

private:
    value_view_t adj_;
};

template <class ValueType>
struct VarView<ValueType, mat>: 
    core::ValueView<ValueType, mat>,
    core::ExprBase<VarView<ValueType, mat>>
{
    using value_view_t = core::ValueView<ValueType, mat>;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::get;
    using value_view_t::bind;
    using value_view_t::size;
    using value_view_t::rows;
    using value_view_t::cols;
    using value_view_t::data;

    VarView(size_t rows, size_t cols) 
        : VarView(nullptr, nullptr, rows, cols) {}

    VarView(value_t* val_begin,
            value_t* adj_begin,
            size_t n_rows,
            size_t n_cols)
        : value_view_t(val_begin, n_rows, n_cols)
        , adj_(adj_begin, n_rows, n_cols)
    {}

    template <class Derived
            , class = std::enable_if_t<
                util::is_convertible_to_ad_v<Derived>> >
    inline auto operator=(const Derived& x) const
    {
        using expr_t = util::convert_to_ad_t<Derived>;
        expr_t expr = x;
        return core::EqNode<VarView, expr_t>(*this, expr);
    }

    const var_t& feval() const { return this->get(); }
    void beval(value_t seed, size_t i, size_t j, util::beval_policy) { adj_.get()(i,j) += seed; }
    const var_t& get_adj() const { return adj_.get(); }
    value_t& get_adj(size_t i, size_t j) { return adj_.get()(i,j); }
    const value_t& get_adj(size_t i, size_t j) const { return adj_.get()(i,j); }
    value_t* bind_adj(value_t* begin) { return adj_.bind(begin); }
    value_t* data_adj() { return adj_.data(); }
    const value_t* data_adj() const { return adj_.data(); }

    size_t size() const { 
        assert(value_view_t::size() == adj_.size());
        return value_view_t::size(); 
    }

    size_t rows() const {
        assert(value_view_t::rows() == adj_.rows());
        return value_view_t::rows();
    }

    size_t cols() const {
        assert(value_view_t::cols() == adj_.cols());
        return value_view_t::cols();
    }

    void reset_adj() { adj_.get().setZero(); }
    constexpr size_t bind_size() const { return 0; }
    constexpr size_t single_bind_size() const { return 0; }

private:
    value_view_t adj_;
};

template <class ValueType>
struct VarView<ValueType, selfadjmat>: 
    core::ValueView<ValueType, mat>,    // note: this should be mat, not selfadjmat
    core::ExprBase<VarView<ValueType, selfadjmat>>
{
    using value_view_t = core::ValueView<ValueType, mat>;
    using typename value_view_t::value_t;
    using shape_t = selfadjmat;
    using typename value_view_t::var_t;
    using value_view_t::get;
    using value_view_t::bind;
    using value_view_t::size;
    using value_view_t::rows;
    using value_view_t::cols;
    using value_view_t::data;

    VarView(size_t rows, size_t cols) 
        : VarView(nullptr, nullptr, rows, cols) {}

    VarView(value_t* val_begin,
            value_t* adj_begin,
            size_t n_rows,
            size_t n_cols)
        : value_view_t(val_begin, n_rows, n_cols)
        , adj_(adj_begin, n_rows, n_cols)
        , val_flat_(nullptr, 0)
        , adj_flat_(nullptr, 0)
    {}

    // constructor only available if guaranteed to be square
    VarView(value_t* val_begin,
            value_t* val_flat_begin,
            value_t* adj_flat_begin,
            size_t n_rows)
        : value_view_t(val_begin, n_rows, n_rows)
        , adj_(nullptr, n_rows, n_rows) // unused
        , val_flat_(val_flat_begin, (n_rows * (n_rows+1)) / 2)
        , adj_flat_(adj_flat_begin, (n_rows * (n_rows+1)) / 2)
    {}

    template <class Derived
            , class = std::enable_if_t<
                util::is_convertible_to_ad_v<Derived>> >
    inline auto operator=(const Derived& x) const
    {
        using expr_t = util::convert_to_ad_t<Derived>;
        expr_t expr = x;
        return core::EqNode<VarView, expr_t>(*this, expr);
    }

    const var_t& feval() { 
        if (is_flat_set()) {
            size_t k = 0;
            // copy from flat vector into lower half matrix 
            for (size_t j = 0; j < adj_.cols(); ++j) {
                for (size_t i = j; i < adj_.rows(); ++i, ++k) {
                    this->get()(i,j) = val_flat_(k);
                }
            }
        }
        return this->get() = 
            this->get().template selfadjointView<Eigen::Lower>(); 
    }

    void beval(value_t seed, size_t i, size_t j, util::beval_policy) 
    { 
        if (is_flat_set()) {
            adj_flat_(flat_idx(i,j)) += seed;
        } else {
            if (i >= j) adj_.get()(i,j) += seed; 
            else adj_.get()(j,i) += seed;
        }
    }

    const var_t& get_adj() const { 
        return (is_flat_set()) ? adj_flat_ : adj_.get(); 
    }

    value_t& get_adj(size_t i, size_t j) { 
        return (is_flat_set()) ? adj_flat_(flat_idx(i,j)) : adj_.get()(i,j); 
    }

    const value_t& get_adj(size_t i, size_t j) const { 
        return (is_flat_set()) ? adj_flat_(flat_idx(i,j)) : adj_.get()(i,j); 
    }

    value_t* bind_adj(value_t* begin) { 
        assert(!is_flat_set());
        return adj_.bind(begin); 
    }

    value_t* data_adj() { 
        return (is_flat_set()) ? adj_flat_.data() : adj_.data(); 
    }

    const value_t* data_adj() const { 
        return (is_flat_set()) ? adj_flat_.data() : adj_.data(); 
    }

    size_t size() const { 
        assert(value_view_t::size() == adj_.size());
        return value_view_t::size(); 
    }

    size_t rows() const {
        assert(value_view_t::rows() == adj_.rows());
        return value_view_t::rows();
    }

    size_t cols() const {
        assert(value_view_t::cols() == adj_.cols());
        return value_view_t::cols();
    }

    void reset_adj() { 
        if (is_flat_set()) adj_flat_.setZero();
        else adj_.get().setZero(); 
    }

    constexpr size_t bind_size() const { return 0; }
    constexpr size_t single_bind_size() const { return 0; }

private:

    size_t flat_idx(size_t i, size_t j) const {
        return (i >= j) ? 
            ((rows() * j) - (j * (j+1)) / 2) + i:
            ((rows() * i) - (i * (i+1)) / 2) + j;
    }

    bool is_flat_set() const {
        return val_flat_.data() != nullptr &&
               adj_flat_.data() != nullptr;
    }

    using vec_view_t = util::shape_to_raw_view_t<value_t, ad::vec>;
    value_view_t adj_;
    vec_view_t val_flat_;
    vec_view_t adj_flat_;
};

/*
 * Useful operator overloads
 */
#define ADNODE_OPEQ_FUNC(name, strct) \
    template <class ValueType \
            , class ShapeType \
            , class Derived \
            , class = std::enable_if_t< \
                util::is_convertible_to_ad_v<Derived>> > \
    inline auto name(const VarView<ValueType, ShapeType>& var, \
                     const Derived& x)  \
    { \
        using var_view_t = VarView<ValueType, ShapeType>; \
        using expr_t = util::convert_to_ad_t<Derived>; \
        expr_t expr = x; \
        return core::OpEqNode<core::strct, var_view_t, expr_t>(var, expr); \
    } 

ADNODE_OPEQ_FUNC(operator+=, AddEq)
ADNODE_OPEQ_FUNC(operator-=, SubEq)
ADNODE_OPEQ_FUNC(operator*=, MulEq)
ADNODE_OPEQ_FUNC(operator/=, DivEq)

} // namespace ad
