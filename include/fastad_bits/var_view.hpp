#pragma once
#include <fastad_bits/expr_base.hpp>
#include <fastad_bits/shape_traits.hpp>
#include <fastad_bits/value_view.hpp>
#include <Eigen/Core>

namespace ad {

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
private:
    using value_view_t = core::ValueView<ValueType, scl>;

public:
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::get;
    using value_view_t::bind;
    using value_view_t::size;
    using value_view_t::rows;
    using value_view_t::cols;
    using value_view_t::data;

    VarView(value_t* val_begin,
            value_t* adj_begin,
            size_t=1,
            size_t=1)
        : value_view_t{val_begin}
        , adj_{adj_begin}
    {}

    /* 
     * (leaf = non-leaf expression) returns EqNode
     */
    template <class Derived>
    inline auto operator=(const core::ExprBase<Derived>&) const;

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
    void beval(value_t seed, size_t, size_t) { adj_.get() += seed; }

    /**
     * Get underlying (full) adjoint.
     * @return  const reference to underlying adjoint.
     */
    const value_t& get_adj(size_t, size_t) const { return adj_.get(); }

    /**
     * Binds adjoint pointer to view the same adjoint that adj_begin points to.
     * @return  the next pointer from adj_begin that is not viewed by current object.
     */
    value_t* bind_adj(value_t* begin) { return adj_.bind(begin); }

    /**
     * Returns the raw pointer to the first adjoint viewed by current object.
     * @return  raw pointer
     */
    value_t* data_adj() const { return adj_.data(); }
    
    /**
     * Resets adjoints to all zeros.
     */
    void reset_adj() { return adj_.get() = 0; }

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
private:
    using value_view_t = core::ValueView<ValueType, vec>;

public:
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::get;
    using value_view_t::bind;
    using value_view_t::size;
    using value_view_t::rows;
    using value_view_t::cols;
    using value_view_t::data;

    VarView(value_t* val_begin,
            value_t* adj_begin,
            size_t rows,
            size_t = 1)
        : value_view_t(val_begin, rows)
        , adj_(adj_begin, rows)
    {}

    /* 
     * (leaf = non-leaf expression) returns EqNode
     */
    template <class Derived>
    inline auto operator=(const core::ExprBase<Derived>&) const;

    const var_t& feval() const { return this->get(); }
    void beval(value_t seed, size_t i, size_t) { adj_.get()(i) += seed; }
    const value_t& get_adj(size_t i, size_t) const { return adj_.get()(i); }
    value_t* bind_adj(value_t* begin) { return adj_.bind(begin); }
    value_t* data_adj() const { return adj_.data(); }

    size_t size() const {
        assert(value_view_t::size() == adj_.size());
        return value_view_t::size();
    }

    size_t rows() const {
        assert(value_view_t::rows() == adj_.rows());
        return value_view_t::rows();
    }

    void reset_adj() { adj_.get().setZero(); }

private:
    value_view_t adj_;
};

template <class ValueType>
struct VarView<ValueType, mat>: 
    core::ValueView<ValueType, mat>,
    core::ExprBase<VarView<ValueType, mat>>
{
private:
    using value_view_t = core::ValueView<ValueType, mat>;

public:
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::get;
    using value_view_t::bind;
    using value_view_t::size;
    using value_view_t::rows;
    using value_view_t::cols;
    using value_view_t::data;

    VarView(value_t* val_begin,
            value_t* adj_begin,
            size_t n_rows,
            size_t n_cols)
        : value_view_t(val_begin, n_rows, n_cols)
        , adj_(adj_begin, n_rows, n_cols)
    {}

    /* 
     * (leaf = non-leaf expression) returns EqNode
     */
    template <class Derived>
    inline auto operator=(const core::ExprBase<Derived>&) const;

    const var_t& feval() const { return this->get(); }
    void beval(value_t seed, size_t i, size_t j) { adj_.get()(i,j) += seed; }
    const value_t& get_adj(size_t i, size_t j) const { return adj_.get()(i,j); }
    value_t* bind_adj(value_t* begin) { return adj_.bind(begin); }

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

    value_t* data_adj() const { return adj_.data(); }
    void reset_adj() { adj_.get().setZero(); }

private:
    value_view_t adj_;
};

} // namespace ad