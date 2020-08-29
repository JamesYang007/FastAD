#pragma once
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/util/shape_traits.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/util/value.hpp>
#include <fastad_bits/reverse/core/value_adj_view.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <Eigen/Core>

namespace ad {

// forward declaration
template <class ValueType
        , class ShapeType=scl>
struct VarView;

namespace core {

template <class VarViewType, class ExprType>
struct EqNode;
template <class Op, class VarViewType, class ExprType>
struct OpEqNode;
struct AddEq;
struct SubEq;
struct MulEq;
struct DivEq;

template <class VarViewType>
struct VarViewBase;

template <class ValueType
        , class ShapeType>
struct VarViewBase<VarView<ValueType, ShapeType>>:
    core::ValueAdjView<ValueType, ShapeType>,
    core::ExprBase<VarView<ValueType, ShapeType>>
{
    using value_adj_view_t = core::ValueAdjView<ValueType, ShapeType>;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;
    using var_view_t = VarView<value_t, shape_t>;

    VarViewBase(size_t rows, size_t cols) 
        : VarViewBase(nullptr, nullptr, rows, cols) {}

    VarViewBase(value_t* val,
                value_t* adj,
                size_t rows,
                size_t cols)
        : value_adj_view_t(val, adj, rows, cols)
    {}

    template <class Derived
            , class = std::enable_if_t<
                util::is_convertible_to_ad_v<Derived>> >
    inline auto operator=(const Derived& x) const
    {
        using expr_t = util::convert_to_ad_t<Derived>;
        expr_t expr = x;
        return EqNode<var_view_t, expr_t>(
                static_cast<const var_view_t&>(*this), expr);
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
     *
     * Keep it templated since seed can be scalar or Eigen type regardless of current var view shape.
     * Helper to_array function converts them properly to make the operation make sense in all cases.
     */
    template <class T>
    void beval(const T& seed) { 
        util::to_array(this->get_adj()) += seed; 
    }

    /**
     * Cache bind size is 0 since it will never get rebound once an expression is constructed.
     */
    template <class T>
    constexpr T bind_cache(T begin) { return begin; }
    util::SizePack bind_cache_size() const { return {0,0}; }
    util::SizePack single_bind_cache_size() const { return {0,0}; }
};

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

template <class ValueType>
struct VarView<ValueType, scl>: 
    core::VarViewBase<VarView<ValueType, scl>>
{
    using base_t = core::VarViewBase<VarView<ValueType, scl>>;
    using typename base_t::value_t;
    using base_t::operator=;

    VarView() : VarView(nullptr, nullptr) {}

    VarView(value_t* val,
            value_t* adj,
            size_t=1,
            size_t=1)
        : base_t(val, adj, 1, 1)
    {}
};

template <class ValueType>
struct VarView<ValueType, vec>: 
    core::VarViewBase<VarView<ValueType, vec>>
{
    using base_t = core::VarViewBase<VarView<ValueType, vec>>;
    using typename base_t::value_t;
    using base_t::operator=;

    VarView(value_t* val,
            value_t* adj,
            size_t rows,
            size_t = 1)
        : base_t(val, adj, rows, 1)
    {}

    // subviews
    auto operator()(size_t i) {
        assert(i < base_t::size());
        return VarView<value_t, scl>(base_t::data() + i, 
                                     base_t::data_adj() + i);
    }
    auto operator[](size_t i) {
        return operator()(i);
    }
    auto head(size_t n) {
        assert(n <= base_t::size());
        return VarView(base_t::data(), base_t::data_adj(), n);
    }
    auto tail(size_t n) {
        assert(n <= base_t::size());
        size_t offset = base_t::size() - n;
        return VarView(base_t::data() + offset, 
                       base_t::data_adj() + offset,
                       n);
    }
};

template <class ValueType>
struct VarView<ValueType, mat>: 
    core::VarViewBase<VarView<ValueType, mat>>
{
    using base_t = core::VarViewBase<VarView<ValueType, mat>>;
    using typename base_t::value_t;
    using base_t::operator=;

    VarView(value_t* val,
            value_t* adj,
            size_t rows,
            size_t cols)
        : base_t(val, adj, rows, cols)
    {}
};

// Explicit template instantiation to help compile-time
template struct VarView<double, scl>;
template struct VarView<double, vec>;
template struct VarView<double, mat>;

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
