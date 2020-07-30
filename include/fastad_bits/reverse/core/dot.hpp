#pragma once
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_view.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/util/type_traits.hpp>

namespace ad {
namespace core {
namespace details {

/*
 * Returns the the dot-product shape given left and right shapes
 */
template <class T, class U, class=void>
struct dot_shape;

template <class T, class U>
struct dot_shape<T, U, std::enable_if_t<
                        util::is_mat_v<T> &&
                        util::is_vec_v<U>> >
{
    using type = ad::vec;
};

template <class T, class U>
struct dot_shape<T, U, std::enable_if_t<
                        util::is_mat_v<T> &&
                        util::is_mat_v<U>> >
{
    using type = ad::mat;
};

template <class T, class U>
using dot_shape_t = typename dot_shape<T,U>::type;

} // namespace details

/**
 * DotNode represents a matrix multiplication.
 * Indeed, the left expression must be a matrix shape, and the right be a matrix or column vector.
 * No other shapes are permitted for this node.
 * At construction, the actual sizes of the two are checked -
 * specifically, the number of columns for matrix must equal the number of rows for vector.
 *
 * We assert that the value type be the same for the two expressions.
 * The output shape is always a (column) vector.
 *
 * @tparam  LHSExprType     type of left expression
 * @tparam  RHSExprType     type of right expression
 */

template <class LHSExprType
        , class RHSExprType>
struct DotNode:
    ValueView<typename util::expr_traits<LHSExprType>::value_t,
              details::dot_shape_t<LHSExprType, RHSExprType>>,
    ExprBase<DotNode<LHSExprType, RHSExprType>>
{
private:
    using lhs_t = LHSExprType;
    using rhs_t = RHSExprType;
    using lhs_value_t = typename 
        util::expr_traits<lhs_t>::value_t;

    // assert that both expressions have same value type
    static_assert(std::is_same_v<
            typename util::expr_traits<lhs_t>::value_t,
            typename util::expr_traits<rhs_t>::value_t>);

public:
    using value_view_t = ValueView<lhs_value_t,
          details::dot_shape_t<lhs_t, rhs_t> >;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::bind;

    DotNode(const lhs_t& lhs,
            const rhs_t& rhs)
        : value_view_t(nullptr, lhs.rows(), rhs.cols())
        , lhs_{lhs}
        , rhs_{rhs}
    {
        assert(lhs.cols() == rhs.rows());
    }

    const var_t& feval()
    {
        auto&& lhs_val = lhs_.feval();
        auto&& rhs_val = rhs_.feval();
        return this->get() = lhs_val * rhs_val;
    }

    void beval(value_t seed, size_t i, size_t j, util::beval_policy pol)
    {
        if (seed == 0) return;
        for (size_t k = 0; k < rhs_.rows(); ++k) {
            rhs_.beval(seed * lhs_.get(i,k), k, j, pol);
        }
        for (size_t k = 0; k < lhs_.cols(); ++k) {
            lhs_.beval(seed * rhs_.get(k,j), i, k, pol);
        }
    }

    value_t* bind(value_t* begin) 
    {
        value_t* next = begin;
        if constexpr (!util::is_var_view_v<lhs_t>) {
            next = lhs_.bind(next);
        }
        if constexpr (!util::is_var_view_v<rhs_t>) {
            next = rhs_.bind(next);
        }
        return value_view_t::bind(next);
    }

    size_t bind_size() const 
    { 
        return single_bind_size() + 
                lhs_.bind_size() + 
                rhs_.bind_size();
    }

    size_t single_bind_size() const
    {
        return this->size();
    }


private:
    lhs_t lhs_;
    rhs_t rhs_;
};

} // namespace core

template <class T1
        , class T2
        , class = std::enable_if_t<
            !util::is_scl_v<util::convert_to_ad_t<T1>> &&
            !util::is_scl_v<util::convert_to_ad_t<T2>> &&
            util::any_ad_v<T1, T2>
        >
    >
inline auto dot(const T1& x,
                const T2& y)
{
    using expr1_t = util::convert_to_ad_t<T1>;
    using expr2_t = util::convert_to_ad_t<T2>;
    using expr1_value_t = typename util::expr_traits<expr1_t>::value_t;
    using expr2_value_t = typename util::expr_traits<expr2_t>::value_t;

    expr1_t expr1 = x;
    expr2_t expr2 = y;

    // optimization for when both expressions are constant
    if constexpr (util::is_constant_v<expr1_t> &&
                  util::is_constant_v<expr2_t>) {
        static_assert(std::is_same_v<expr1_value_t, expr2_value_t>);
        using shape_t = core::details::dot_shape_t<expr1_t, expr2_t>;
        using var_t = core::details::constant_var_t<expr2_value_t, shape_t>;
        var_t out = expr1.feval() * expr2.feval();
        return ad::constant(out);
    } else {
        return core::DotNode<expr1_t, expr2_t>(
                expr1, expr2);
    }
}

} // namespace ad
