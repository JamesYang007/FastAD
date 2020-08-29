#pragma once
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_adj_view.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/util/value.hpp>
#include <fastad_bits/util/size_pack.hpp>

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
    ValueAdjView<typename util::expr_traits<LHSExprType>::value_t,
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
    using value_adj_view_t = ValueAdjView<lhs_value_t,
          details::dot_shape_t<lhs_t, rhs_t> >;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    DotNode(const lhs_t& lhs,
            const rhs_t& rhs)
        : value_adj_view_t(nullptr, nullptr, lhs.rows(), rhs.cols())
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

    template <class T>
    void beval(const T& seed)
    {
        util::to_array(this->get_adj()) = seed;
        auto a_ladj = util::to_array(this->get_adj() * rhs_.get().transpose());
        auto a_radj = util::to_array(lhs_.get().transpose() * this->get_adj());
        rhs_.beval(a_radj);
        lhs_.beval(a_ladj);
    }

    ptr_pack_t bind_cache(ptr_pack_t begin)
    {
        begin = lhs_.bind_cache(begin);
        begin = rhs_.bind_cache(begin);
        return value_adj_view_t::bind(begin);
    }

    util::SizePack bind_cache_size() const 
    { 
        return single_bind_cache_size() + 
                lhs_.bind_cache_size() + 
                rhs_.bind_cache_size();
    }

    util::SizePack single_bind_cache_size() const
    {
        return {this->size(), this->size()};
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
        using var_t = util::constant_var_t<expr2_value_t, shape_t>;
        var_t out = expr1.feval() * expr2.feval();
        return ad::constant(out);
    } else {
        return core::DotNode<expr1_t, expr2_t>(expr1, expr2);
    }
}

} // namespace ad
