#pragma once
#include <tuple>
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_adj_view.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/util/numeric.hpp>
#include <Eigen/Dense>

namespace ad {
namespace stat {
namespace details {

template <class XExprType
        , class VExprType
        , class NExprType>
struct WishartBase:
    core::ValueAdjView<util::common_value_t<
                        XExprType, 
                        VExprType>, ad::scl>
{
    static_assert(util::is_mat_v<XExprType>);
    static_assert(util::is_mat_v<VExprType>);
    static_assert(util::is_scl_v<NExprType>);
    static_assert(util::is_constant_v<NExprType>);

    using x_t = XExprType;
    using v_t = VExprType;
    using n_t = NExprType;
    using common_value_t = util::common_value_t<x_t, v_t>;
    using value_adj_view_t = core::ValueAdjView<common_value_t, ad::scl>;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    WishartBase(const x_t& x,
                const v_t& v,
                const n_t& n)
        : value_adj_view_t(nullptr, nullptr, 1, 1)
        , x_{x}
        , v_{v}
        , n_{n}
    {}

    // Note: we do not bind for n_ since it is a constant
    ptr_pack_t bind_cache(ptr_pack_t begin)
    {
        begin = x_.bind_cache(begin);
        begin = v_.bind_cache(begin);
        auto adj = begin.adj;
        begin.adj = nullptr;
        begin = value_adj_view_t::bind(begin);
        begin.adj = adj;
        return begin;
    }

    util::SizePack bind_cache_size() const 
    { 
        return single_bind_cache_size() + 
                x_.bind_cache_size() +
                v_.bind_cache_size() +
                n_.bind_cache_size();
    }

    util::SizePack single_bind_cache_size() const
    { 
        return {this->size(), 0}; 
    }

protected:
    x_t x_;
    v_t v_;
    n_t n_;
};

} // namespace details

/**
 * WishartAdjLogPDFNode represents the adjusted log-pdf of Wishart distribution,
 * omitting any constants.
 *
 * It assumes the value type that is common to the two expressions.
 * Since it represents a log-pdf, it is always a scalar expression.
 *
 * Note: n MUST be a constant.
 *
 * The only possible shape combinations are as follows:
 * x -> matrix (or self-adj), 
 * v -> matrix (or self-adj)
 * n -> scalar
 *
 * No other shapes are permitted for this node.
 *
 * @tparam  XExprType           type of x expression at which to evaluate log-pdf
 * @tparam  VExprType           type of V expression
 * @tparam  NExprType           type of n expression
 */
template <class XExprType
        , class VExprType
        , class NExprType
        , class = std::tuple<
            typename util::shape_traits<XExprType>::shape_t,
            typename util::shape_traits<VExprType>::shape_t,
            typename util::shape_traits<NExprType>::shape_t> >
struct WishartAdjLogPDFNode;

template <class XExprType
        , class VExprType
        , class NExprType>
struct WishartAdjLogPDFNode<XExprType,
                            VExprType,
                            NExprType,
                            std::tuple<
                                std::enable_if_t<
                                    util::is_mat_v<XExprType>, 
                                    typename util::shape_traits<XExprType>::shape_t>,
                                std::enable_if_t<
                                    util::is_mat_v<VExprType>, 
                                    typename util::shape_traits<VExprType>::shape_t>,
                                scl> >:
    details::WishartBase<XExprType, VExprType, NExprType>,
    core::ExprBase<WishartAdjLogPDFNode<XExprType, VExprType, NExprType>>
{
private:
    using base_t = details::WishartBase<XExprType, VExprType, NExprType>;

public:
    using typename base_t::x_t;
    using typename base_t::v_t;
    using typename base_t::n_t;
    using typename base_t::value_t;
    using typename base_t::var_t;
    using base_t::x_;
    using base_t::v_;
    using base_t::n_;

    WishartAdjLogPDFNode(const x_t& x,
                         const v_t& v,
                         const n_t& n)
        : base_t(x, v, n)
        , x_llt_(x.rows())
        , v_llt_(v.rows())
        , log_x_det_(0)
        , log_v_det_(0)
        , is_x_pos_def_(false)
        , is_v_pos_def_(false)
        , x_inv_(x.rows(), x.cols())
        , v_inv_(v.rows(), v.cols())
        , xv_inv_(x.rows(), v.rows())
    {
        if constexpr (util::is_constant_v<v_t>) {
            update_v_cache();
        }
        if constexpr (util::is_constant_v<x_t>) {
            update_x_cache();
        }
    }

    const var_t& feval()
    {
        x_.feval();
        v_.feval();
        auto&& n = n_.feval();

        if constexpr (!util::is_constant_v<v_t>) {
            update_v_cache();
        }
        if constexpr (!util::is_constant_v<x_t>) {
            update_x_cache();
        }

        if (!valid()) {
            return this->get() = util::neg_inf<value_t>;
        }

        value_t p = v_.rows();
        return this->get() = (n-p-1.) * log_x_det_ 
                              - 0.5 * xv_inv_.trace()
                              - n * log_v_det_;
    }

    void beval(value_t seed)
    {
        if (seed == 0 || !valid()) return;

        value_t n = n_.get();
        value_t p = v_.rows();

        auto x_adj = (0.5 * seed) * ((n-p-1) * x_inv_ - v_inv_);
        auto v_adj = (0.5 * seed) * (v_inv_ * xv_inv_ - n * v_inv_);
        v_.beval(v_adj.array());
        x_.beval(x_adj.array());
    }

private:
    void update_v_cache() {
        v_llt_.compute(v_.get());
        is_v_pos_def_ = (v_llt_.info() == Eigen::Success);
        if (is_v_pos_def_) {
            log_v_det_ = std::log(v_llt_.matrixL().determinant());
            v_inv_ = v_llt_.solve(mat_t::Identity(v_.rows(), v_.cols()));
        }
    }

    void update_x_cache() {
        if (is_v_pos_def_) {
            x_llt_.compute(x_.get());
            is_x_pos_def_ = (x_llt_.info() == Eigen::Success);
            if (is_x_pos_def_) {
                log_x_det_ = std::log(x_llt_.matrixL().determinant());
                x_inv_ = x_llt_.solve(mat_t::Identity(x_.rows(), x_.cols()));
                xv_inv_ = x_.get() * v_inv_;
            }
        }
    }

    bool valid() { 
        return is_x_pos_def_ && 
               is_v_pos_def_ && 
               (n_.get() + 1 > v_.rows());
    }

    using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;

    Eigen::LLT<mat_t, Eigen::Lower> x_llt_;
    Eigen::LLT<mat_t, Eigen::Lower> v_llt_;
    value_t log_x_det_;
    value_t log_v_det_;
    bool is_x_pos_def_;
    bool is_v_pos_def_;
    mat_t x_inv_;
    mat_t v_inv_;
    mat_t xv_inv_;
};

} // namespace stat

template <class XType
        , class VType
        , class NType
        , class = std::enable_if_t<
            util::is_convertible_to_ad_v<XType> &&
            util::is_convertible_to_ad_v<VType> &&
            util::is_convertible_to_ad_v<NType> &&
            util::any_ad_v<XType, VType, NType> > >
inline auto wishart_adj_log_pdf(const XType& x,
                                const VType& v,
                                const NType& n)
{
    using x_expr_t = util::convert_to_ad_t<XType>;
    using v_expr_t = util::convert_to_ad_t<VType>;
    using n_expr_t = util::convert_to_ad_t<NType>;
    x_expr_t x_expr = x;
    v_expr_t v_expr = v;
    n_expr_t n_expr = n;
    return stat::WishartAdjLogPDFNode<
        x_expr_t, v_expr_t, n_expr_t>(x_expr, v_expr, n_expr);
}

} // namespace ad
