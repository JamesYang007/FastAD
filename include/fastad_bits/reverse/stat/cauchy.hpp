#pragma once
#include <tuple>
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_adj_view.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/util/value.hpp>
#include <fastad_bits/util/numeric.hpp>

namespace ad {
namespace stat {
namespace details {

template <class XExprType
        , class LocExprType
        , class ScaleExprType>
struct CauchyBase:
    core::ValueAdjView<util::common_value_t<
                        XExprType, 
                        LocExprType, 
                        ScaleExprType>, ad::scl>
{
    using x_t = XExprType;
    using loc_t = LocExprType;
    using scale_t = ScaleExprType;
    using common_value_t = util::common_value_t<
        x_t, loc_t, scale_t>;
    using value_adj_view_t = core::ValueAdjView<common_value_t, ad::scl>;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    CauchyBase(const x_t& x,
               const loc_t& loc,
               const scale_t& scale)
        : value_adj_view_t(nullptr, nullptr, 1, 1)
        , x_{x}
        , loc_{loc}
        , scale_{scale}
    {}

    ptr_pack_t bind_cache(ptr_pack_t begin)
    {
        begin = x_.bind_cache(begin);
        begin = loc_.bind_cache(begin);
        begin = scale_.bind_cache(begin);
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
                loc_.bind_cache_size() +
                scale_.bind_cache_size();
    }

    util::SizePack single_bind_cache_size() const
    {
        return {this->size(), 0}; 
    }

protected:
    x_t x_;
    loc_t loc_;
    scale_t scale_;
};

} // namespace details

/**
 * CauchyAdjLogPDFNode represents the cauchy log pdf 
 * adjusted to omit all fixed constants.
 *
 * It assumes the value type that is common to all three expressions.
 * Since it represents a log-pdf, it is always a scalar expression.
 *
 * The only possible shape combinations are as follows:
 * x -> scalar, loc -> scalar, scale -> scalar
 * x -> vec, loc -> scalar | vector, scale -> scalar | vector
 *
 * No other shapes are permitted for this node.
 *
 * @tparam  XExprType           type of x expression at which to evaluate log-pdf
 * @tparam  LocExprType         type of loc expression
 * @tparam  ScaleExprType       type of scale expression
 */
template <class XExprType
        , class LocExprType
        , class ScaleExprType
        , class = std::tuple<
            typename util::shape_traits<XExprType>::shape_t,
            typename util::shape_traits<LocExprType>::shape_t,
            typename util::shape_traits<ScaleExprType>::shape_t> >
struct CauchyAdjLogPDFNode;

// Case 1: sss
template <class XExprType
        , class LocExprType
        , class ScaleExprType>
struct CauchyAdjLogPDFNode<XExprType,
                            LocExprType,
                            ScaleExprType,
                            std::tuple<scl, scl, scl> >:
    details::CauchyBase<XExprType, LocExprType, ScaleExprType>,
    core::ExprBase<CauchyAdjLogPDFNode<XExprType, LocExprType, ScaleExprType>>
{
private:
    using base_t = details::CauchyBase<
        XExprType, LocExprType, ScaleExprType>;
    
public:
    using typename base_t::x_t;
    using typename base_t::loc_t;
    using typename base_t::scale_t;
    using typename base_t::value_t;
    using typename base_t::var_t;
    using base_t::x_;
    using base_t::loc_;
    using base_t::scale_;

    CauchyAdjLogPDFNode(const x_t& x,
                        const loc_t& loc,
                        const scale_t& scale)
        : base_t(x, loc, scale)
        , inner_term_(0)
    {}

    const var_t& feval()
    {
        auto&& x = x_.feval();
        auto&& x0 = loc_.feval();
        auto&& gamma = scale_.feval();

        if (!within_range()) {
            return this->get() = util::neg_inf<value_t>;
        }

        auto diff = x-x0;
        inner_term_ = gamma + (diff * diff) / gamma;
        return this->get() = -std::log(inner_term_);
    }

    void beval(value_t seed)
    {
        if (seed == 0 || !within_range()) return;

        auto&& x = x_.get();
        auto&& x0 = loc_.get();
        auto&& gamma = scale_.get();

        auto diff = x-x0;
        auto x0_adj = 2. * diff / (gamma * inner_term_);
        auto x_adj = -x0_adj;
        auto gamma_adj = 1./gamma * (x0_adj * diff - 1);

        scale_.beval(seed * gamma_adj);
        loc_.beval(seed * x0_adj);
        x_.beval(seed * x_adj);
    }

private:
    bool within_range() const {
        return scale_.get() > 0;
    }

    value_t inner_term_;
};

// Case 2: vss
template <class XExprType
        , class LocExprType
        , class ScaleExprType>
struct CauchyAdjLogPDFNode<XExprType,
                            LocExprType,
                            ScaleExprType,
                            std::tuple<vec, scl, scl> >:
    details::CauchyBase<XExprType, LocExprType, ScaleExprType>,
    core::ExprBase<CauchyAdjLogPDFNode<XExprType, LocExprType, ScaleExprType>>
{
private:
    using base_t = details::CauchyBase<
        XExprType, LocExprType, ScaleExprType>;
    
public:
    using typename base_t::x_t;
    using typename base_t::loc_t;
    using typename base_t::scale_t;
    using typename base_t::value_t;
    using typename base_t::var_t;
    using base_t::x_;
    using base_t::loc_;
    using base_t::scale_;

    CauchyAdjLogPDFNode(const x_t& x,
                        const loc_t& loc,
                        const scale_t& scale)
        : base_t(x, loc, scale)
    {}

    const var_t& feval()
    {
        auto&& x = x_.feval();
        auto&& x0 = loc_.feval();
        auto&& gamma = scale_.feval();

        if (!within_range()) {
            return this->get() = util::neg_inf<value_t>;
        }

        auto diff_sq = (x.array() - x0).square();
        return this->get() = -(gamma + (1./gamma) * diff_sq).log().sum();
    }

    void beval(value_t seed)
    {
        if (seed == 0 || !within_range()) return;

        auto&& x = x_.get().array();
        auto&& x0 = loc_.get();
        auto&& gamma = scale_.get();
        auto gamma_sq = gamma * gamma;

        auto diff = (x - x0);
        auto dx = (-2. * seed) * diff / (gamma_sq + diff.square());
        value_t dx0 = (-seed) * dx.sum();
        value_t dgamma = (-seed/gamma) * ((dx * diff).sum() + x.size());
        
        scale_.beval(dgamma);
        loc_.beval(dx0);
        x_.beval(dx);
    }

private:
    bool within_range() const {
        return scale_.get() > 0;
    }
};

// Case 3: vsv
template <class XExprType
        , class LocExprType
        , class ScaleExprType>
struct CauchyAdjLogPDFNode<XExprType,
                            LocExprType,
                            ScaleExprType,
                            std::tuple<vec, scl, vec> >:
    details::CauchyBase<XExprType, LocExprType, ScaleExprType>,
    core::ExprBase<CauchyAdjLogPDFNode<XExprType, LocExprType, ScaleExprType>>
{
private:
    using base_t = details::CauchyBase<
        XExprType, LocExprType, ScaleExprType>;
    
public:
    using typename base_t::x_t;
    using typename base_t::loc_t;
    using typename base_t::scale_t;
    using typename base_t::value_t;
    using typename base_t::var_t;
    using base_t::x_;
    using base_t::loc_;
    using base_t::scale_;

    CauchyAdjLogPDFNode(const x_t& x,
                        const loc_t& loc,
                        const scale_t& scale)
        : base_t(x, loc, scale)
    {}

    const var_t& feval()
    {
        auto&& x = x_.feval().array();
        auto&& x0 = loc_.feval();
        auto&& gamma = scale_.feval().array();

        if (!within_range()) {
            return this->get() = util::neg_inf<value_t>;
        }
        
        auto diff = x - x0;
        return this->get() = -(gamma + (diff.square() / gamma)).log().sum();
    }

    void beval(value_t seed)
    {
        if (seed == 0 || !within_range()) return;

        auto&& x = x_.get().array();
        auto&& x0 = loc_.get();
        auto&& gamma = scale_.get().array();

        auto diff = x - x0;
        auto dx = (-2. * seed) * diff / (gamma.square() + diff.square());
        value_t dx0 = (-seed) * dx.sum();
        auto dgamma = (-seed) * (dx * diff + 1) / gamma;

        scale_.beval(dgamma);
        loc_.beval(dx0);
        x_.beval(dx);
    }

private:
    bool within_range() const {
        return (scale_.get().array() > 0).all();
    }
};

// Case 4: vvs
template <class XExprType
        , class LocExprType
        , class ScaleExprType>
struct CauchyAdjLogPDFNode<XExprType,
                            LocExprType,
                            ScaleExprType,
                            std::tuple<vec, vec, scl> >:
    details::CauchyBase<XExprType, LocExprType, ScaleExprType>,
    core::ExprBase<CauchyAdjLogPDFNode<XExprType, LocExprType, ScaleExprType>>
{
private:
    using base_t = details::CauchyBase<
        XExprType, LocExprType, ScaleExprType>;
    
public:
    using typename base_t::x_t;
    using typename base_t::loc_t;
    using typename base_t::scale_t;
    using typename base_t::value_t;
    using typename base_t::var_t;
    using base_t::x_;
    using base_t::loc_;
    using base_t::scale_;

    CauchyAdjLogPDFNode(const x_t& x,
                        const loc_t& loc,
                        const scale_t& scale)
        : base_t(x, loc, scale)
    {}

    const var_t& feval()
    {
        auto&& x = x_.feval().array();
        auto&& x0 = loc_.feval().array();
        auto&& gamma = scale_.feval();

        if (!within_range()) {
            return this->get() = util::neg_inf<value_t>;
        }

        auto diff = x - x0;
        return this->get() = -(gamma + (1./gamma) * diff.square()).log().sum();
    }

    void beval(value_t seed)
    {
        if (seed == 0 || !within_range()) return;

        auto&& x = x_.get().array();
        auto&& x0 = loc_.get().array();
        auto&& gamma = scale_.get();
        auto gamma_sq = gamma * gamma;

        auto diff = (x - x0);
        auto dx = (-2. * seed) * diff / (gamma_sq + diff.square());
        auto dx0 = (-seed) * dx;
        value_t dgamma = (-seed/gamma) * ((dx * diff).sum() + x.size());

        scale_.beval(dgamma);
        loc_.beval(dx0);
        x_.beval(dx);
    }

private:
    bool within_range() const {
        return scale_.get() > 0;
    }
};

// Case 5: vvv
template <class XExprType
        , class LocExprType
        , class ScaleExprType>
struct CauchyAdjLogPDFNode<XExprType,
                            LocExprType,
                            ScaleExprType,
                            std::tuple<vec, vec, vec> >:
    details::CauchyBase<XExprType, LocExprType, ScaleExprType>,
    core::ExprBase<CauchyAdjLogPDFNode<XExprType, LocExprType, ScaleExprType>>
{
private:
    using base_t = details::CauchyBase<
        XExprType, LocExprType, ScaleExprType>;
    
public:
    using typename base_t::x_t;
    using typename base_t::loc_t;
    using typename base_t::scale_t;
    using typename base_t::value_t;
    using typename base_t::var_t;
    using base_t::x_;
    using base_t::loc_;
    using base_t::scale_;

    CauchyAdjLogPDFNode(const x_t& x,
                        const loc_t& loc,
                        const scale_t& scale)
        : base_t(x, loc, scale)
    {}

    const var_t& feval()
    {
        auto&& x = x_.feval().array();
        auto&& x0 = loc_.feval().array();
        auto&& gamma = scale_.feval().array();

        if (!within_range()) {
            return this->get() = util::neg_inf<value_t>;
        }

        auto diff = x - x0;
        return this->get() = -(gamma + (diff.square()/gamma)).log().sum();
    }

    void beval(value_t seed)
    {
        if (seed == 0 || !within_range()) return;

        auto&& x = x_.get().array();
        auto&& x0 = loc_.get().array();
        auto&& gamma = scale_.get().array();

        auto diff = (x - x0);
        auto dx = (-2. * seed) * diff / (gamma.square() + diff.square());
        auto dx0 = (-seed) * dx;
        auto dgamma = (-seed/gamma) * ((dx * diff) + 1);

        scale_.beval(dgamma);
        loc_.beval(dx0);
        x_.beval(dx);
    }

private:
    bool within_range() const {
        return (scale_.get().array() > 0).all();
    }
};

} // namespace stat

template <class XType
        , class LocType
        , class ScaleType
        , class = std::enable_if_t<
            util::is_convertible_to_ad_v<XType> &&
            util::is_convertible_to_ad_v<LocType> &&
            util::is_convertible_to_ad_v<ScaleType> &&
            util::any_ad_v<XType, LocType, ScaleType> > >
inline auto cauchy_adj_log_pdf(const XType& x,
                               const LocType& loc,
                               const ScaleType& scale)
{
    using x_expr_t = util::convert_to_ad_t<XType>;
    using loc_expr_t = util::convert_to_ad_t<LocType>;
    using scale_expr_t = util::convert_to_ad_t<ScaleType>;
    x_expr_t x_expr = x;
    loc_expr_t loc_expr = loc;
    scale_expr_t scale_expr = scale;
    return stat::CauchyAdjLogPDFNode<
        x_expr_t, loc_expr_t, scale_expr_t>(x_expr, loc_expr, scale_expr);
}

} // namespace ad
