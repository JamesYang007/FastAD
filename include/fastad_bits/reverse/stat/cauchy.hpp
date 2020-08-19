#pragma once
#include <tuple>
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_view.hpp>
#include <fastad_bits/reverse/core/var_view.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/util/numeric.hpp>

namespace ad {
namespace stat {
namespace details {

template <class XExprType
        , class LocExprType
        , class ScaleExprType>
struct CauchyBase:
    core::ValueView<util::common_value_t<XExprType, 
                                   LocExprType, 
                                   ScaleExprType>, ad::scl>
{
    using x_t = XExprType;
    using loc_t = LocExprType;
    using scale_t = ScaleExprType;
    using common_value_t = util::common_value_t<
        x_t, loc_t, scale_t>;
    using value_view_t = core::ValueView<common_value_t, ad::scl>;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::bind;

    CauchyBase(const x_t& x,
               const loc_t& loc,
               const scale_t& scale)
        : value_view_t(nullptr, 1, 1)
        , x_{x}
        , loc_{loc}
        , scale_{scale}
    {}

    value_t* bind(value_t* begin) 
    {
        value_t* next = begin;
        if constexpr (!util::is_var_view_v<x_t>) {
            next = x_.bind(next);
        }
        if constexpr (!util::is_var_view_v<loc_t>) {
            next = loc_.bind(next);
        }
        if constexpr (!util::is_var_view_v<scale_t>) {
            next = scale_.bind(next);
        }
        return value_view_t::bind(next);
    }

    size_t bind_size() const 
    { 
        return single_bind_size() + 
                x_.bind_size() +
                loc_.bind_size() +
                scale_.bind_size();
    }

    constexpr size_t single_bind_size() const { return this->size(); }

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
    using base_t::bind;
    using base_t::bind_size;
    using base_t::single_bind_size;

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

    void beval(value_t seed, size_t, size_t, util::beval_policy pol)
    {
        if (seed == 0 || !within_range()) return;

        auto&& x = x_.get();
        auto&& x0 = loc_.get();
        auto&& gamma = scale_.get();

        auto diff = x-x0;
        auto x0_adj = 2. * diff / (gamma * inner_term_);
        auto x_adj = -x0_adj;
        auto gamma_adj = 1./gamma * (x0_adj * diff - 1);

        scale_.beval(seed * gamma_adj, 0, 0, pol);
        loc_.beval(seed * x0_adj, 0, 0, pol);
        x_.beval(seed * x_adj, 0, 0, pol);
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
    using base_t::bind;
    using base_t::bind_size;
    using base_t::single_bind_size;

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

        auto diff = x.array() - x0;
        return this->get() = -(gamma + (1./gamma) * diff * diff).log().sum();
    }

    void beval(value_t seed, size_t, size_t, util::beval_policy pol)
    {
        if (seed == 0 || !within_range()) return;

        auto&& x = x_.get();
        auto&& x0 = loc_.get();
        auto&& gamma = scale_.get();
        auto gamma_sq = gamma * gamma;

        auto diff = (x.array() - x0);
        auto dx = -2. * diff / (gamma_sq + diff * diff);
        value_t dx0 = -dx.sum();
        value_t dgamma = (-1./gamma) * ((dx * diff).sum() + x.size());
        
        scale_.beval(seed * dgamma, 0, 0, pol);
        loc_.beval(seed * dx0, 0, 0, pol);

        for (int i = 0; i < x.rows(); ++i) {
            x_.beval(seed * dx(i), i, 0, pol);
        }
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
    using base_t::bind;
    using base_t::bind_size;
    using base_t::single_bind_size;

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
        return this->get() = -(gamma + (1./gamma) * diff * diff).log().sum();
    }

    void beval(value_t seed, size_t, size_t, util::beval_policy pol)
    {
        if (seed == 0 || !within_range()) return;

        auto&& x = x_.get().array();
        auto&& x0 = loc_.get();
        auto&& gamma = scale_.get().array();

        auto diff = x - x0;
        auto dx = -2. * diff / (gamma * gamma + diff * diff);
        value_t dx0 = -dx.sum();
        auto dgamma = (-1./gamma) * (dx * diff + 1);

        for (size_t i = 0; i < scale_.rows(); ++i) {
            scale_.beval(seed * dgamma(i), i, 0, pol);
        }

        loc_.beval(seed * dx0, 0, 0, pol);

        for (size_t i = 0; i < x_.rows(); ++i) {
            x_.beval(seed * dx(i), i, 0, pol);
        }
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
    using base_t::bind;
    using base_t::bind_size;
    using base_t::single_bind_size;

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
        return this->get() = -(gamma + (1./gamma) * diff * diff).log().sum();
    }

    void beval(value_t seed, size_t, size_t, util::beval_policy pol)
    {
        if (seed == 0 || !within_range()) return;

        auto&& x = x_.get().array();
        auto&& x0 = loc_.get().array();
        auto&& gamma = scale_.get();
        auto gamma_sq = gamma * gamma;

        auto diff = (x - x0);
        auto dx = -2. * diff / (gamma_sq + diff * diff);
        auto dx0 = -dx;
        value_t dgamma = (-1./gamma) * ((dx * diff).sum() + x.size());

        scale_.beval(seed * dgamma, 0, 0, pol);

        for (int i = 0; i < x0.rows(); ++i) {
            loc_.beval(seed * dx0(i), i, 0, pol);
        }

        for (int i = 0; i < x.rows(); ++i) {
            x_.beval(seed * dx(i), i, 0, pol);
        }
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
    using base_t::bind;
    using base_t::bind_size;
    using base_t::single_bind_size;

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
        return this->get() = -(gamma + (1./gamma) * diff * diff).log().sum();
    }

    void beval(value_t seed, size_t, size_t, util::beval_policy pol)
    {
        if (seed == 0 || !within_range()) return;

        auto&& x = x_.get().array();
        auto&& x0 = loc_.get().array();
        auto&& gamma = scale_.get().array();

        auto diff = (x - x0);
        auto dx = -2. * diff / (gamma * gamma + diff * diff);
        auto dx0 = -dx;
        auto dgamma = (-1./gamma) * ((dx * diff) + 1);

        for (int i = 0; i < gamma.rows(); ++i) {
            scale_.beval(seed * dgamma(i), i, 0, pol);
        }

        for (int i = 0; i < x0.rows(); ++i) {
            loc_.beval(seed * dx0(i), i, 0, pol);
        }

        for (int i = 0; i < x.rows(); ++i) {
            x_.beval(seed * dx(i), i, 0, pol);
        }
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
