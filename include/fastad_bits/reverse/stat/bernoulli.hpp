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
        , class PExprType>
struct BernoulliBase:
    core::ValueAdjView<util::common_value_t<
                        XExprType, PExprType>, ad::scl>
{
    using x_t = XExprType;
    using p_t = PExprType;
    using p_value_t = typename util::expr_traits<p_t>::value_t;
    using value_adj_view_t = core::ValueAdjView<p_value_t, ad::scl>;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    BernoulliBase(const x_t& x,
                  const p_t& p)
        : value_adj_view_t(nullptr, nullptr, 1, 1)
        , x_{x}
        , p_{p}
    {}

    ptr_pack_t bind_cache(ptr_pack_t begin)
    {
        begin = x_.bind_cache(begin);
        begin = p_.bind_cache(begin);
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
                p_.bind_cache_size();
    }

    util::SizePack single_bind_cache_size() const
    {
        return {this->size(), 0}; 
    }

protected:
    x_t x_;
    p_t p_;
};

} // namespace details

/**
 * BernoulliAdjLogPDFNode represents the bernoulli log pdf (pmf)
 * adjusted to omit all fixed constants (omits nothing in this case).
 *
 * It assumes the value type of p.
 * Since it represents a log-pdf, it is always a scalar expression.
 *
 * The only possible shape combinations are as follows:
 * x -> scalar, p -> scalar
 * x -> vec, p -> scalar | vector
 *
 * No other shapes are permitted for this node.
 *
 * At construction, the actual sizes of the three expressions are checked -
 * specifically if x and p are vectors,
 * then size of x must be the same as that of p.
 *
 * @tparam  XExprType           type of x expression at which to evaluate log-pdf
 * @tparam  MinExprType         type of min expression
 * @tparam  PExprType         type of max expression
 */
template <class XExprType
        , class PExprType
        , class = std::tuple<
            typename util::shape_traits<XExprType>::shape_t,
            typename util::shape_traits<PExprType>::shape_t> >
struct BernoulliAdjLogPDFNode;

// Case 1: ss
template <class XExprType
        , class PExprType>
struct BernoulliAdjLogPDFNode<XExprType,
                              PExprType,
                              std::tuple<scl, scl> >:
    details::BernoulliBase<XExprType, PExprType>,
    core::ExprBase<BernoulliAdjLogPDFNode<XExprType, PExprType>>
{
private:
    using base_t = details::BernoulliBase<
        XExprType, PExprType>;
    
public:
    using typename base_t::x_t;
    using typename base_t::p_t;
    using typename base_t::value_t;
    using typename base_t::var_t;
    using base_t::x_;
    using base_t::p_;

    BernoulliAdjLogPDFNode(const x_t& x,
                           const p_t& p)
        : base_t(x, p)
        , log_p_{0}
        , log_p_dual_{0}
    {
        if constexpr (util::is_constant_v<p_t>) {
            this->update_cache();
        }
    }

    const var_t& feval()
    {
        x_.feval();
        p_.feval();

        if constexpr (!util::is_constant_v<p_t>) {
            this->update_cache();
        }

        // if out of range, clip p to [0,1]
        if (!within_range()) {
            if (p_.get() <= 0) {
                return this->get() = (x_.get() == 0) ? 0 : util::neg_inf<value_t>;
            } else {
                return this->get() = (x_.get() == 1) ? 0 : util::neg_inf<value_t>;
            }
        }

        if (x_.get() == 0) {
            return this->get() = log_p_dual_;
        } else if (x_.get() == 1) {
            return this->get() = log_p_;
        } else {
            return this->get() = util::neg_inf<value_t>;
        }
    }

    void beval(value_t seed)
    {
        if (seed == 0 || !within_range() || (x_.get() != 0 && x_.get() != 1)) return;

        auto adj = (x_.get() == 0) ? -seed / (1-p_.get()) : seed / p_.get();
        p_.beval(adj);
    }

private:
    void update_cache() {
        if (within_range()) {
            log_p_ = std::log(p_.get());
            log_p_dual_ = std::log(1-p_.get());
        }
    }

    bool within_range() const {
        return 0 < p_.get() && 
                p_.get() < 1;
    }

    value_t log_p_;
    value_t log_p_dual_;
};

// Case 2: vs
template <class XExprType
        , class PExprType>
struct BernoulliAdjLogPDFNode<XExprType,
                              PExprType,
                              std::tuple<vec, scl> >:
    details::BernoulliBase<XExprType, PExprType>,
    core::ExprBase<BernoulliAdjLogPDFNode<XExprType, PExprType>>
{
private:
    using base_t = details::BernoulliBase<
        XExprType, PExprType>;
    
public:
    using typename base_t::x_t;
    using typename base_t::p_t;
    using typename base_t::value_t;
    using typename base_t::var_t;
    using base_t::x_;
    using base_t::p_;

    BernoulliAdjLogPDFNode(const x_t& x,
                           const p_t& p)
        : base_t(x, p)
        , log_p_{0}
        , log_p_dual_{0}
        , is_x_zero_one_{false}
        , x_sum_{0}
    {
        if constexpr (util::is_constant_v<p_t>) {
            this->update_cache();
        }
        if constexpr (util::is_constant_v<x_t>) {
            this->update_x_cache();
        }
    }

    const var_t& feval()
    {
        x_.feval();
        p_.feval();

        if constexpr (!util::is_constant_v<p_t>) {
            this->update_cache();
        }
        if constexpr (!util::is_constant_v<x_t>) {
            this->update_x_cache();
        }

        // if out of range, clip p to [0,1]
        if (!within_range()) {
            if (p_.get() <= 0) {
                return this->get() = (x_sum_ == 0 && is_x_zero_one_) ? 
                    0 : util::neg_inf<value_t>;
            } else {
                return this->get() = (x_sum_ == x_.size() && is_x_zero_one_) ? 
                    0 : util::neg_inf<value_t>;
            }
        }

        if (is_x_zero_one_) {
            return this->get() = x_sum_ * log_p_ + 
                (x_.size() - x_sum_) * log_p_dual_;
        } else {
            return this->get() = util::neg_inf<value_t>;
        }
    }

    void beval(value_t seed)
    {
        if (seed == 0 || !within_range() || !is_x_zero_one_) return;

        value_t adj = (x_sum_ - (x_.size() * p_.get())) /
                        (p_.get() * (1-p_.get()));
        p_.beval(seed * adj);
    }

private:
    void update_cache() {
        if (within_range()) {
            log_p_ = std::log(p_.get());
            log_p_dual_ = std::log(1-p_.get());
        }
    }

    void update_x_cache() {
        is_x_zero_one_ = (x_.get().array() == 0).max(
                         (x_.get().array() == 1)).all();
        x_sum_ = x_.get().array().sum();
    }

    bool within_range() const {
        return 0 < p_.get() && 
                p_.get() < 1;
    }

    value_t log_p_;
    value_t log_p_dual_;
    bool is_x_zero_one_;
    value_t x_sum_;
};

// Case 3: vv
template <class XExprType
        , class PExprType>
struct BernoulliAdjLogPDFNode<XExprType,
                              PExprType,
                              std::tuple<vec, vec> >:
    details::BernoulliBase<XExprType, PExprType>,
    core::ExprBase<BernoulliAdjLogPDFNode<XExprType, PExprType>>
{
private:
    using base_t = details::BernoulliBase<
        XExprType, PExprType>;
    
public:
    using typename base_t::x_t;
    using typename base_t::p_t;
    using typename base_t::value_t;
    using typename base_t::var_t;
    using base_t::x_;
    using base_t::p_;

    BernoulliAdjLogPDFNode(const x_t& x,
                           const p_t& p)
        : base_t(x, p)
        , is_p_within_range_{false}
        , is_x_zero_one_{false}
    {
        if constexpr (util::is_constant_v<p_t>) {
            this->update_cache();
        }
        if constexpr (util::is_constant_v<x_t>) {
            this->update_x_cache();
        }
    }

    const var_t& feval()
    {
        auto&& x = x_.feval().array();
        auto&& p = p_.feval().array();

        if constexpr (!util::is_constant_v<p_t>) {
            this->update_cache();
        }
        if constexpr (!util::is_constant_v<x_t>) {
            this->update_x_cache();
        }

        if (is_x_zero_one_) {
            if (!is_p_within_range_) {
                this->get() = 0;
                for (size_t i = 0; i < p_.size(); ++i) {
                    if (p(i) <= 0 && x(i) != 0) {
                        return this->get() = util::neg_inf<value_t>;
                    } else if (p(i) >= 1 && x(i) != 1){
                        return this->get() = util::neg_inf<value_t>;
                    } else if (0 < p(i) && p(i) < 1){
                        this->get() += (x(i) == 1) ? 
                            std::log(p(i)) : std::log(1-p(i));
                    }
                }
                return this->get();
            }
            return this->get() = (x.template cast<value_t>()*p + 
                                  (1-x.template cast<value_t>())*(1-p)).log().sum();
        } else {
            return this->get() = util::neg_inf<value_t>;
        }
    }

    void beval(value_t seed)
    {
        if (seed == 0 || !is_x_zero_one_) return;

        auto&& x = x_.get().array();
        auto&& p = p_.get().array();

        using vec_t = std::decay_t<decltype(p)>;
        auto adj = vec_t::NullaryExpr(x.size(),
                [&](size_t i) {
                    return (0. < p(i) && p(i) < 1.) ?
                               ( (x(i) == 1) ? seed / p(i) : (-seed) / (1. - p(i)) ) :
                               0.;
                });
        p_.beval(adj);
    }

private:
    void update_cache() {
        is_p_within_range_ = (p_.get().array() > 0).min(
                             (p_.get().array() < 1)).all();
    }

    void update_x_cache() {
        is_x_zero_one_ = (x_.get().array() == 0).max(
                         (x_.get().array() == 1)).all();
    }

    bool is_p_within_range_;
    bool is_x_zero_one_;
};

} // namespace stat

template <class XType
        , class PType
        , class = std::enable_if_t<
            util::is_convertible_to_ad_v<XType> &&
            util::is_convertible_to_ad_v<PType> &&
            util::any_ad_v<XType, PType> > >
inline auto bernoulli_adj_log_pdf(const XType& x,
                                  const PType& p)
{
    using x_expr_t = util::convert_to_ad_t<XType>;
    using p_expr_t = util::convert_to_ad_t<PType>;
    x_expr_t x_expr = x;
    p_expr_t p_expr = p;
    return stat::BernoulliAdjLogPDFNode<
        x_expr_t, p_expr_t>(x_expr, p_expr);
}

} // namespace ad


