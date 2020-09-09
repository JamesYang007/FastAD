#pragma once
#include <tuple>
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_adj_view.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/util/numeric.hpp>

namespace ad {
namespace stat {
namespace details {

template <class XExprType
        , class MinExprType
        , class MaxExprType>
struct UniformBase:
    core::ValueAdjView<util::common_value_t<
                        XExprType, 
                        MinExprType, 
                        MaxExprType>, ad::scl>
{
    using x_t = XExprType;
    using min_t = MinExprType;
    using max_t = MaxExprType;
    using common_value_t = util::common_value_t<
        x_t, min_t, max_t>;
    using value_adj_view_t = core::ValueAdjView<common_value_t, ad::scl>;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    UniformBase(const x_t& x,
                const min_t& min,
                const max_t& max)
        : value_adj_view_t(nullptr, nullptr, 1, 1)
        , x_{x}
        , min_{min}
        , max_{max}
    {}

    ptr_pack_t bind_cache(ptr_pack_t begin)
    {
        begin = x_.bind_cache(begin);
        begin = min_.bind_cache(begin);
        begin = max_.bind_cache(begin);
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
                min_.bind_cache_size() +
                max_.bind_cache_size();
    }

    util::SizePack single_bind_cache_size() const
    {
        return {this->size(), 0};
    }

protected:
    x_t x_;
    min_t min_;
    max_t max_;
};

} // namespace details

/**
 * UniformAdjLogPDFNode represents the uniform log pdf 
 * adjusted to omit all fixed constants (omits nothing in this case).
 *
 * It assumes the value type that is common to all three expressions.
 * Since it represents a log-pdf, it is always a scalar expression.
 *
 * The only possible shape combinations are as follows:
 * x -> scalar, mean -> scalar, sigma -> scalar
 * x -> vec, mean -> scalar | vector, sigma -> scalar | vector
 *
 * No other shapes are permitted for this node.
 *
 * @tparam  XExprType           type of x expression at which to evaluate log-pdf
 * @tparam  MinExprType         type of min expression
 * @tparam  MaxExprType         type of max expression
 */
template <class XExprType
        , class MinExprType
        , class MaxExprType
        , class = std::tuple<
            typename util::shape_traits<XExprType>::shape_t,
            typename util::shape_traits<MinExprType>::shape_t,
            typename util::shape_traits<MaxExprType>::shape_t> >
struct UniformAdjLogPDFNode;

// Case 1: sss
template <class XExprType
        , class MinExprType
        , class MaxExprType>
struct UniformAdjLogPDFNode<XExprType,
                            MinExprType,
                            MaxExprType,
                            std::tuple<scl, scl, scl> >:
    details::UniformBase<XExprType, MinExprType, MaxExprType>,
    core::ExprBase<UniformAdjLogPDFNode<XExprType, MinExprType, MaxExprType>>
{
private:
    using base_t = details::UniformBase<
        XExprType, MinExprType, MaxExprType>;
    
public:
    using typename base_t::x_t;
    using typename base_t::min_t;
    using typename base_t::max_t;
    using typename base_t::value_t;
    using typename base_t::var_t;
    using base_t::x_;
    using base_t::min_;
    using base_t::max_;

    UniformAdjLogPDFNode(const x_t& x,
                         const min_t& min,
                         const max_t& max)
        : base_t(x, min, max)
        , log_diff_{0}
    {
        if constexpr (util::is_constant_v<min_t> &&
                      util::is_constant_v<max_t>) {
            this->update_cache();
        }
    }

    const var_t& feval()
    {
        x_.feval();
        min_.feval();
        max_.feval();

        if constexpr (!util::is_constant_v<min_t> ||
                      !util::is_constant_v<max_t>) {
            this->update_cache();
        }

        // if out of range
        if (!within_range()) {
            return this->get() = util::neg_inf<value_t>;
        }

        return this->get() = -log_diff_;
    }

    void beval(value_t seed)
    {
        if (seed == 0 || !within_range()) return;
        value_t adj = seed / (max_.get() - min_.get());
        max_.beval(-adj);
        min_.beval(adj);
    }

private:
    void update_cache() {
        log_diff_ = std::log(max_.get() - min_.get());
    }

    bool within_range() const {
        return min_.get() < x_.get() && 
                x_.get() < max_.get();
    }

    value_t log_diff_;
};

// Case 2: vss
template <class XExprType
        , class MinExprType
        , class MaxExprType>
struct UniformAdjLogPDFNode<XExprType,
                            MinExprType,
                            MaxExprType,
                            std::tuple<vec, scl, scl> >:
    details::UniformBase<XExprType, MinExprType, MaxExprType>,
    core::ExprBase<UniformAdjLogPDFNode<XExprType, MinExprType, MaxExprType>>
{
private:
    using base_t = details::UniformBase<
        XExprType, MinExprType, MaxExprType>;
    
public:
    using typename base_t::x_t;
    using typename base_t::min_t;
    using typename base_t::max_t;
    using typename base_t::value_t;
    using typename base_t::var_t;
    using base_t::x_;
    using base_t::min_;
    using base_t::max_;

    UniformAdjLogPDFNode(const x_t& x,
                         const min_t& min,
                         const max_t& max)
        : base_t(x, min, max)
        , log_diff_{0}
        , x_min_{0}
        , x_max_{0}
    {
        if constexpr (util::is_constant_v<min_t> &&
                      util::is_constant_v<max_t>) {
            update_log_diff_cache();
        }
        if constexpr (util::is_constant_v<x_t>) {
            update_x_cache();
        }
    }

    const var_t& feval()
    {
        x_.feval();
        min_.feval();
        max_.feval();

        if constexpr (!util::is_constant_v<min_t> ||
                      !util::is_constant_v<max_t>) {
            update_log_diff_cache();
        }

        if constexpr (!util::is_constant_v<x_t>) {
            update_x_cache();
        }

        if (!within_range()) {
            return this->get() = util::neg_inf<value_t>;
        }

        value_t n = x_.size();
        return this->get() = -n * log_diff_;
    }

    void beval(value_t seed)
    {
        if (seed == 0 || !within_range()) return;
        value_t adj = seed * static_cast<value_t>(x_.size()) /
            (max_.get() - min_.get());
        max_.beval(-adj);
        min_.beval(adj);
    }

private:
    void update_log_diff_cache() {
        log_diff_ = std::log(max_.get() - min_.get());
    }

    void update_x_cache() {
        x_min_ = x_.get().minCoeff();
        x_max_ = x_.get().maxCoeff();
    }

    bool within_range() const {
        return min_.get() < x_min_ && 
                x_max_ < max_.get();
    }

    value_t log_diff_;
    value_t x_min_;
    value_t x_max_;
};

// Case 3: vsv
template <class XExprType
        , class MinExprType
        , class MaxExprType>
struct UniformAdjLogPDFNode<XExprType,
                            MinExprType,
                            MaxExprType,
                            std::tuple<vec, scl, vec> >:
    details::UniformBase<XExprType, MinExprType, MaxExprType>,
    core::ExprBase<UniformAdjLogPDFNode<XExprType, MinExprType, MaxExprType>>
{
private:
    using base_t = details::UniformBase<
        XExprType, MinExprType, MaxExprType>;
    
public:
    using typename base_t::x_t;
    using typename base_t::min_t;
    using typename base_t::max_t;
    using typename base_t::value_t;
    using typename base_t::var_t;
    using base_t::x_;
    using base_t::min_;
    using base_t::max_;

    UniformAdjLogPDFNode(const x_t& x,
                         const min_t& min,
                         const max_t& max)
        : base_t(x, min, max)
        , log_diff_{0}
        , x_min_{0}
        , x_bounded_above_{false}
    {
        if constexpr (util::is_constant_v<min_t> &&
                      util::is_constant_v<max_t>) {
            update_log_diff_cache();
        }
        if constexpr (util::is_constant_v<x_t>) {
            update_x_cache();
        }
    }

    const var_t& feval()
    {
        x_.feval();
        min_.feval();
        max_.feval();

        if constexpr (!util::is_constant_v<min_t> ||
                      !util::is_constant_v<max_t>) {
            update_log_diff_cache();
        }

        if constexpr (!util::is_constant_v<x_t>) {
            update_x_cache();
        }

        if (!within_range()) {
            return this->get() = util::neg_inf<value_t>;
        }

        return this->get() = -log_diff_;
    }

    void beval(value_t seed)
    {
        if (seed == 0 || !within_range()) return;

        auto&& min = min_.get();
        auto&& max = max_.get().array();
        max_.beval((-seed) / (max - min));
        min_.beval(seed * (1. / (max - min)).sum());
    }

private:
    void update_log_diff_cache() {
        log_diff_ = (max_.get().array() - min_.get()).log().sum();
    }

    void update_x_cache() {
        x_min_ = x_.get().minCoeff();
        x_bounded_above_ = (x_.get().array() < max_.get().array()).all();
    }

    bool within_range() const {
        return min_.get() < x_min_ && x_bounded_above_;
    }

    value_t log_diff_;
    value_t x_min_;
    bool x_bounded_above_;
};

// Case 4: vvs
template <class XExprType
        , class MinExprType
        , class MaxExprType>
struct UniformAdjLogPDFNode<XExprType,
                            MinExprType,
                            MaxExprType,
                            std::tuple<vec, vec, scl> >:
    details::UniformBase<XExprType, MinExprType, MaxExprType>,
    core::ExprBase<UniformAdjLogPDFNode<XExprType, MinExprType, MaxExprType>>
{
private:
    using base_t = details::UniformBase<
        XExprType, MinExprType, MaxExprType>;
    
public:
    using typename base_t::x_t;
    using typename base_t::min_t;
    using typename base_t::max_t;
    using typename base_t::value_t;
    using typename base_t::var_t;
    using base_t::x_;
    using base_t::min_;
    using base_t::max_;

    UniformAdjLogPDFNode(const x_t& x,
                         const min_t& min,
                         const max_t& max)
        : base_t(x, min, max)
        , log_diff_{0}
        , x_max_{0}
        , x_bounded_below_{false}
    {
        if constexpr (util::is_constant_v<min_t> &&
                      util::is_constant_v<max_t>) {
            update_log_diff_cache();
        }
        if constexpr (util::is_constant_v<x_t>) {
            update_x_cache();
        }
    }

    const var_t& feval()
    {
        x_.feval();
        min_.feval();
        max_.feval();

        if constexpr (!util::is_constant_v<min_t> ||
                      !util::is_constant_v<max_t>) {
            update_log_diff_cache();
        }

        if constexpr (!util::is_constant_v<x_t>) {
            update_x_cache();
        }

        if (!within_range()) {
            return this->get() = util::neg_inf<value_t>;
        }

        return this->get() = -log_diff_;
    }

    void beval(value_t seed)
    {
        if (seed == 0 || !within_range()) return;

        auto&& min = min_.get().array();
        auto&& max = max_.get();
        max_.beval((-seed) * (1. / (max - min)).sum());
        min_.beval(seed / (max - min));
    }

private:
    void update_log_diff_cache() {
        log_diff_ = (max_.get() - min_.get().array()).log().sum();
    }

    void update_x_cache() {
        x_max_ = x_.get().maxCoeff();
        x_bounded_below_ = (x_.get().array() > min_.get().array()).all();
    }

    bool within_range() const {
        return x_bounded_below_ && x_max_ < max_.get();
    }

    value_t log_diff_;
    value_t x_max_;
    bool x_bounded_below_;
};

// Case 5: vvv
template <class XExprType
        , class MinExprType
        , class MaxExprType>
struct UniformAdjLogPDFNode<XExprType,
                            MinExprType,
                            MaxExprType,
                            std::tuple<vec, vec, vec> >:
    details::UniformBase<XExprType, MinExprType, MaxExprType>,
    core::ExprBase<UniformAdjLogPDFNode<XExprType, MinExprType, MaxExprType>>
{
private:
    using base_t = details::UniformBase<
        XExprType, MinExprType, MaxExprType>;
    
public:
    using typename base_t::x_t;
    using typename base_t::min_t;
    using typename base_t::max_t;
    using typename base_t::value_t;
    using typename base_t::var_t;
    using base_t::x_;
    using base_t::min_;
    using base_t::max_;

    UniformAdjLogPDFNode(const x_t& x,
                         const min_t& min,
                         const max_t& max)
        : base_t(x, min, max)
        , log_diff_{0}
        , x_bounded_below_{false}
        , x_bounded_above_{false}
    {
        if constexpr (util::is_constant_v<min_t> &&
                      util::is_constant_v<max_t>) {
            update_log_diff_cache();
        }
    }

    const var_t& feval()
    {
        x_.feval();
        min_.feval();
        max_.feval();

        if constexpr (!util::is_constant_v<min_t> ||
                      !util::is_constant_v<max_t>) {
            update_log_diff_cache();
        }

        x_bounded_below_ = (x_.get().array() > min_.get().array()).all();
        x_bounded_above_ = (x_.get().array() < max_.get().array()).all();

        if (!within_range()) {
            return this->get() = util::neg_inf<value_t>;
        }

        return this->get() = -log_diff_;
    }

    void beval(value_t seed)
    {
        if (seed == 0 || !within_range()) return;
        auto&& min = min_.get().array();
        auto&& max = max_.get().array();
        max_.beval((-seed) / (max - min));
        min_.beval(seed / (max - min));
    }

private:
    void update_log_diff_cache() {
        log_diff_ = (max_.get().array() - min_.get().array()).log().sum();
    }

    bool within_range() const {
        return x_bounded_below_ && x_bounded_above_;
    }

    value_t log_diff_;
    bool x_bounded_below_;
    bool x_bounded_above_;
};

} // namespace stat

template <class XType
        , class MinType
        , class MaxType
        , class = std::enable_if_t<
            util::is_convertible_to_ad_v<XType> &&
            util::is_convertible_to_ad_v<MinType> &&
            util::is_convertible_to_ad_v<MaxType> &&
            util::any_ad_v<XType, MinType, MaxType> > >
inline auto uniform_adj_log_pdf(const XType& x,
                                const MinType& min,
                                const MaxType& max)
{
    using x_expr_t = util::convert_to_ad_t<XType>;
    using min_expr_t = util::convert_to_ad_t<MinType>;
    using max_expr_t = util::convert_to_ad_t<MaxType>;
    x_expr_t x_expr = x;
    min_expr_t min_expr = min;
    max_expr_t max_expr = max;
    return stat::UniformAdjLogPDFNode<
        x_expr_t, min_expr_t, max_expr_t>(x_expr, min_expr, max_expr);
}

} // namespace ad
