#pragma once
#include <tuple>
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_view.hpp>
#include <fastad_bits/reverse/core/var_view.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/util/numeric.hpp>
#include <Eigen/Dense>

namespace ad {
namespace stat {
namespace details {

template <class XExprType
        , class MeanExprType
        , class SigmaExprType>
struct NormalBase:
    core::ValueView<util::common_value_t<XExprType, 
                                         MeanExprType, 
                                         SigmaExprType>, ad::scl>
{
    using x_t = XExprType;
    using mean_t = MeanExprType;
    using sigma_t = SigmaExprType;
    using common_value_t = util::common_value_t<
        x_t, mean_t, sigma_t>;
    using value_view_t = core::ValueView<common_value_t, ad::scl>;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::bind;

    NormalBase(const x_t& x,
               const mean_t& mean,
               const sigma_t& sigma)
        : value_view_t(nullptr, 1, 1)
        , x_{x}
        , mean_{mean}
        , sigma_{sigma}
    {}

    value_t* bind(value_t* begin) 
    {
        value_t* next = begin;
        if constexpr (!util::is_var_view_v<x_t>) {
            next = x_.bind(next);
        }
        if constexpr (!util::is_var_view_v<mean_t>) {
            next = mean_.bind(next);
        }
        if constexpr (!util::is_var_view_v<sigma_t>) {
            next = sigma_.bind(next);
        }
        return value_view_t::bind(next);
    }

    size_t bind_size() const 
    { 
        return single_bind_size() + 
                x_.bind_size() +
                mean_.bind_size() +
                sigma_.bind_size();
    }

    constexpr size_t single_bind_size() const { return this->size(); }

protected:
    x_t x_;
    mean_t mean_;
    sigma_t sigma_;
};

} // namespace details

/**
 * NormalAdjLogPDFNode represents the normal log pdf 
 * adjusted to omit all fixed constants, i.e. omits -n/2*log(2*pi).
 *
 * It assumes the value type that is common to all three expressions.
 * Since it represents a log-pdf, it is always a scalar expression.
 *
 * The only possible shape combinations are as follows:
 * x -> scalar, mean -> scalar, sigma -> scalar
 * x -> vec, mean -> scalar | vector, sigma -> scalar | vector | self adjoint matrix
 *
 * No other shapes are permitted for this node.
 *
 * At construction, the actual sizes of the three expressions are checked -
 * specifically if x is a vector, and mean and sigma are not scalar,
 * then size of x must be the same as that of mean rows and sigma rows.
 * Additionally, we check that sigma is square if it is a matrix.
 *
 * @tparam  XExprType           type of x expression at which to evaluate log-pdf
 * @tparam  MeanExprType        type of mean expression
 * @tparam  SigmaExprType       type of sigma expression
 */
template <class XExprType
        , class MeanExprType
        , class SigmaExprType
        , class = std::tuple<
            typename util::shape_traits<XExprType>::shape_t,
            typename util::shape_traits<MeanExprType>::shape_t,
            typename util::shape_traits<SigmaExprType>::shape_t> >
struct NormalAdjLogPDFNode;

// Case 1: sss
template <class XExprType
        , class MeanExprType
        , class SigmaExprType>
struct NormalAdjLogPDFNode<XExprType, MeanExprType, SigmaExprType,
                           std::tuple<scl, scl, scl> >:
    details::NormalBase<XExprType, MeanExprType, SigmaExprType>,
    core::ExprBase<NormalAdjLogPDFNode<XExprType, MeanExprType, SigmaExprType>>
{
private:
    using base_t = details::NormalBase<
        XExprType, MeanExprType, SigmaExprType>;
    
public:
    using typename base_t::x_t;
    using typename base_t::mean_t;
    using typename base_t::sigma_t;
    using typename base_t::value_t;
    using typename base_t::var_t;
    using base_t::x_;
    using base_t::mean_;
    using base_t::sigma_;
    using base_t::bind;
    using base_t::bind_size;
    using base_t::single_bind_size;

    NormalAdjLogPDFNode(const x_t& x,
                        const mean_t& mean,
                        const sigma_t& sigma)
        : base_t(x, mean, sigma)
        , log_sigma_{0}
    {
        if constexpr (util::is_constant_v<sigma_t>) {
            this->update_cache();
        }
    }

    const var_t& feval()
    {
        auto&& x = x_.feval();
        auto&& m = mean_.feval();
        auto&& s = sigma_.feval();
        if (s <= 0) return this->get() = util::neg_inf<value_t>;

        if constexpr (!util::is_constant_v<sigma_t>) {
            this->update_cache();
        }

        auto z = (x - m) / s;
        
        return this->get() = -0.5 * z * z - log_sigma_; 
    }

    void beval(value_t seed, size_t, size_t, util::beval_policy pol)
    {
        if (seed == 0 || sigma_.get() <= 0) return;

        value_t inv_s = 1./sigma_.get();
        value_t z = (x_.get() - mean_.get()) * inv_s;

        if constexpr (!util::is_constant_v<sigma_t>) {
            sigma_.beval(seed * (z*z - 1) * inv_s, 0, 0, pol);
        }
        value_t adj = z * inv_s;
        mean_.beval(seed * adj, 0, 0, pol);
        x_.beval(seed * (-adj), 0, 0, pol);
    }

private:
    void update_cache() {
        log_sigma_ = std::log(sigma_.get());
    }

    value_t log_sigma_;
};

// Case 2: vss
template <class XExprType
        , class MeanExprType
        , class SigmaExprType>
struct NormalAdjLogPDFNode<XExprType, MeanExprType, SigmaExprType,
                           std::tuple<vec, scl, scl> >:
    details::NormalBase<XExprType, MeanExprType, SigmaExprType>,
    core::ExprBase<NormalAdjLogPDFNode<XExprType, MeanExprType, SigmaExprType>>
{
private:
    using base_t = details::NormalBase<
        XExprType, MeanExprType, SigmaExprType>;
    
public:
    using typename base_t::x_t;
    using typename base_t::mean_t;
    using typename base_t::sigma_t;
    using typename base_t::value_t;
    using typename base_t::var_t;
    using base_t::x_;
    using base_t::mean_;
    using base_t::sigma_;
    using base_t::bind;
    using base_t::bind_size;
    using base_t::single_bind_size;

    NormalAdjLogPDFNode(const x_t& x,
                        const mean_t& mean,
                        const sigma_t& sigma)
        : base_t(x, mean, sigma)
        , log_sigma_{0}
        , z_sq{0}
        , x_mean_{0}
        , x_var_{0}
    {
        if constexpr (util::is_constant_v<sigma_t>) {
            this->update_cache();
        }

        // optimization when x_ is constant
        // reduced exponential form
        if constexpr (util::is_constant_v<x_t>) {
            x_mean_ = x_.get().mean();
            x_var_ = (x_.get().array() - x_mean_).matrix().squaredNorm();
        }
    }

    const var_t& feval()
    {
        auto&& x = x_.feval();
        auto&& m = mean_.feval();
        auto&& s = sigma_.feval();
        if (s <= 0) return this->get() = util::neg_inf<value_t>;

        if constexpr (!util::is_constant_v<sigma_t>) {
            this->update_cache();
        }

        if constexpr (util::is_constant_v<x_t>) {
            value_t centered = (m - x_mean_);
            value_t inv_s_sq = 1./(s * s);
            return this->get() = 
                -0.5 * inv_s_sq * (x_var_ + x_.rows() * centered * centered) 
                        - x_.rows() * log_sigma_;
        } else {
            auto z = (x.array() - m).matrix();
            z_sq = z.squaredNorm() / (s * s);
            return this->get() = -0.5 * z_sq - x_.rows() * log_sigma_; 
        }
    }

    void beval(value_t seed, size_t, size_t, util::beval_policy pol)
    {
        if (seed == 0 || sigma_.get() <= 0) return;

        value_t inv_s = 1./sigma_.get();
        value_t inv_s_sq = inv_s * inv_s;

        auto&& x = x_.get();
        auto&& m = mean_.get();

        // if x is constant, more optimized beval
        if constexpr (util::is_constant_v<x_t>) {

            if constexpr (!util::is_constant_v<sigma_t>) {
                value_t c = (m - x_mean_);
                value_t sigma_adj = ((x_var_ + x_.rows() * c * c) * inv_s_sq - x_.rows()) * inv_s;
                sigma_.beval(seed * sigma_adj, 0, 0, pol);
            }

            value_t mean_adj = x_.rows() * (x_mean_ - m) * inv_s_sq;
            mean_.beval(seed * mean_adj, 0, 0, pol);

        } else {

            if constexpr (!util::is_constant_v<sigma_t>) {
                sigma_.beval(seed * (z_sq - x_.rows()) * inv_s, 0, 0, pol);
            }

            value_t mean_adj = (x.array() - m).sum() * inv_s_sq;
            mean_.beval(seed * mean_adj, 0, 0, pol);

            if constexpr (!util::is_constant_v<x_t>) {
                for (size_t i = 0; i < x_.rows(); ++i) {
                    x_.beval(seed * ((m - x(i)) * inv_s_sq), i, 0, pol);
                }
            }

        }
    }

private:
    void update_cache() {
        log_sigma_ = std::log(sigma_.get());
    }

    value_t log_sigma_;

    // only used when x is not constant
    value_t z_sq;

    // only used when x is constant
    value_t x_mean_;    
    value_t x_var_;
};

// Case 3: vvs
template <class XExprType
        , class MeanExprType
        , class SigmaExprType>
struct NormalAdjLogPDFNode<XExprType, MeanExprType, SigmaExprType,
                           std::tuple<vec, vec, scl> >:
    details::NormalBase<XExprType, MeanExprType, SigmaExprType>,
    core::ExprBase<NormalAdjLogPDFNode<XExprType, MeanExprType, SigmaExprType>>
{
private:
    using base_t = details::NormalBase<
        XExprType, MeanExprType, SigmaExprType>;
    
public:
    using typename base_t::x_t;
    using typename base_t::mean_t;
    using typename base_t::sigma_t;
    using typename base_t::value_t;
    using typename base_t::var_t;
    using base_t::x_;
    using base_t::mean_;
    using base_t::sigma_;
    using base_t::bind;
    using base_t::bind_size;
    using base_t::single_bind_size;

    NormalAdjLogPDFNode(const x_t& x,
                        const mean_t& mean,
                        const sigma_t& sigma)
        : base_t(x, mean, sigma)
        , log_sigma_{0}
        , z_sq{0}
    {
        if constexpr (util::is_constant_v<sigma_t>) {
            this->update_cache();
        }
    }

    const var_t& feval()
    {
        auto&& x = x_.feval();
        auto&& m = mean_.feval();
        auto&& s = sigma_.feval();
        if (s <= 0) return this->get() = util::neg_inf<value_t>;

        if constexpr (!util::is_constant_v<sigma_t>) {
            this->update_cache();
        }

        auto z = (x.array() - m.array()).matrix();
        z_sq = z.squaredNorm() / (s * s);
        
        return this->get() = -0.5 * z_sq - x_.rows() * log_sigma_; 
    }

    void beval(value_t seed, size_t, size_t, util::beval_policy pol)
    {
        if (seed == 0 || sigma_.get() <= 0) return;

        value_t inv_s = 1./sigma_.get();
        value_t inv_s_sq = inv_s * inv_s;

        auto&& x = x_.get();
        auto&& m = mean_.get();

        if constexpr (!util::is_constant_v<sigma_t>) {
            sigma_.beval(seed * (z_sq - x_.rows()) * inv_s, 0, 0, pol);
        }

        for (size_t i = 0; i < x_.rows(); ++i) {
            mean_.beval(seed * (x(i) - m(i)) * inv_s_sq, i, 0, pol);
        }

        for (size_t i = 0; i < x_.rows(); ++i) {
            x_.beval(seed * (-(x(i) - m(i)) * inv_s_sq), i, 0, pol);
        }

    }

private:
    void update_cache() {
        log_sigma_ = std::log(sigma_.get());
    }

    value_t log_sigma_;
    value_t z_sq;
};

// Case 4: vsv
template <class XExprType
        , class MeanExprType
        , class SigmaExprType>
struct NormalAdjLogPDFNode<XExprType, MeanExprType, SigmaExprType,
                           std::tuple<vec, scl, vec> >:
    details::NormalBase<XExprType, MeanExprType, SigmaExprType>,
    core::ExprBase<NormalAdjLogPDFNode<XExprType, MeanExprType, SigmaExprType>>
{
private:
    using base_t = details::NormalBase<
        XExprType, MeanExprType, SigmaExprType>;
    
public:
    using typename base_t::x_t;
    using typename base_t::mean_t;
    using typename base_t::sigma_t;
    using typename base_t::value_t;
    using typename base_t::var_t;
    using base_t::x_;
    using base_t::mean_;
    using base_t::sigma_;
    using base_t::bind;
    using base_t::bind_size;
    using base_t::single_bind_size;

    NormalAdjLogPDFNode(const x_t& x,
                        const mean_t& mean,
                        const sigma_t& sigma)
        : base_t(x, mean, sigma)
        , log_sigma_{0}
        , is_pos_def_{false}
        , sq_term_{0}
        , lin_term_{0}
        , const_term_{0}
    {
        if constexpr (util::is_constant_v<sigma_t>) {
            this->update_cache();

            // if additionally x is constant, more optimized form
            if constexpr (util::is_constant_v<x_t>) {
                auto&& x = x_.get().array();
                auto&& s = sigma_.get().array();
                sq_term_ = (x/s).matrix().squaredNorm(); 
                lin_term_ = (x/(s * s)).sum();
                const_term_ = (1./s).matrix().squaredNorm();
            }
        }
    }

    const var_t& feval()
    {
        auto&& x = x_.feval();
        auto&& m = mean_.feval();
        auto&& s = sigma_.feval();

        if constexpr (!util::is_constant_v<sigma_t>) {
            this->update_cache();
        }

        if (!is_pos_def_) {
            return this->get() = util::neg_inf<value_t>;
        }

        if constexpr (util::is_constant_v<x_t> &&
                      util::is_constant_v<sigma_t>) {
            return this->get() = 
                -0.5 * (sq_term_ - 2 * m * lin_term_ + m * m * const_term_)
                    - log_sigma_;
        } else {
            auto z = ((x.array() - m) / s.array()).matrix();
            return this->get() = -0.5 * z.squaredNorm() - log_sigma_; 
        }
    }

    void beval(value_t seed, size_t, size_t, util::beval_policy pol)
    {
        if (seed == 0 || !is_pos_def_) return;

        auto&& x = x_.get();
        auto&& m = mean_.get();
        auto&& s = sigma_.get();

        if constexpr (util::is_constant_v<x_t> &&
                      util::is_constant_v<sigma_t>) {
            value_t mean_adj = lin_term_ - m * const_term_;
            mean_.beval(seed * mean_adj, 0, 0, pol);
        } else {

            if constexpr (!util::is_constant_v<sigma_t>) {
                for (size_t i = 0; i < sigma_.rows(); ++i) {
                    value_t inv_s = 1./s(i);
                    value_t sigma_adj = (x(i) - m) * inv_s;
                    sigma_adj *= sigma_adj;
                    sigma_adj = (sigma_adj - 1) * inv_s;
                    sigma_.beval(seed * sigma_adj, i, 0, pol);
                }
            }

            value_t mean_adj = ((x.array() - m) / s.array().square()).sum();
            mean_.beval(seed * mean_adj, 0, 0, pol);

            for (size_t i = 0; i < x_.rows(); ++i) {
                value_t inv_s_sq = 1./(s(i) * s(i));
                x_.beval(seed * (-(x(i) - m) * inv_s_sq), i, 0, pol);
            }

        }
    }

private:
    void update_cache() {
        log_sigma_ = sigma_.get().array().log().sum();
        is_pos_def_ = !(sigma_.get().array() <= 0).any();
    }

    value_t log_sigma_;
    size_t is_pos_def_;

    // only used when x and sigma are both constant
    value_t sq_term_;
    value_t lin_term_;
    value_t const_term_;
};

// Case 5: vvv
template <class XExprType
        , class MeanExprType
        , class SigmaExprType>
struct NormalAdjLogPDFNode<XExprType, MeanExprType, SigmaExprType,
                           std::tuple<vec, vec, vec> >:
    details::NormalBase<XExprType, MeanExprType, SigmaExprType>,
    core::ExprBase<NormalAdjLogPDFNode<XExprType, MeanExprType, SigmaExprType>>
{
private:
    using base_t = details::NormalBase<
        XExprType, MeanExprType, SigmaExprType>;
    
public:
    using typename base_t::x_t;
    using typename base_t::mean_t;
    using typename base_t::sigma_t;
    using typename base_t::value_t;
    using typename base_t::var_t;
    using base_t::x_;
    using base_t::mean_;
    using base_t::sigma_;
    using base_t::bind;
    using base_t::bind_size;
    using base_t::single_bind_size;

    NormalAdjLogPDFNode(const x_t& x,
                        const mean_t& mean,
                        const sigma_t& sigma)
        : base_t(x, mean, sigma)
        , log_sigma_{0}
        , is_pos_def_{false}
    {
        if constexpr (util::is_constant_v<sigma_t>) {
            this->update_cache();
        }
    }

    const var_t& feval()
    {
        auto&& x = x_.feval();
        auto&& m = mean_.feval();
        auto&& s = sigma_.feval();

        if constexpr (!util::is_constant_v<sigma_t>) {
            this->update_cache();
        }

        if (!is_pos_def_) {
            return this->get() = util::neg_inf<value_t>;
        }

        auto z = ((x.array() - m.array()) / s.array()).matrix();
        
        return this->get() = -0.5 * z.squaredNorm() - log_sigma_; 
    }

    void beval(value_t seed, size_t, size_t, util::beval_policy pol)
    {
        if (seed == 0 || !is_pos_def_) return;

        auto&& x = x_.get();
        auto&& m = mean_.get();
        auto&& s = sigma_.get();

        if constexpr (!util::is_constant_v<sigma_t>) {
            for (size_t i = 0; i < sigma_.rows(); ++i) {
                value_t inv_s = 1./s(i);
                value_t sigma_adj = (x(i) - m(i)) * inv_s;
                sigma_adj *= sigma_adj;
                sigma_adj = (sigma_adj - 1) * inv_s;
                sigma_.beval(seed * sigma_adj, i, 0, pol);
            }
        }

        for (size_t i = 0; i < mean_.rows(); ++i) {
            value_t mean_adj = (x(i) - m(i))/(s(i) * s(i));
            mean_.beval(seed * mean_adj, i, 0, pol);
        }

        for (size_t i = 0; i < x_.rows(); ++i) {
            value_t x_adj = (m(i) - x(i))/(s(i) * s(i));
            x_.beval(seed * x_adj, i, 0, pol);
        }

    }

private:
    void update_cache()
    {
        log_sigma_ = sigma_.get().array().log().sum();
        is_pos_def_ = !(sigma_.get().array() <= 0).any();
    }

    value_t log_sigma_;
    size_t is_pos_def_;
};

// Case 6: vsm
template <class XExprType
        , class MeanExprType
        , class SigmaExprType>
struct NormalAdjLogPDFNode<XExprType, MeanExprType, SigmaExprType,
                           std::tuple<vec, scl, 
                                std::enable_if_t<util::is_mat_v<SigmaExprType>,
                                    typename util::shape_traits<SigmaExprType>::shape_t>> >:
    details::NormalBase<XExprType, MeanExprType, SigmaExprType>,
    core::ExprBase<NormalAdjLogPDFNode<XExprType, MeanExprType, SigmaExprType>>
{
private:
    using base_t = details::NormalBase<
        XExprType, MeanExprType, SigmaExprType>;
    
public:
    using typename base_t::x_t;
    using typename base_t::mean_t;
    using typename base_t::sigma_t;
    using typename base_t::value_t;
    using typename base_t::var_t;
    using base_t::x_;
    using base_t::mean_;
    using base_t::sigma_;
    using base_t::bind;
    using base_t::bind_size;
    using base_t::single_bind_size;

    NormalAdjLogPDFNode(const x_t& x,
                        const mean_t& mean,
                        const sigma_t& sigma)
        : base_t(x, mean, sigma)
        , llt_(sigma.rows())
        , log_det_{0}
        , is_pos_def_{false}
        , inv_(sigma.rows(), sigma.cols())
        , z_(mean.cols())
    {
        // must be square matrix
        assert(sigma_.rows() == sigma_.cols());
        assert(x_.rows() == sigma_.rows());

        if constexpr (util::is_constant_v<sigma_t>) {
            this->update_cache();
        }
    }

    const var_t& feval()
    {
        auto&& x = x_.feval();
        auto&& m = mean_.feval();
        sigma_.feval();

        if constexpr (!util::is_constant_v<sigma_t>) {
            this->update_cache();
        }

        if (!is_pos_def_) {
            return this->get() = util::neg_inf<value_t>;
        }
        
        z_ = inv_ * (x.array() - m).matrix();
        value_t sq_term = (x.array() - m).matrix().transpose() * z_;
        
        return this->get() = -0.5 * sq_term - log_det_; 
    }

    void beval(value_t seed, size_t, size_t, util::beval_policy pol)
    {
        if (seed == 0 || !is_pos_def_) return;

        if constexpr (!util::is_constant_v<sigma_t>) {
            for (size_t j = 0; j < sigma_.cols(); ++j) {
                for (size_t i = 0; i < sigma_.rows(); ++i) {
                    value_t sigma_adj = -0.5 * (inv_(i,j) - z_(i)*z_(j));
                    sigma_.beval(seed * sigma_adj, i, j, pol);
                }
            }
        }

        mean_.beval(seed * z_.sum(), 0, 0, pol);

        for (size_t i = 0; i < x_.rows(); ++i) {
            x_.beval(seed * -z_(i), i, 0, pol);
        }

    }

private:
    void update_cache() {
        llt_.compute(sigma_.get());
        is_pos_def_ = (llt_.info() == Eigen::Success);
        if (is_pos_def_) {
            log_det_ = std::log(llt_.matrixL().determinant());
            inv_ = llt_.solve(mat_t::Identity(sigma_.rows(), sigma_.cols()));
        }
    }

    using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;

    Eigen::LLT<mat_t, Eigen::Lower> llt_;
    value_t log_det_;
    bool is_pos_def_;
    mat_t inv_;
    vec_t z_;
};

// Case 7: vvm
template <class XExprType
        , class MeanExprType
        , class SigmaExprType>
struct NormalAdjLogPDFNode<XExprType, MeanExprType, SigmaExprType,
                           std::tuple<vec, vec, 
                                std::enable_if_t<util::is_mat_v<SigmaExprType>,
                                    typename util::shape_traits<SigmaExprType>::shape_t>> >:
    details::NormalBase<XExprType, MeanExprType, SigmaExprType>,
    core::ExprBase<NormalAdjLogPDFNode<XExprType, MeanExprType, SigmaExprType>>
{
private:
    using base_t = details::NormalBase<
        XExprType, MeanExprType, SigmaExprType>;
    
public:
    using typename base_t::x_t;
    using typename base_t::mean_t;
    using typename base_t::sigma_t;
    using typename base_t::value_t;
    using typename base_t::var_t;
    using base_t::x_;
    using base_t::mean_;
    using base_t::sigma_;
    using base_t::bind;
    using base_t::bind_size;
    using base_t::single_bind_size;

    NormalAdjLogPDFNode(const x_t& x,
                        const mean_t& mean,
                        const sigma_t& sigma)
        : base_t(x, mean, sigma)
        , llt_(sigma.rows())
        , log_det_{0}
        , is_pos_def_{false}
        , inv_(sigma.rows(), sigma.cols())
        , z_(mean.cols())
    {
        // must be square matrix
        assert(sigma_.rows() == sigma_.cols());
        assert(x_.rows() == mean_.rows());
        assert(x_.rows() == sigma_.rows());

        if constexpr (util::is_constant_v<sigma_t>) {
            this->update_cache();
        }
    }

    const var_t& feval()
    {
        auto&& x = x_.feval();
        auto&& m = mean_.feval();
        sigma_.feval();

        if constexpr (!util::is_constant_v<sigma_t>) {
            this->update_cache();
        }

        if (!is_pos_def_) {
            return this->get() = util::neg_inf<value_t>;
        }
        
        z_ = inv_ * (x - m);
        value_t sq_term = (x - m).transpose() * z_;
        
        return this->get() = -0.5 * sq_term - log_det_; 
    }

    void beval(value_t seed, size_t, size_t, util::beval_policy pol)
    {
        if (seed == 0 || !is_pos_def_) return;

        if constexpr (!util::is_constant_v<sigma_t>) {
            for (size_t j = 0; j < sigma_.cols(); ++j) {
                for (size_t i = 0; i < sigma_.rows(); ++i) {
                    value_t sigma_adj = -0.5 * (inv_(i,j) - z_(i)*z_(j));
                    sigma_.beval(seed * sigma_adj, i, j, pol);
                }
            }
        }

        for (size_t i = 0; i < mean_.rows(); ++i) {
            mean_.beval(seed * z_(i), i, 0, pol);
        }

        for (size_t i = 0; i < x_.rows(); ++i) {
            x_.beval(seed * -z_(i), i, 0, pol);
        }

    }

private:
    void update_cache() {
        llt_.compute(sigma_.get());
        is_pos_def_ = (llt_.info() == Eigen::Success);
        if (is_pos_def_) {
            log_det_ = std::log(llt_.matrixL().determinant());
            inv_ = llt_.solve(mat_t::Identity(sigma_.rows(), sigma_.cols()));
        }
    }

    using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;

    Eigen::LLT<mat_t, Eigen::Lower> llt_;
    value_t log_det_;
    bool is_pos_def_;
    mat_t inv_;
    vec_t z_;
};

} // namespace stat

template <class XType
        , class MeanType
        , class SigmaType
        , class = std::enable_if_t<
            util::is_convertible_to_ad_v<XType> &&
            util::is_convertible_to_ad_v<MeanType> &&
            util::is_convertible_to_ad_v<SigmaType> &&
            util::any_ad_v<XType, MeanType, SigmaType> > >
inline auto normal_adj_log_pdf(const XType& x,
                               const MeanType& mean,
                               const SigmaType& sigma)
{
    using x_expr_t = util::convert_to_ad_t<XType>;
    using mean_expr_t = util::convert_to_ad_t<MeanType>;
    using sigma_expr_t = util::convert_to_ad_t<SigmaType>;
    x_expr_t x_expr = x;
    mean_expr_t mean_expr = mean;
    sigma_expr_t sigma_expr = sigma;
    return stat::NormalAdjLogPDFNode<
        x_expr_t, mean_expr_t, sigma_expr_t>(x_expr, mean_expr, sigma_expr);
}

} // namespace ad
