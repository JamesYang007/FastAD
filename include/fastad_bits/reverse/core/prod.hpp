#pragma once
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_adj_view.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/util/shape_traits.hpp>
#include <fastad_bits/util/size_pack.hpp>
#include <fastad_bits/util/value.hpp>

namespace ad {
namespace core {

/**
 * ProdIterNode represents a product of expressions.
 * It is a vectorized operation.
 *
 * Ex. f(e1) * f(e2) * ... * f(en)
 * 
 * @tparam  VecType     type of vector which stores the expressions.
 */

template <class VecType>
struct ProdIterNode :
    ValueAdjView<typename util::expr_traits< 
                    typename VecType::value_type >::value_t,
                 typename util::shape_traits< 
                    typename VecType::value_type >::shape_t >,
    ExprBase<ProdIterNode<VecType>>
{
private:
    using vec_elem_t = typename VecType::value_type;
    using elem_value_t = typename util::expr_traits<vec_elem_t>::value_t;
    using elem_shape_t = typename util::shape_traits<vec_elem_t>::shape_t;
    
public:
    using value_adj_view_t = ValueAdjView<elem_value_t, elem_shape_t>;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    ProdIterNode(const VecType& exprs)
        : value_adj_view_t(nullptr, nullptr,
                       (exprs.size() == 0) ? 0 : exprs[0].rows(),
                       (exprs.size() == 0) ? 0 : exprs[0].cols())
        , exprs_{exprs}
    {}

    /** 
     * Forward evaluate by evaluating every expression left to right
     * and multiplying the results.
     *
     * @return forward evaluation of product of every expression.
     */
    const var_t& feval()
    {
        this->ones();
        for (auto& expr : exprs_) {
            util::to_array(this->get()) *= util::to_array(expr.feval());
        }
        return this->get();
    }

    /** 
     * Backward evaluate from right to left.
     */
    template <class T>
    void beval(const T& seed)
    {
        if (exprs_.size() == 0) return;
        auto&& a_val = util::to_array(this->get());
        auto&& a_adj = util::to_array(this->get_adj());
        a_adj = seed;

        size_t idx = exprs_.size() - 1;
        std::for_each(exprs_.rbegin(), exprs_.rend(),
            [&](auto& expr) {
                auto&& a_expr = util::to_array(expr.get());

                // if current expr's value is 0, cannot optimize,
                // must recompute product leaving current index out.
                bool no_optimize;
                if constexpr (util::is_scl_v<vec_elem_t>) {
                    no_optimize = (expr.get() == 0);
                } else {
                    no_optimize = (expr.get().array() == 0).any();
                }

                if (no_optimize) {

                    util::ones(a_val);
                    for (size_t k = 0; k < exprs_.size(); ++k) {
                        if (k != idx) a_val *= util::to_array(exprs_[k].get());
                    }
                    expr.beval(a_adj * a_val);
                    a_val *= util::to_array(exprs_[idx].get());

                } else {
                    expr.beval(a_adj * a_val / a_expr);
                }

                --idx;
            });
    }

    /**
     * Bind every expression from left to right then bind itself.
     *
     * @return  the next pointer not bound by any of the expressions and itself.
     */
    ptr_pack_t bind_cache(ptr_pack_t begin)
    {
        for (auto& expr : exprs_) {
            begin = expr.bind_cache(begin);
        }
        return value_adj_view_t::bind(begin);
    }

    util::SizePack bind_cache_size() const 
    { 
        util::SizePack out = util::SizePack::Zero();
        for (const auto& expr : exprs_) {
            out += expr.bind_cache_size();
        }
        return out + single_bind_cache_size();
    }

    util::SizePack single_bind_cache_size() const
    { 
        return {this->size(), this->size()}; 
    }

private:
    VecType exprs_;
};

/**
 * ProdElemNode represents a product of elements of an expression.
 * 
 * @tparam  VecType     type of vector which stores the expressions.
 */

template <class ExprType>
struct ProdElemNode :
    ValueAdjView<typename util::expr_traits<ExprType>::value_t,
                 ad::scl>,
    ExprBase<ProdElemNode<ExprType>>
{
private:
    using expr_t = ExprType;
    using expr_value_t = typename util::expr_traits<expr_t>::value_t;
    using expr_shape_t = typename util::expr_traits<expr_t>::shape_t;
    
public:
    using value_adj_view_t = ValueAdjView<expr_value_t, ad::scl>;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    ProdElemNode(const expr_t& expr)
        : value_adj_view_t(nullptr, nullptr, 1, 1)
        , expr_{expr}
        , adj_cache_(nullptr, expr.rows(), expr.cols())
    {}

    /** 
     * Forward evaluate by evaluating the expression and multiplying the elements.
     *
     * @return forward evaluation of sum of functor on every expr.
     */
    const var_t& feval()
    {
        auto&& res = expr_.feval();
        if constexpr (util::is_scl_v<expr_t>) {
            return this->get() = res;
        } else {
            return this->get() = res.prod();
        }
    }

    /** 
     * Backward evaluate from right to left.
     *
     * Note that GlueNode and alike guarantee to seed 0 when pol is single.
     * See EqNode for why we can preemptively return.
     */
    void beval(value_t seed)
    {
        for (size_t k = 0; k < expr_.cols(); ++k) {
            for (size_t l = 0; l < expr_.rows(); ++l) {

                adj_cache_.get(l,k) = seed;

                // if current expr's value is 0, cannot optimize,
                // must recompute product leaving current index out.
                if (expr_.get(l,k) == 0) {
                    for (size_t kk = 0; kk < expr_.cols(); ++kk) {
                        for (size_t ll = 0; ll < expr_.rows(); ++ll) {
                            if (ll != l || kk != k) {
                                adj_cache_.get(l,k) *= expr_.get(ll, kk);
                            }
                        }
                    }
                } else {
                    adj_cache_.get(l,k) *= this->get() / expr_.get(l,k);
                }

            }
        }
        expr_.beval(util::to_array(adj_cache_.get()));
    }

    /**
     * Bind every expression from left to right then bind itself.
     *
     * @return  the next pointer pack not bound by any of the expressions and itself.
     */
    ptr_pack_t bind_cache(ptr_pack_t begin)
    {
        begin = expr_.bind_cache(begin);
        begin.adj = adj_cache_.bind(begin.adj);
        auto adj = begin.adj;
        begin.adj = nullptr;
        begin = value_adj_view_t::bind(begin);
        begin.adj = adj;
        return begin;
    }

    util::SizePack bind_cache_size() const 
    { 
        return expr_.bind_cache_size() + 
                single_bind_cache_size();
    }

    util::SizePack single_bind_cache_size() const
    { 
        return {this->size(), expr_.size()}; 
    }

private:
    using value_view_t = ValueView<value_t, expr_shape_t>;
    expr_t expr_;
    value_view_t adj_cache_;
};

} // namespace core

/**
 * Helper function to create a ProdIterNode.
 * If there are no expressions to iterate over, 
 * and if each expression types are constant,
 * then returns a constant of 0 scalar, or empty Eigen vector/matrix.
 * If not constant, then ProdIterNode is returned that will effectively be a noop.
 * Otherwise, if there is at least one expression to iterate over, 
 * returns a ProdIterNode that will not be a noop.
 */

template <class Iter, class Lmda>
inline auto prod(Iter begin, Iter end, Lmda f)
{
    using expr_t = std::decay_t<decltype(f(*begin))>;
    using value_t = typename util::expr_traits<expr_t>::value_t;
    using shape_t = typename util::shape_traits<expr_t>::shape_t;
    using var_t = util::constant_var_t<value_t, shape_t>;

    // optimized for f that returns a constant node
    if constexpr (util::is_constant_v<expr_t>) {
        if (std::distance(begin, end) <= 0) return ad::constant(var_t(0));
        var_t prod = f(*begin).feval();     // value_t or Eigen::Matrix
        std::for_each(std::next(begin), end, 
                [&](const auto& x) 
                { 
                    util::to_array(prod) *= util::to_array(f(x).feval()); 
                });
        return ad::constant(prod);
    } else {
        std::vector<expr_t> exprs;
        exprs.reserve(std::distance(begin, end));
        std::for_each(begin, end, 
                [&](const auto& x) {
                    exprs.emplace_back(f(x));
                });
        return core::ProdIterNode<std::vector<expr_t>>(exprs);
    }
}

template <class Derived
        , class = std::enable_if_t<
            util::is_convertible_to_ad_v<Derived> &&
            util::any_ad_v<Derived> > >
inline auto prod(const Derived& x)
{
    using expr_t = util::convert_to_ad_t<Derived>;
    expr_t expr = x;

    // optimized when expr is constant
    if constexpr (util::is_constant_v<expr_t>) {
        if constexpr (util::is_scl_v<expr_t>) return expr;
        else {
            return ad::constant(expr.feval().array().prod());
        }
    } else {
        return core::ProdElemNode<expr_t>(expr);
    }
}

} // namespace ad
