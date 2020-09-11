#pragma once
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_adj_view.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/util/size_pack.hpp>
#include <fastad_bits/util/value.hpp>

namespace ad {
namespace core {

/**
 * LogDetNode represents the log (absolute) determinant of a matrix.
 * No other shapes are permitted for this node.
 * Decomposition functor of type DecompType is provided to
 * define the policy in how to compute forward and backward-evaluation.
 *
 * The node assumes the same value type as that of the vector expression.
 * It is always a scalar shape.
 *
 * @tparam  DecompType      decomposition type
 * @tparam  ExprType        type of vector expression
 */

template <class DecompType, class ExprType>
struct LogDetNode:
    ValueAdjView<typename util::expr_traits<ExprType>::value_t,
                 ad::scl>,
    ExprBase<LogDetNode<DecompType, ExprType>>
{
private:
    using decomp_t = DecompType;
    using expr_t = ExprType;
    using expr_value_t = typename util::expr_traits<expr_t>::value_t;

    static_assert(!util::is_scl_v<expr_t>);

public:
    using value_adj_view_t = ValueAdjView<expr_value_t, ad::scl>;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    LogDetNode(const expr_t& expr)
        : value_adj_view_t(nullptr, nullptr, 1, 1)
        , expr_{expr}
        , decomp_(expr.rows())
    {
        assert(expr.rows() == expr.cols());
    }

    const var_t& feval()
    {
        return this->get() = decomp_.fmap(expr_.feval());
    }

    void beval(value_t seed)
    {
        if (seed == 0 || !decomp_.valid()) return;
        auto a_inv_t = decomp_.bmap().array();
        expr_.beval(seed * a_inv_t);
    }

    ptr_pack_t bind_cache(ptr_pack_t begin)
    {
        begin = expr_.bind_cache(begin);
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
        return {this->size(), 0}; 
    }

private:
    using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;
    expr_t expr_;
    decomp_t decomp_;
};

} // namespace core

/*
 * Default method for decomposing a matrix for log determinant.
 */
template <class ValueType>
struct LogDetFullPivLU
{
    using value_t = ValueType;

    LogDetFullPivLU(size_t rows)
        : lu_(rows, rows)
    {}
    
    template <class T>
    value_t fmap(const Eigen::MatrixBase<T>& X)
    {
        lu_.compute(X);
        return std::log(std::abs(lu_.determinant()));
    }

    auto bmap() const 
    {
        return lu_.inverse().transpose();
    }

    bool valid() const { return lu_.isInvertible(); }

private:
    using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;
    Eigen::FullPivLU<mat_t> lu_;
};

/*
 * Decomposing a positive or negative semi-definite matrix for log determinant.
 */
template <class ValueType>
struct LogDetLDLT
{
    using value_t = ValueType;

    LogDetLDLT(size_t rows)
        : ldlt_(rows)
        , inv_()
    {}
    
    template <class T>
    value_t fmap(const Eigen::MatrixBase<T>& X)
    {
        ldlt_.compute(X);
        value_t logdet = std::log(std::abs(ldlt_.vectorD().prod()));
        valid_ = (std::isfinite(logdet)) && (ldlt_.info() == Eigen::Success);
        return logdet;
    }

    // Important to save the inverse, since otherwise Eigen
    // will dynamically allocate every time and result in bad-alloc.
    const auto& bmap() 
    {
        size_t n = ldlt_.rows();
        return inv_ = ldlt_.solve(mat_t::Identity(n, n));
    }

    bool valid() const { return valid_; }

private:
    using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;
    bool valid_ = false;
    Eigen::LDLT<mat_t> ldlt_;
    mat_t inv_;
};

/*
 * Decomposing a positive definite matrix for log determinant.
 */
template <class ValueType>
struct LogDetLLT
{
    using value_t = ValueType;

    LogDetLLT(size_t rows)
        : llt_(rows)
        , inv_()
    {}
    
    template <class T>
    value_t fmap(const Eigen::MatrixBase<T>& X)
    {
        llt_.compute(X);
        value_t logdet = std::log(std::abs(llt_.matrixL().determinant()));
        return 2. * logdet;
    }

    // Important to save the inverse, since otherwise Eigen
    // will dynamically allocate every time and result in bad-alloc.
    const auto& bmap() 
    {
        size_t n = llt_.rows();
        return inv_ = llt_.solve(mat_t::Identity(n, n));
    }

    bool valid() const 
    {
        return (llt_.info() == Eigen::Success); 
    }

private:
    using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;
    Eigen::LLT<mat_t> llt_;
    mat_t inv_;
};

/*
 * Creates a determinant expression node with a policy that defines the decomposition.
 * The default decomposition is Eigen::FullPivLU.
 * Currently, we support DetLDLT and DetLLT for some specialized matrices.
 * If x is a constant, the decomposition is ignored and 
 * will always just invoke member function determinant of the underlying Eigen object.
 */
template <template <class> class DecompType = LogDetFullPivLU
        , class T
        , class = std::enable_if_t<
            util::is_convertible_to_ad_v<T> &&
            util::any_ad_v<T> > >
inline auto log_det(const T& x)
{
    using expr_t = util::convert_to_ad_t<T>;
    using value_t = typename util::expr_traits<expr_t>::value_t;
    expr_t expr = x;

    // optimization for when expression is constant
    if constexpr (util::is_constant_v<expr_t>) {
        static_assert(!util::is_scl_v<expr_t>);
        using var_t = util::constant_var_t<value_t, ad::scl>;
        var_t out = std::log(std::abs(expr.feval().determinant()));
        return ad::constant(out);
    } else {
        return core::LogDetNode<DecompType<value_t>, expr_t>(expr);
    }
}

} // namespace ad
