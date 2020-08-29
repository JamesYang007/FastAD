#pragma once
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_adj_view.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/util/size_pack.hpp>
#include <fastad_bits/util/ptr_pack.hpp>
#include <fastad_bits/util/value.hpp>

namespace ad {
namespace core {

/** 
 * EqNode represents the mathematical placeholders for later substitution.
 * Ex. u = sin(cos(y))
 * u is the placeholder for the expression sin(cos(y)).
 * EqNode allows users to optimize calculation by placing placeholders 
 * for quantities that are used often.
 * As a result, users can form new expressions using "u".
 * It is guaranteed then that sin(cos(y)) will only be evaluated once.
 *
 * Currently, it is not able to support trivial expressions of the type
 * u = x
 * where x is some variable (or variable viewer).
 *
 * @tparam VarViewType  type of variable viewer acting as a placeholder
 * @tparam ExprType     type of expression to placehold
 */

template <class VarViewType, class ExprType>
struct EqNode:
    ValueAdjView<typename util::expr_traits<VarViewType>::value_t,
                 typename util::shape_traits<VarViewType>::shape_t>,
    ExprBase<EqNode<VarViewType, ExprType>>
{
private:
    using var_view_t = VarViewType;
    using expr_t = ExprType;
    using var_view_value_t = typename 
        util::expr_traits<var_view_t>::value_t;
    using var_view_shape_t = typename 
        util::expr_traits<var_view_t>::shape_t;

    // check that VarViewType is indeed a VarView
    static_assert(util::is_var_view_v<var_view_t>);

    // check that ExprType is indeed an AD expression
    static_assert(util::is_expr_v<expr_t>);

    // assert that ExprType is not a VarView
    static_assert(!util::is_var_view_v<expr_t>);

    // value types of VarViewType and ExprType must match
    static_assert(std::is_same_v<
            var_view_value_t,
            typename util::expr_traits<expr_t>::value_t>);

    // shape types of VarViewType and ExprType must match
    static_assert(std::is_same_v<
            var_view_shape_t,
            typename util::expr_traits<expr_t>::shape_t>);

public:
    using value_adj_view_t = ValueAdjView<var_view_value_t,
                                          var_view_shape_t>;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    EqNode(const var_view_t& var_view, 
           const expr_t& expr)
        : value_adj_view_t(nullptr, nullptr,
                           var_view.rows(), 
                           var_view.cols())
        , var_view_(var_view)
        , expr_(expr)
    {
        // assert that the two have same dimensions 
        assert(var_view_.rows() == expr_.rows());
        assert(var_view_.cols() == expr_.cols());
    }

    /** 
     * Forward evaluation evaluates expression and returns result.
     * Note that bind would have been called before this function.
     *
     * @return  result of forward evaluating expression 
     */
    const var_t& feval()
    {
        return this->get() = var_view_.get() = expr_.feval();
    }

    /** 
     * Backward evaluate all expressions that have non-zero (full) adjoint.
     * We use the full adjoint because many expressions could use the same placeholder,
     * and hence current seed is only a component of the full partial derivative.
     * It is assumed that at the time of calling beval,
     * all expressions using placeholder have backward evaluated.
     */
    template <class T>
    void beval(const T& seed)
    {
        var_view_.beval(seed);
        auto&& a_adj = util::to_array(var_view_.get_adj());
        expr_.beval(a_adj);
    }

    /**
     * Binds the expression, strips the data from root of expression,
     * rebinds the root of expression to view the same thing as the placeholder,
     * then returns the stripped data.
     * Effectively, the placeholder, the current EqNode, and the root of expression
     * are viewing the same values to save space and copying.
     * Ignores expression if it is a VarView.
     *
     * @return  next pointer not bound by expression.
     */
    ptr_pack_t bind_cache(ptr_pack_t begin)
    {
        ptr_pack_t var_ptr_pack(var_view_.data(), var_view_.data_adj());

        // bind current eqnode to var_view's values
        value_adj_view_t::bind(var_ptr_pack);

        begin = expr_.bind_cache(begin);
        auto size_pack = expr_.single_bind_cache_size();
        begin.val -= size_pack(0);
        begin.adj -= size_pack(1);

        // only bind root to var_view's values, not recursively down
        using expr_value_adj_view_t = typename expr_t::value_adj_view_t;
        static_cast<expr_value_adj_view_t&>(expr_).bind(var_ptr_pack);

        return begin;
    }

    /**
     * Recursively gets the total number of values needed by the expression.
     * Since a EqNode simply binds to that of the variable viewer,
     * it does not bind any extra amount.
     * It also strips the root of expr_ its binding and rebind it
     * to that of variable view, so we must subtract its size from recursing.
     *
     * @return  bind size
     */
    util::SizePack bind_cache_size() const 
    { 
        assert((expr_.bind_cache_size() >= expr_.single_bind_cache_size()).all());
        return expr_.bind_cache_size() - expr_.single_bind_cache_size();
    }

    util::SizePack single_bind_cache_size() const
    { return {0,0}; }

private:
    var_view_t var_view_;
    expr_t expr_;
};

/** 
 * OpEqNode represents modification of a variable by some operation.
 * Ex. u += 2 * x
 * u is the variable that will be updated with +2*x.
 * It is exactly same memory overhead as EqNode, and internally invokes more copies.
 * This is due to possible self-aliasing issues during backward-evaluation.
 * As a result, it is in general faster to use a regular placeholder like:
 *
 * w = u + 2*x
 *
 * This is only advised if the variable is a scalar, or small-sized multi-dimensional variable,
 * and if it makes the code more readable.
 *
 * General Notes:
 * - Don't use this if the placeholder values must remain the same; 
 *   you must use another placeholder variable with EqNode instead.
 *   This method will modify value viewed by LHS.
 *
 * While EqNode is almost a special case of OpEqNode, we distinguish the two because
 * there are differences in the inherent properties not apparent in the implementation:
 *
 * - an EqNode can only show up once in an expression for a given placeholder,
 *   but OpEqNodes can show up multiple times for the same placeholder.
 *   This is because multiple EqNodes for the same placeholder introduces ambiguity
 *   when back-evaluating.
 *
 * - an EqNode can optimize to have RHS expression bind to the placeholder value area
 *   and directly place forward-evaluated values there, but OpEqNodes cannot simply replace.
 *
 * @tparam VarViewType  type of variable viewer acting as a placeholder
 * @tparam ExprType     type of expression to placehold
 */

template <class Op, class VarViewType, class ExprType>
struct OpEqNode:
    ValueAdjView<typename util::expr_traits<VarViewType>::value_t,
                 typename util::shape_traits<VarViewType>::shape_t>,
    ExprBase<OpEqNode<Op, VarViewType, ExprType>>
{
private:
    using var_view_t = VarViewType;
    using expr_t = ExprType;
    using var_view_value_t = typename 
        util::expr_traits<var_view_t>::value_t;
    using var_view_shape_t = typename 
        util::expr_traits<var_view_t>::shape_t;

    static_assert(util::is_var_view_v<var_view_t>);
    static_assert(util::is_expr_v<expr_t>);
    static_assert(std::is_same_v<
            var_view_value_t,
            typename util::expr_traits<expr_t>::value_t>);
    static_assert(util::is_scl_v<expr_t> ||
                  std::is_same_v<var_view_shape_t,
                      typename util::expr_traits<expr_t>::shape_t>);

public:
    using value_adj_view_t = ValueAdjView<var_view_value_t,
                                          var_view_shape_t>;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    OpEqNode(const var_view_t& var_view, 
             const expr_t& expr)
        : value_adj_view_t(nullptr, nullptr,
                           var_view.rows(), 
                           var_view.cols())
        , cache_(nullptr, nullptr, var_view.rows(), var_view.cols())
        , var_view_(var_view)
        , expr_(expr)
    {
        if constexpr (!util::is_scl_v<expr_t>) {
            assert(var_view_.rows() == expr_.rows());
            assert(var_view_.cols() == expr_.cols());
        }
    }

    const var_t& feval()
    {
        cache_.get() = var_view_.get();  // save previous lhs
        auto&& a_v = util::to_array(var_view_.get());
        auto&& a_expr = util::to_array(expr_.feval());
        util::to_array(this->get()) = Op::fmap(a_v, a_expr);
        return this->get();
    }

    template <class T>
    void beval(const T& seed)
    {
        var_view_.beval(seed);

        // copy old value first before back-evaluating 
        // because expr_ may depend on var_view_, which would have been the old value.
        var_view_.get() = cache_.get();

        // copy current seed such that it doesn't get modified during back-evaluation
        // expr may depend on the var_view_, which may modify its adjoint.
        // MUST reset adjoint so that back evaluation properly accumulates.
        cache_.get_adj() = var_view_.get_adj(); 
        var_view_.reset_adj();

        auto&& a_val = util::to_array(var_view_.get());
        auto&& a_adj = util::to_array(cache_.get_adj());
        auto&& a_expr = util::to_array(expr_.get());
        auto lseed = Op::blmap(a_adj, a_val, a_expr);
        auto rseed = Op::brmap(a_adj, a_val, a_expr);
        expr_.beval(rseed);
        var_view_.beval(lseed);
    }

    /**
     * Binds itself to view the same values as the variable viewer,
     * binds the RHS expression, but DOES NOT bind the root of RHS to LHS like EqNode.
     * Needs to cache the previous LHS value, so we store a member value viewer to bind.
     *
     * @return  next pointer not bound by expression or itself.
     */
    ptr_pack_t bind_cache(ptr_pack_t begin)
    {
        value_adj_view_t::bind({var_view_.data(), var_view_.data_adj()});
        begin = expr_.bind_cache(begin);
        begin = cache_.bind(begin);
        return begin;
    }

    util::SizePack bind_cache_size() const 
    {
        return single_bind_cache_size() + 
                expr_.bind_cache_size(); 
    }

    util::SizePack single_bind_cache_size() const
    {
        return {cache_.size(), cache_.size()}; 
    }

private:
    value_adj_view_t cache_;
    var_view_t var_view_;
    expr_t expr_;
};

struct AddEq
{
    template <class T, class U>
    static inline constexpr T& fmap(T& x, const U& y) 
    { return x += y; }

    template <class S, class T, class U>
    static inline constexpr auto blmap(const S& seed, const T&, const U&) 
    {
        if constexpr (!util::is_eigen_v<T> && util::is_eigen_v<U>) {
            return seed.sum();
        } else {
            return seed;
        }
    }

    template <class S, class T, class U>
    static inline constexpr auto brmap(const S& seed, const T&, const U&) 
    { 
        if constexpr (util::is_eigen_v<T> && !util::is_eigen_v<U>) {
            return seed.sum();
        } else {
            return seed;
        }
    }
};

struct SubEq
{
    template <class T, class U>
    static inline constexpr T& fmap(T& x, const U& y) 
    { return x -= y; }

    template <class S, class T, class U>
    static inline constexpr auto blmap(const S& seed, const T&, const U&) 
    {
        if constexpr (!util::is_eigen_v<T> && util::is_eigen_v<U>) {
            return seed.sum();
        } else {
            return seed;
        }
    }

    template <class S, class T, class U>
    static inline constexpr auto brmap(const S& seed, const T&, const U&) 
    { 
        if constexpr (util::is_eigen_v<T> && !util::is_eigen_v<U>) {
            return -(seed.sum());
        } else {
            return -seed;
        }
    }
};

struct MulEq
{
    template <class T, class U>
    static inline constexpr T& fmap(T& x, const U& y) 
    { return x *= y; }

    template <class S, class T, class U>
    static inline constexpr auto blmap(const S& seed, const T&, const U& y) 
    { 
        if constexpr (!util::is_eigen_v<T> && util::is_eigen_v<U>) {
            return (seed * y).sum();
        } else {
            return seed * y;
        }
    }

    template <class S, class T, class U>
    static inline constexpr auto brmap(const S& seed, const T& x, const U&) 
    { 
        if constexpr (util::is_eigen_v<T> && !util::is_eigen_v<U>) {
            return (seed * x).sum();
        } else {
            return seed * x;
        }
    }
};

struct DivEq
{
    template <class T, class U>
    static inline constexpr T& fmap(T& x, const U& y) 
    { return x /= y; }

    template <class S, class T, class U>
    static inline constexpr auto blmap(const S& seed, const T&, const U& y) 
    { 
        if constexpr (!util::is_eigen_v<T> && util::is_eigen_v<U>) {
            return (seed / y).sum();
        } else {
            return seed / y;
        }
    }

    template <class S, class T, class U>
    static inline constexpr auto brmap(const S& seed, const T& x, const U& y) 
    { 
        static_cast<void>(x);
        if constexpr (util::is_eigen_v<T> && !util::is_eigen_v<U>) {
            return (-seed * x / (y * y)).sum();
        } else {
            return -seed * x / (y * y);
        }
    }
        
};

} // namespace core
} // namespace ad
