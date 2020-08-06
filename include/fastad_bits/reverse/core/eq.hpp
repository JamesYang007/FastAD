#pragma once
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_view.hpp>
#include <fastad_bits/util/type_traits.hpp>

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
    ValueView<typename util::expr_traits<VarViewType>::value_t,
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
    using value_view_t = ValueView<var_view_value_t,
                                   var_view_shape_t>;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::bind;

    EqNode(const var_view_t& var_view, 
           const expr_t& expr)
        : value_view_t(nullptr, 
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
     *
     * When pol is "all", it is a special signal from GlueNode or ForEachIterNode
     * that we should back-evaluate every non-zero adjoint expressions.
     * In general, after the right-most expression is backward-evaluated,
     * every element of every left-ward expression may have been a dependency.
     * Note: this is why users should always reset adjoint before back-evaluating -
     * adjoints can accumulate otherwise.
     *
     * Assumptions:
     * - The only exceptional node is this EqNode and any further embedded
     *   EqNode inside the (right) expression cannot be referenced outside this current node.
     *
     * - If the i,j adjoint of variable is 0 after accounting for seeding,
     *   there is no need to back-evaluate the i,jth expression by the above reason;
     *   it implies that the adjoints of these further placeholders will remain 0
     *   and all other nodes will have only computed a bunch of numbers 
     *   only to be multiplied by 0 (their seed).
     */
    void beval(value_t seed, size_t i, size_t j, util::beval_policy pol)
    {
        if (pol == util::beval_policy::all) {
            assert(seed == 0);
            for (size_t k = 0; k < var_view_.cols(); ++k) {
                for (size_t l = 0; l < var_view_.rows(); ++l) {
                    if (var_view_.get_adj(l,k)) {
                        expr_.beval(var_view_.get_adj(l,k), l, k, 
                                    util::beval_policy::single);
                    }
                }
            }
        } else {
            var_view_.beval(seed, i, j, pol);
            expr_.beval(var_view_.get_adj(i,j), i, j, pol);
        }
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
    value_t* bind(value_t* begin) 
    {
        // bind current eqnode to var_view's values
        value_view_t::bind(var_view_.data());

        begin = expr_.bind(begin);
        begin -= expr_.single_bind_size();

        // only bind root to var_view's values, not recursively down
        expr_.value_view_t::bind(var_view_.data());

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
    size_t bind_size() const 
    { 
        assert(expr_.bind_size() >= expr_.single_bind_size());
        return expr_.bind_size() - expr_.single_bind_size();
    }

    constexpr size_t single_bind_size() const { return 0; }

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
    ValueView<typename util::expr_traits<VarViewType>::value_t,
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
    using value_view_t = ValueView<var_view_value_t,
                                   var_view_shape_t>;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::bind;

    OpEqNode(const var_view_t& var_view, 
             const expr_t& expr)
        : value_view_t(nullptr, 
                       var_view.rows(), 
                       var_view.cols())
        , cache_(nullptr, var_view.rows(), var_view.cols())
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
        auto& v_val = var_view_.get();
        auto& expr_val = expr_.feval();
        if constexpr (util::is_scl_v<var_view_t>) {
            return this->get() = Op::fmap(v_val, expr_val);
        } else if constexpr (util::is_scl_v<expr_t>) {
            auto v_arr = v_val.array();
            return this->get() = Op::fmap(v_arr, expr_val);
        } else {
            auto v_arr = v_val.array();
            return this->get() = Op::fmap(v_arr, expr_val.array());
        }
    }

    void beval(value_t seed, size_t i, size_t j, util::beval_policy pol)
    {
        if (pol == util::beval_policy::all) {
            assert(seed == 0);

            // must reset var view to previous val
            var_view_.get() = cache_.get();

            // take advantage of the fact that cache size is the same as adjoint size
            // need to save old adjoints since once we start beval, the adjoints of lhs may get updated.
            cache_.get() = var_view_.get_adj();

            // update left-side adjoint (with minor math optimizations)
            // technically, this should happen after expr beval,
            // but there shouldn't be any issues since the only problem that may arise
            // in switching the order is if feval of expr depends on any changed states from
            // feval of old lhs (note current expr is equivalent to u = u_old @ expr),
            // but old lhs feval is trivial and does not change any states except possibly itself.
            // In the latter case, feval of expr will perform that change anyway if it references old lhs.
            //
            // Must be careful about aliasing issues. 
            // Since var_view_ already got replaced with old values, should be ok.
            for (size_t k = 0; k < var_view_.cols(); ++k) {
                for (size_t l = 0; l < var_view_.rows(); ++l) {
                    if (var_view_.get_adj(l,k)) {
                        var_view_.get_adj(l,k) *= 
                            Op::blmap(var_view_.get(l,k), expr_.get(l,k));
                    }
                }
            }

            // special optimization when expression is scalar
            if constexpr (util::is_scl_v<expr_t>) {
                auto rseed = Op::brmap_scl(var_view_.get(), 
                                           cache_.get(), 
                                           expr_.get());
                expr_.beval(rseed, 0, 0, util::beval_policy::single);

            } else {
                for (size_t k = 0; k < var_view_.cols(); ++k) {
                    for (size_t l = 0; l < var_view_.rows(); ++l) {
                        if (cache_.get(l,k)) {
                            auto rseed = Op::brmap(var_view_.get(l,k), expr_.get(l,k));
                            expr_.beval(cache_.get(l,k) * rseed,
                                        l, k, util::beval_policy::single);
                        }
                    }
                }
            }

        } else {
            // must reset var view to previous val (only for (i,j) elt)
            var_view_.get(i,j) = cache_.get(i,j);
            var_view_.beval(seed, i, j, pol);
            auto lseed = Op::blmap(var_view_.get(i,j), expr_.get(i,j));
            auto rseed = Op::brmap(var_view_.get(i,j), expr_.get(i,j));
            auto orig_adj = var_view_.get_adj(i,j);
            var_view_.get_adj(i,j) *= lseed;
            expr_.beval(orig_adj * rseed, i, j, pol);
        }
    }

    /**
     * Binds itself to view the same values as the variable viewer,
     * binds the RHS expression, but DOES NOT bind the root of RHS to LHS like EqNode.
     * Needs to cache the previous LHS value, so we store a member value viewer to bind.
     *
     * @return  next pointer not bound by expression or itself.
     */
    value_t* bind(value_t* begin) 
    {
        value_view_t::bind(var_view_.data());
        if constexpr (!util::is_var_view_v<expr_t>) {
            begin = expr_.bind(begin);
        }
        return cache_.bind(begin);
    }

    size_t bind_size() const { 
        return single_bind_size() + 
                expr_.bind_size(); 
    }

    size_t single_bind_size() const { return cache_.size(); }

private:
    value_view_t cache_;
    var_view_t var_view_;
    expr_t expr_;
};

struct AddEq
{
    template <class T, class U>
    static inline constexpr T& fmap(T& x, const U& y) 
    { return x += y; }

    template <class T, class U>
    static inline constexpr auto blmap(const T&, const U&) 
    { return 1.; }

    template <class T, class U>
    static inline constexpr auto brmap(const T&, const U&) 
    { return 1.; }

    template <class T, class S, class U>
    static inline constexpr double brmap_scl(const T&, 
                                             const S& x_adj,
                                             const U&) 
    { 
        if constexpr (util::is_eigen_v<T>) { return x_adj.array().sum(); } 
        else { return x_adj; }
    }
};

struct SubEq
{
    template <class T, class U>
    static inline constexpr T& fmap(T& x, const U& y) 
    { return x -= y; }

    template <class T, class U>
    static inline constexpr auto blmap(const T&, const U&) 
    { return 1.; }

    template <class T, class U>
    static inline constexpr auto brmap(const T&, const U&) 
    { return -1.; }

    template <class T, class S, class U>
    static inline constexpr double brmap_scl(const T&, 
                                             const S& x_adj,
                                             const U&) 
    { 
        if constexpr (util::is_eigen_v<T>) { return -x_adj.array().sum(); } 
        else { return -x_adj; }
    }
};

struct MulEq
{
    template <class T, class U>
    static inline constexpr T& fmap(T& x, const U& y) 
    { return x *= y; }

    template <class T, class U>
    static inline constexpr auto blmap(const T&, const U& y) 
    { return y; }

    template <class T, class U>
    static inline constexpr auto brmap(const T& x, const U&) 
    { return x; }

    template <class T, class S, class U>
    static inline constexpr double brmap_scl(const T& x, 
                                             const S& x_adj,
                                             const U&) 
    { 
        if constexpr (util::is_eigen_v<T>) { 
            return (x_adj.array() * x.array()).sum();
        }
        else { return x_adj * x; }
    }
};

struct DivEq
{
    template <class T, class U>
    static inline constexpr T& fmap(T& x, const U& y) 
    { return x /= y; }

    template <class T, class U>
    static inline constexpr auto blmap(const T&, const U& y) 
    { return 1. / y; }

    template <class T, class U>
    static inline constexpr auto brmap(const T& x, const U& y) 
    { return -x / (y*y); }

    template <class T, class S, class U>
    static inline constexpr double brmap_scl(const T& x, 
                                             const S& x_adj,
                                             const U& y) 
    { 
        if constexpr (util::is_eigen_v<T>) { 
            return -(x_adj.array() * x.array()).sum() / (y*y);
        }
        else { return -x_adj * x / (y*y); }
    }
};

} // namespace core
} // namespace ad
