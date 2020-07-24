#pragma once
#include <fastad_bits/expr_base.hpp>
#include <fastad_bits/value_view.hpp>
#include <fastad_bits/type_traits.hpp>

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
     * By this point, expression root is bound to the value region
     * as the one that var_view (and current EqNode) points to.
     * Then, forward evaluating expression will have already cached the result
     * where var_view is pointing to.
     *
     * @return  result of forward evaluating expression 
     */
    const var_t& feval()
    {
        return expr_.feval();
    }

    /** 
     * Backward evaluate i,jth expression and 
     * then backward evaluates expression using leaf's (full) adjoint.
     * We use the full adjoint because many expressions could use the same placeholder,
     * and hence current seed is only a component of the full partial derivative.
     * It is assumed that at the time of calling beval,
     * all expressions using placeholder have backward evaluated.
     */
    void beval(value_t seed, size_t i, size_t j)
    {
        var_view_.beval(seed, i, j);
        expr_.beval(var_view_.get_adj(i,j), i, j);
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

    constexpr size_t single_bind_size() const
    { return 0; }

private:
    var_view_t var_view_;
    expr_t expr_;
};

} // namespace core
} // namespace ad
