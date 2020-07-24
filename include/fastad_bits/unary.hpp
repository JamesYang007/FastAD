#pragma once
#include <fastad_bits/expr_base.hpp>
#include <fastad_bits/type_traits.hpp>
#include <fastad_bits/shape_traits.hpp>
#include <fastad_bits/value_view.hpp>

namespace ad {
namespace core {

/**
 * UnaryNode represents a univariate function on an expression.
 * All mathematical functions defined in math.hpp will
 * simply return a UnaryNode that stores all information 
 * to compute forward and backward direction.
 *
 * The unary function is a vectorized mapping when the underlying
 * expression is a vector or a matrix.
 *
 * The value type, shape type, and variable type
 * are the same as those of the underlying expression.
 *
 * @tparam  Unary       univariate functor that stores fmap and bmap defining
 *                      its corresponding function and derivative mapping
 * @tparam  ExprType    type of expression to apply Unary on
 */

template <class Unary
        , class ExprType>
struct UnaryNode:
    core::ValueView<typename util::expr_traits<ExprType>::value_t,
                    typename util::shape_traits<ExprType>::shape_t>,
    core::ExprBase<UnaryNode<Unary, ExprType>>
{
private:
    using expr_t = ExprType;
    static_assert(util::is_expr_v<expr_t>);

public:
    using value_view_t = core::ValueView<
        typename util::expr_traits<expr_t>::value_t, 
        typename util::shape_traits<expr_t>::shape_t >;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::bind;

    UnaryNode(const expr_t& expr)
        : value_view_t(nullptr, expr.rows(), expr.cols())
        , expr_(expr)
    {}

    /**
     * Forward evaluation first evaluates given expression,
     * evaluates univariate functor on the result, and caches the result.
     *
     * @return  const reference of the cached result.
     */
    const var_t& feval()
    {
        if constexpr (util::is_scl_v<expr_t>) {
            return this->get() = Unary::fmap(expr_.feval());
        } else {
            return this->get() = Unary::fmap(expr_.feval().array()).matrix();
        }
    }

    /**
     * Backward evaluation sets current adjoint to seed,
     * multiplies seed with univariate function derivative on expression value,
     * and backward evaluate expression with the result as the new seed.
     *
     * seed * df/dx(w) -> new seed for expression
     *
     * where f is the univariate function, and w is the expression value.
     * It is assumed that feval is called before beval.
     */
    void beval(value_t seed, size_t i, size_t j)
    {
        expr_.beval(seed * Unary::bmap(expr_.get(i,j)), i, j);
    }

    /**
     * Disables usual binding rules and applies a recursive form.
     * First binds for underlying expression then binds itself.
     * Ignores binding VarViews since the precondition states that
     * they will have been bound prior to AD expression construction.
     *
     * @return  next pointer not bound by underlying expression and itself.
     */
    value_t* bind(value_t* begin)
    { 
        value_t* next = begin;
        if constexpr (!util::is_var_view_v<expr_t>) {
            next = expr_.bind(next);
        }
        return value_view_t::bind(next);
    }

    /**
     * Recursively gets the total number of values needed by the expression.
     * Since a UnaryNode is a vectorized operation, it binds exactly
     * the same number as its size.
     * @return  bind size
     */
    size_t bind_size() const 
    { 
        return single_bind_size() + expr_.bind_size();
    }

    size_t single_bind_size() const
    {
        return this->size();
    }

private:
    expr_t expr_;
};

} // namespace core
} // namespace ad
