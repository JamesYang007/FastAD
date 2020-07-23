#pragma once
#include <fastad_bits/expr_base.hpp>
#include <fastad_bits/value_view.hpp>
#include <fastad_bits/type_traits.hpp>

namespace ad {
namespace core {

/**
 * BinaryNode represents a binary function on two expressions.
 * All mathematical functions defined math.hpp, for example, will
 * simply return a BinaryNode that stores all information 
 * to compute forward and backward direction.
 *
 * Note that if left or right expressions are multi-dimensional,
 * Binary operation must be a vectorized operation.
 *
 * The only possible combinations of shapes is the following:
 * 1) left or right is a scalar
 * 2) both vector
 * 3) both matrix
 *
 * Left and right expressions must have a common value type as per std::common_type.
 * This is the value type that the BinaryNode assumes.
 *
 * @tparam  Binary          binary functor that stores fmap, blmap, brmap defining
 *                          its corresponding function and derivative mapping w.r.t both variables.
 *                          Must be a vectorized function for multi-dimensional inputs.
 * @tparam  LeftExprType    type of left expression to apply Binary on
 * @tparam  RightExprType   type of right expression to apply Binary on
 */

template <class Binary
        , class LeftExprType
        , class RightExprType>
struct BinaryNode:
    ValueView<typename util::expr_traits<LeftExprType>::value_t,
              util::max_shape_t<typename util::shape_traits<LeftExprType>::shape_t,
                                typename util::shape_traits<RightExprType>::shape_t>
                >,
    ExprBase<BinaryNode<Binary, LeftExprType, RightExprType>>
{
private:
    using left_t = LeftExprType;
    using right_t = RightExprType;
    using common_value_t = std::common_type_t<
        typename util::expr_traits<left_t>::value_t,
        typename util::expr_traits<right_t>::value_t
            >;
    using max_shape_t = util::max_shape_t<
        typename util::shape_traits<left_t>::shape_t,
        typename util::shape_traits<right_t>::shape_t
            >;

    // both left and right must AD expressions
    static_assert(util::is_expr_v<left_t> &&
                  util::is_expr_v<right_t>);
    
    // restrict shape combinations
    static_assert(
        util::is_scl_v<left_t> ||
        util::is_scl_v<right_t> ||
        (util::is_vec_v<left_t> && util::is_vec_v<right_t>) ||
        (util::is_mat_v<left_t> && util::is_mat_v<right_t>)
            );

public:
    using value_view_t = ValueView<common_value_t, max_shape_t>;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::bind;

    BinaryNode(const left_t& expr_lhs, 
               const right_t& expr_rhs)
        : value_view_t(nullptr, 
                       std::max(expr_lhs.rows(), expr_rhs.rows()),
                       std::max(expr_lhs.cols(), expr_rhs.cols()))
        , expr_lhs_(expr_lhs)
        , expr_rhs_(expr_rhs)
    {
        // assert that the two have same dimensions if both multi-dimensional
        if constexpr (!util::is_scl_v<left_t> &&
                      !util::is_scl_v<right_t>) {
            assert(expr_lhs_.rows() == expr_rhs_.rows());
            assert(expr_lhs_.cols() == expr_rhs_.cols());
        }
    }

    /**
     * Forward evaluation first evaluates both expressions,
     * computes Binary value on the two values,
     * caches the result, and returns a const& of the cache.
     *
     * @return  const reference of forward evaluation value
     */
    const var_t& feval()
    {
        auto&& lval = expr_lhs_.feval();
        auto&& rval = expr_rhs_.feval();

        // depending on whether left or right is a vector/matrix
        // we have to pass as an array to do component-wise operation
        if constexpr (util::is_scl_v<left_t> &&
                      util::is_scl_v<right_t>) {
            return this->get() = Binary::fmap(lval, rval);
        } else if constexpr (util::is_scl_v<left_t> &&
                             !util::is_scl_v<right_t>) {
            return this->get() = Binary::fmap(lval, rval.array()).matrix();
        } else if constexpr (!util::is_scl_v<left_t> &&
                             util::is_scl_v<right_t>) {
            return this->get() = Binary::fmap(lval.array(), rval).matrix();
        } else {
            return this->get() = Binary::fmap(lval.array(), rval.array()).matrix();
        }
    }

    /**
     * Backward evaluation computes Binary partial derivative on expression values, 
     * and multiplies the two quantities as the new seed for the respective expression
     * backward evaluation.
     *
     * seed * df(w,z)/dx -> new seed for left expression
     * seed * df(w,z)/dy -> new seed for right expression
     *
     * where f is the bivariate function, w and z are the left and right expression values, respectively.
     * It is assumed that feval is called before beval.
     */
    void beval(value_t seed, size_t i, size_t j)
    {
        auto rhs_seed = seed * Binary::brmap(expr_lhs_.get(i,j), expr_rhs_.get(i,j));
        auto lhs_seed = seed * Binary::blmap(expr_lhs_.get(i,j), expr_rhs_.get(i,j));
        expr_rhs_.beval(rhs_seed, i, j);
        expr_lhs_.beval(lhs_seed, i, j);
    }

    /**
     * Binds left expression, then right expression, then itself.
     * Ignores a child expression if it is a VarView.
     *
     * @return  next pointer not bound by left, right, or itself.
     */
    value_t* bind(value_t* begin) 
    {
        value_t* next = begin;
        if constexpr (!util::is_var_view_v<left_t>) {
            next = expr_lhs_.bind(next);
        }
        if constexpr (!util::is_var_view_v<right_t>) {
            next = expr_rhs_.bind(next);
        }
        return value_view_t::bind(next);
    }

private:

    left_t expr_lhs_;
    right_t expr_rhs_;
};

} // namespace core
} // namespace ad
