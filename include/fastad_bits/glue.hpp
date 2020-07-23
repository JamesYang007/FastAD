#pragma once
#include <fastad_bits/expr_base.hpp>
#include <fastad_bits/type_traits.hpp>
#include <fastad_bits/value_view.hpp>

namespace ad {
namespace core {

/* 
 * GlueNode represents evaluation of multiple expressions in a certain order.
 * Ex.
 * u = sin(x) * cos(y), z = exp(u) + x
 * We must first forward evaluate the first expression then the second.
 * Conversely, we must first backward evaluate the second expression, then the first.
 * GlueNode delegates evaluations in the correct order.
 *
 * GlueNode assumes the value and shape type of the right expression.
 * It is a value viewer and it views precisely whatever the right expression views.
 *
 * @tparam  LeftExprType    type of left expression to evaluate 
 * @tparam  RightExprType   type of right expression to evaluate 
 */

template <class LeftExprType, class RightExprType>
struct GlueNode:
    ValueView<typename util::expr_traits<RightExprType>::value_t,
              typename util::shape_traits<RightExprType>::shape_t>,
    ExprBase<GlueNode<LeftExprType, RightExprType>>
{
private:
    using left_t = LeftExprType;
    using right_t = RightExprType;
    using right_value_t = typename 
        util::expr_traits<right_t>::value_t;
    using right_shape_t = typename 
        util::expr_traits<right_t>::shape_t;

    // both expressions must be AD expressions
    static_assert(util::is_expr_v<left_t> &&
                  util::is_expr_v<right_t>);

public:
    using value_view_t = ValueView<right_value_t, right_shape_t>;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::bind;

    GlueNode(const left_t& expr_lhs, 
             const right_t& expr_rhs)
        : value_view_t(nullptr,
                       expr_rhs.rows(),
                       expr_rhs.cols())
        , expr_lhs_(expr_lhs)
        , expr_rhs_(expr_rhs)
    {}

    /** 
     * Forward evaluates the left expression first,
     * then the right expression and returns the cached result.
     * Note that by this point, bind has already been called,
     * so the current GlueNode is viewing the same values as right expression.
     *
     * @return  right expression forward evaluation result
     */
    const var_t& feval()
    {
        expr_lhs_.feval(); 
        return expr_rhs_.feval();
    }

    /**
     * Backward evaluates the i,jth right expression with seed,
     * and then evaluates left expression with seed equal to 0.
     * We do not seed the left expression since all seeding has been implicitly done
     * from the backward evaluation on right expression, which will have updated
     * all placeholder adjoints (assuming user passed the correct order of expressions to evaluate).
     */
    void beval(value_t seed, size_t i, size_t j)
    {
        expr_rhs_.beval(seed, i, j); 
        expr_lhs_.beval(0, i, j);
    }

    /**
     * Binds left, then right expression, and binds itself
     * to whatever the right expression root is bound to.
     *
     * @return  the next pointer not bound by left or right expressions
     */
    value_t* bind(value_t* begin)
    {
        value_t* next = expr_lhs_.bind(begin);
        next = expr_rhs_.bind(next);
        value_view_t::bind(expr_rhs_.data());
        return next;
    }

private:
    left_t expr_lhs_;
    right_t expr_rhs_;
};

// operator, overload to create GlueNode
template <class Derived1, class Derived2>
inline auto operator,(const ExprBase<Derived1>& node1, 
                      const ExprBase<Derived2>& node2)
{
    using convert_to_view1_t = util::convert_to_view_t<Derived1>;
    using convert_to_view2_t = util::convert_to_view_t<Derived2>;
    return GlueNode<convert_to_view1_t, 
                    convert_to_view2_t>(
            node1.self(), node2.self() );
}

} // namespace core
} // namespace ad
