#pragma once
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_adj_view.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/util/size_pack.hpp>

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
    ValueAdjView<typename util::expr_traits<RightExprType>::value_t,
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
    using value_adj_view_t = ValueAdjView<right_value_t, right_shape_t>;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    GlueNode(const left_t& expr_lhs, 
             const right_t& expr_rhs)
        : value_adj_view_t(nullptr, nullptr,
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
        return this->get() = expr_rhs_.feval();
    }

    /**
     * Backward evaluates the i,jth right expression with seed,
     * and then evaluates left expression with seed equal to 0.
     *
     * We do not seed the left expression since all seeding has been implicitly done
     * from the backward evaluation on right expression, which will have updated
     * all placeholder adjoints (assuming user passed the correct order of expressions to evaluate).
     */
    template <class T>
    void beval(const T& seed)
    {
        expr_rhs_.beval(seed); 
        expr_lhs_.beval(0);
    }

    /**
     * Binds left, then right expression, and binds itself
     * to whatever the right expression root is bound to.
     *
     * @return  the next pointer pack not bound by left or right expressions
     */
    ptr_pack_t bind_cache(ptr_pack_t begin)
    {
        begin = expr_lhs_.bind_cache(begin);
        begin = expr_rhs_.bind_cache(begin);
        value_adj_view_t::bind({expr_rhs_.data(), expr_rhs_.data_adj()});
        return begin;
    }

    /**
     * Recursively gets the total number of values needed by the expression.
     * Since a GlueNode simply binds to that of right expression,
     * it does not bind any extra amount.
     *
     * @return  bind size
     */
    util::SizePack bind_cache_size() const 
    { 
        return expr_lhs_.bind_cache_size() + 
                expr_rhs_.bind_cache_size();
    }

    util::SizePack single_bind_cache_size() const
    { return {0,0}; }

private:
    left_t expr_lhs_;
    right_t expr_rhs_;
};

// operator, overload to create GlueNode
template <class Derived1
        , class Derived2
        , class = std::enable_if_t<
            util::is_convertible_to_ad_v<Derived1> &&
            util::is_convertible_to_ad_v<Derived2> &&
            util::any_ad_v<Derived1, Derived2>
        >>
inline auto operator,(const Derived1& node1, 
                      const Derived2& node2)
{
    using expr1_t = util::convert_to_ad_t<Derived1>;
    using expr2_t = util::convert_to_ad_t<Derived2>;
    expr1_t expr1 = node1;
    expr2_t expr2 = node2;
    return GlueNode<expr1_t, expr2_t>(expr1, expr2);
}

} // namespace core
} // namespace ad
