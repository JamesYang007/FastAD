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
    ValueAdjView<typename util::expr_traits<LeftExprType>::value_t,
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
    using value_adj_view_t = ValueAdjView<common_value_t, max_shape_t>;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    BinaryNode(const left_t& expr_lhs, 
               const right_t& expr_rhs)
        : value_adj_view_t(nullptr, nullptr,
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
        auto&& lval = util::to_array(expr_lhs_.feval());
        auto&& rval = util::to_array(expr_rhs_.feval());
        util::to_array(this->get()) = util::cast_to<value_t>(Binary::fmap(lval, rval));
        return this->get();
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
     * We make a slight optimization to create no-op when we know the Binary operation is simply a comparison.
     */
    template <class T>
    void beval(const T& seed)
    {
        static_cast<void>(seed);
        if constexpr (!Binary::is_comparison) {
            auto&& a_val = util::to_array(this->get());
            auto&& a_adj = util::to_array(this->get_adj());
            auto&& a_l = util::to_array(expr_lhs_.get());
            auto&& a_r = util::to_array(expr_rhs_.get());

            a_adj = seed;
            auto&& rhs_seed = Binary::brmap(a_adj, a_l, a_r, a_val);
            auto&& lhs_seed = Binary::blmap(a_adj, a_l, a_r, a_val);
            expr_rhs_.beval(rhs_seed);
            expr_lhs_.beval(lhs_seed);
        }
    }

    /**
     * Binds left expression, then right expression, then itself.
     * If Binary operation is only comparison, bind value only.
     *
     * @return  next pointer pack not bound by left, right, or itself.
     */
    ptr_pack_t bind_cache(ptr_pack_t begin)
    {
        begin = expr_lhs_.bind_cache(begin);
        begin = expr_rhs_.bind_cache(begin);
        if constexpr (Binary::is_comparison) {
            auto adj = begin.adj;
            begin.adj = nullptr;
            begin = value_adj_view_t::bind(begin);
            begin.adj = adj;
            return begin;
        } else {
            return value_adj_view_t::bind(begin);
        }
    }

    /**
     * Recursively gets the total number of values needed by the expression.
     * Since a BinaryNode is a vectorized operation, it binds exactly
     * the same number as its size for both val and adj.
     *
     * @return  total bind size pack
     */
    util::SizePack bind_cache_size() const 
    { 
        return single_bind_cache_size() + 
                expr_lhs_.bind_cache_size() + 
                expr_rhs_.bind_cache_size();
    }

    util::SizePack single_bind_cache_size() const
    {
        if constexpr (Binary::is_comparison) {
            return {this->size(), 0};
        } else {
            return {this->size(), this->size()};
        }
    }

private:
    left_t expr_lhs_;
    right_t expr_rhs_;
};

/* 
 * Defines a binary struct with name "name".
 * Binary struct contains three static functions: fmap, blmap, brmap.
 *
 * - fmap evaluates f(x,y).
 * - blmap evaluates df/dx. 
 * - brmap evaluates df/dy
 *
 * blmap and brmap are additionally given seed and f values in case
 * reusing the values makes the operation more efficient.
 */

#define BINARY_STRUCT(name, comp, fmap_body, blmap_body, brmap_body) \
struct name \
{ \
    static constexpr bool is_comparison = comp; \
    template <class T, class U> \
	inline static auto fmap(T x, U y) \
	{ \
        fmap_body \
    } \
    template <class S, class T, class U, class F> \
	inline static auto blmap(const S& seed, \
                             const T& x, \
                             const U& y, \
                             const F& f) \
	{ \
        blmap_body \
    } \
    template <class S, class T, class U, class F> \
	inline static auto brmap(const S& seed, \
                             const T& x, \
                             const U& y, \
                             const F& f) \
	{ \
        brmap_body \
    } \
}

/* 
 * Defines function with name associated with struct_name.
 * Overload for constant for eager evaluation.
 * @tparam  Derived1    the actual type of node1 in CRTP
 * @tparam  Derived2    the actual type of node2 in CRTP
 * @tparam  value_type  the underlying data type.
 *                      By default, it is the common value_type of Derived1 and Derived2.
 * @return  Binary Node that will evaluate forward and backward direction
 *          defined by "struct_name"'s fmap, blmap, and brmap acting on node1 and node2
 */

#define ADNODE_BINARY_FUNC(name, struct_name) \
template <class Derived1 \
        , class Derived2 \
        , class = std::enable_if_t< \
            util::is_convertible_to_ad_v<Derived1> && \
            util::is_convertible_to_ad_v<Derived2> && \
            util::any_ad_v<Derived1, Derived2> >> \
inline auto name(const Derived1& node1, \
                 const Derived2& node2) \
{ \
    using expr1_t = util::convert_to_ad_t<Derived1>; \
    using expr2_t = util::convert_to_ad_t<Derived2>; \
    expr1_t expr1 = node1; \
    expr2_t expr2 = node2; \
    if constexpr (util::is_constant_v<expr1_t> && \
                  util::is_constant_v<expr2_t>) { \
        return ad::constant(struct_name::fmap( \
                    util::to_array(expr1.feval()), \
                    util::to_array(expr2.feval()) \
                )); \
\
    } else { \
        return BinaryNode<struct_name, \
                          expr1_t, \
                          expr2_t>(expr1, expr2); \
    } \
}

// Add
BINARY_STRUCT(Add, false,
        return x + y;, 
        static_cast<void>(x); 
        static_cast<void>(y); 
        static_cast<void>(f); 
        if constexpr (!util::is_eigen_v<T> && util::is_eigen_v<U>) {
            return seed.sum();
        } else {
            return seed;
        },
        static_cast<void>(x); 
        static_cast<void>(y); 
        static_cast<void>(f); 
        if constexpr (util::is_eigen_v<T> && !util::is_eigen_v<U>) {
            return seed.sum();
        } else {
            return seed;
        });

// Subtract
BINARY_STRUCT(Sub, false,
        return x - y;,
        static_cast<void>(x); 
        static_cast<void>(y); 
        static_cast<void>(f); 
        if constexpr (!util::is_eigen_v<T> && util::is_eigen_v<U>) {
            return seed.sum();
        } else {
            return seed;
        },
        static_cast<void>(x); 
        static_cast<void>(y); 
        static_cast<void>(f); 
        if constexpr (util::is_eigen_v<T> && !util::is_eigen_v<U>) {
            return -(seed.sum());
        } else {
            return -seed;
        });

// Multiply
BINARY_STRUCT(Mul, false,
        return x * y;, 
        static_cast<void>(x); 
        static_cast<void>(f); 
        if constexpr (!util::is_eigen_v<T> && util::is_eigen_v<U>) {
            return (seed * y).sum();
        } else {
            return seed * y;
        },
        static_cast<void>(y); 
        static_cast<void>(f); 
        if constexpr (util::is_eigen_v<T> && !util::is_eigen_v<U>) {
            return (seed * x).sum();
        } else {
            return seed * x;
        });

// Divide
BINARY_STRUCT(Div, false,
        return x / y;, 
        static_cast<void>(x); 
        static_cast<void>(f); 
        if constexpr (!util::is_eigen_v<T> && util::is_eigen_v<U>) {
            return (seed / y).sum();
        } else {
            return seed / y;
        },
        static_cast<void>(x);
        if constexpr (util::is_eigen_v<T> && !util::is_eigen_v<U>) {
            return (-seed * f / y).sum();
        } else {
            return -seed * f / y;
        });

/* 
 * Comparison operators
 * By convention, derivatives always return 0.
 * Backward evaluation should not be called for such binary operators.
 */

// LessThan
BINARY_STRUCT(LessThan, true,
        return x < y;,
        static_cast<void>(seed); 
        static_cast<void>(x); 
        static_cast<void>(y); 
        static_cast<void>(f); 
        return 0;,
        static_cast<void>(seed); 
        static_cast<void>(x); 
        static_cast<void>(y); 
        static_cast<void>(f); 
        return 0;);

// LessThanEq
BINARY_STRUCT(LessThanEq, true,
        return x <= y;,
        static_cast<void>(seed); 
        static_cast<void>(x); 
        static_cast<void>(y); 
        static_cast<void>(f); 
        return 0;,
        static_cast<void>(seed); 
        static_cast<void>(x); 
        static_cast<void>(y); 
        static_cast<void>(f); 
        return 0;);

// GreaterThan
BINARY_STRUCT(GreaterThan, true,
        return x > y;,
        static_cast<void>(seed); 
        static_cast<void>(x); 
        static_cast<void>(y); 
        static_cast<void>(f); 
        return 0;,
        static_cast<void>(seed); 
        static_cast<void>(x); 
        static_cast<void>(y); 
        static_cast<void>(f); 
        return 0;);

// GreaterThanEq
BINARY_STRUCT(GreaterThanEq, true,
        return x >= y;,
        static_cast<void>(seed); 
        static_cast<void>(x); 
        static_cast<void>(y); 
        static_cast<void>(f); 
        return 0;,
        static_cast<void>(seed); 
        static_cast<void>(x); 
        static_cast<void>(y); 
        static_cast<void>(f); 
        return 0;);

// Equal
BINARY_STRUCT(Equal, true,
        return x == y;,
        static_cast<void>(seed); 
        static_cast<void>(x); 
        static_cast<void>(y); 
        static_cast<void>(f); 
        return 0;,
        static_cast<void>(seed); 
        static_cast<void>(x); 
        static_cast<void>(y); 
        static_cast<void>(f); 
        return 0;);

// NotEqual
BINARY_STRUCT(NotEqual, true,
        return x != y;,
        static_cast<void>(seed); 
        static_cast<void>(x); 
        static_cast<void>(y); 
        static_cast<void>(f); 
        return 0;,
        static_cast<void>(seed); 
        static_cast<void>(x); 
        static_cast<void>(y); 
        static_cast<void>(f); 
        return 0;);

// Logical AND
// Note: only well-defined when inputs are of boolean context
BINARY_STRUCT(LogicalAnd, true,
        if constexpr (util::is_eigen_v<T> &&
                      !util::is_eigen_v<U>) return (x.min(y));
        else if constexpr (!util::is_eigen_v<T> &&
                           util::is_eigen_v<U>) return (y.min(x));
        else if constexpr (util::is_eigen_v<T> &&
                           util::is_eigen_v<U>) return (x.min(y));
        else return x && y;,
        static_cast<void>(seed); 
        static_cast<void>(x); 
        static_cast<void>(y); 
        static_cast<void>(f); 
        return 0;,
        static_cast<void>(seed); 
        static_cast<void>(x); 
        static_cast<void>(y); 
        static_cast<void>(f); 
        return 0;);

// Logical OR
// Note: only well-defined when inputs are of boolean context
BINARY_STRUCT(LogicalOr, true,
        if constexpr (util::is_eigen_v<T> &&
                      !util::is_eigen_v<U>) return (x.max(y));
        else if constexpr (!util::is_eigen_v<T> &&
                           util::is_eigen_v<U>) return (y.max(x));
        else if constexpr (util::is_eigen_v<T> &&
                           util::is_eigen_v<U>) return (x.max(y));
        else return x || y;,
        static_cast<void>(seed); 
        static_cast<void>(x); 
        static_cast<void>(y); 
        static_cast<void>(f); 
        return 0;,
        static_cast<void>(seed); 
        static_cast<void>(x); 
        static_cast<void>(y); 
        static_cast<void>(f); 
        return 0;);

// NOTE: ALL OPERATOR OVERLOADS MUST BE IN namespace core

// ad::core::operator+(ADNode)
ADNODE_BINARY_FUNC(operator+, Add)
// ad::core::operator-(ADNode)
ADNODE_BINARY_FUNC(operator-, Sub)
// ad::core::operator*(ADNode)
ADNODE_BINARY_FUNC(operator*, Mul)
// ad::core::operator/(ADNode)
ADNODE_BINARY_FUNC(operator/, Div)

// ad::core::operator<(ADNode)
ADNODE_BINARY_FUNC(operator<, LessThan)
// ad::core::operator<=(ADNode)
ADNODE_BINARY_FUNC(operator<=, LessThanEq)
// ad::core::operator>(ADNode)
ADNODE_BINARY_FUNC(operator>, GreaterThan)
// ad::core::operator>=(ADNode)
ADNODE_BINARY_FUNC(operator>=, GreaterThanEq)
// ad::core::operator==(ADNode)
ADNODE_BINARY_FUNC(operator==, Equal)
// ad::core::operator!=(ADNode)
ADNODE_BINARY_FUNC(operator!=, NotEqual)
// ad::core::operator&&(ADNode)
ADNODE_BINARY_FUNC(operator&&, LogicalAnd)
// ad::core::operator||(ADNode)
ADNODE_BINARY_FUNC(operator||, LogicalOr)

} // namespace core
} // namespace ad

#undef BINARY_STRUCT
#undef ADNODE_BINARY_FUNC

