#pragma once
#include <cmath>
#include <unsupported/Eigen/SpecialFunctions>       // needed for erf
#include <fastad_bits/forward/core/forward.hpp>    
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/reverse/core/value_adj_view.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/util/shape_traits.hpp>
#include <fastad_bits/util/size_pack.hpp>
#include <fastad_bits/util/value.hpp>

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
    ValueAdjView<typename util::expr_traits<ExprType>::value_t,
                 typename util::shape_traits<ExprType>::shape_t>,
    ExprBase<UnaryNode<Unary, ExprType>>
{
private:
    using expr_t = ExprType;
    static_assert(util::is_expr_v<expr_t>);

public:
    using value_adj_view_t = ValueAdjView<
        typename util::expr_traits<expr_t>::value_t, 
        typename util::shape_traits<expr_t>::shape_t >;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    UnaryNode(const expr_t& expr)
        : value_adj_view_t(nullptr, nullptr, expr.rows(), expr.cols())
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
        auto&& a_expr = util::to_array(expr_.feval());
        util::to_array(this->get()) = Unary::fmap(a_expr);
        return this->get();
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
     * This is true for all shapes so long as both arguments are arrays or scalars.
     */
    template <class T>
    void beval(const T& seed)
    {
        auto&& a_val = util::to_array(this->get());
        auto&& a_adj = util::to_array(this->get_adj());
        auto&& a_expr = util::to_array(expr_.get());
        a_adj = seed;
        expr_.beval(Unary::bmap(a_adj, a_expr, a_val));
    }

    /**
     * First binds for underlying expression then binds itself.
     * @return  next pointer pack not bound by underlying expression and itself.
     */
    ptr_pack_t bind_cache(ptr_pack_t begin)
    { 
        begin = expr_.bind_cache(begin);
        return value_adj_view_t::bind(begin);
    }

    /**
     * Recursively gets the total number of values needed by the expression.
     * Since a UnaryNode is a vectorized operation, and does not require any
     * extra temporaries, it binds exactly the same number as its size in both val and adj.
     * @return  size pack
     */
    util::SizePack bind_cache_size() const 
    { 
        return single_bind_cache_size() + 
                expr_.bind_cache_size();
    }

    util::SizePack single_bind_cache_size() const
    {
        return {this->size(), this->size()};
    }

private:
    expr_t expr_;
};

//////////////////////////////////////////////////////////////////////////
// Unary struct and helper function definitions
//////////////////////////////////////////////////////////////////////////

/*
 * Expose fname from namespace various namespaces for look-up.
 * forward/core/forward.hpp defines some of these overloads.
 * Rest of the overloads are in this header
 */
#define USING_STD_AD_EIGEN(fname) \
    using std::fname;\
    using ad::fname;\
    using Eigen::fname;

/* 
 * Defines a unary struct with name "name".
 * This struct acts as a functor that will be passed as a type to UnaryNode.
 *
 * fmap evaluates f(x)
 * bmap evaluates seed * df/dx 
 *
 * bmap is also given the value of f if it is more efficient to reuse its value (see Exp).
 * Both functions are kept templatized since any combination of scalar or Eigen arrays can be passed.
 */

#define UNARY_STRUCT(name, fmap_body, bmap_body) \
struct name \
{ \
    template <class T> \
	inline static auto fmap(const T& x) \
	{ \
		fmap_body \
	} \
\
    template <class S, class T, class U> \
	inline static auto bmap(const S& seed, \
                            const T& x, \
                            const U& f) \
	{ \
		bmap_body \
	} \
}

/* 
 * Defines function with name associated with struct_name.
 * Overloaded for constant nodes to be eager-evaluated.
 * @tparam  Derived     the actual type of node in CRTP
 * @return  Unary Node that will evaluate forward and backward direction 
 *          defined by "struct_name"'s fmap and bmap acting on "node"
 */

#define ADNODE_UNARY_FUNC(name, struct_name) \
    template <class Derived \
            , class = std::enable_if_t< \
                util::is_convertible_to_ad_v<Derived> && \
                util::any_ad_v<Derived> >> \
    inline auto name(const Derived& node) \
    { \
        using expr_t = util::convert_to_ad_t<Derived>; \
        expr_t expr = node; \
        if constexpr (util::is_constant_v<expr_t>) { \
            return ad::constant(core::struct_name::fmap(\
                        util::to_array(expr.feval())) ); \
        } else { \
            return core::UnaryNode<core::struct_name, expr_t>(expr); \
        } \
    }

// UnaryMinus struct
UNARY_STRUCT(UnaryMinus, 
             return -x;, 
             static_cast<void>(x); 
             static_cast<void>(f); 
             return -seed;);

// Sin struct
UNARY_STRUCT(Sin, 
             USING_STD_AD_EIGEN(sin);
             return sin(x);, 
             static_cast<void>(f); 
             USING_STD_AD_EIGEN(cos); 
             return seed * cos(x););

// Cos struct
UNARY_STRUCT(Cos, 
             USING_STD_AD_EIGEN(cos); 
             return cos(x);, 
             static_cast<void>(f); 
             return -seed * Sin::fmap(x););

// Tan struct
UNARY_STRUCT(Tan, 
             USING_STD_AD_EIGEN(tan); 
             return tan(x);, 
             static_cast<void>(f); 
             auto tmp = Cos::fmap(x); 
             return seed / (tmp * tmp););

// Arcsin struct (degrees)
UNARY_STRUCT(Arcsin, 
             USING_STD_AD_EIGEN(asin);
             return asin(x);, 
             static_cast<void>(f); 
             USING_STD_AD_EIGEN(sqrt);
             return seed / sqrt(1. - x * x););

// Arccos struct (degrees)
UNARY_STRUCT(Arccos, 
             USING_STD_AD_EIGEN(acos);
             return acos(x);, 
             static_cast<void>(f); 
             return -Arcsin::bmap(seed, x, f););

// Arctan struct (degrees)
UNARY_STRUCT(Arctan, 
             USING_STD_AD_EIGEN(atan);
             return atan(x);, 
             static_cast<void>(f); 
             return seed / (1. + x * x););

// Exp struct
UNARY_STRUCT(Exp, 
             USING_STD_AD_EIGEN(exp);
             return exp(x);, 
             static_cast<void>(x); 
             return seed * f;);

// Log struct
UNARY_STRUCT(Log, 
             USING_STD_AD_EIGEN(log);
             return log(x);, 
             static_cast<void>(f); 
             return seed / x;);

// Sqrt struct
UNARY_STRUCT(Sqrt,
             USING_STD_AD_EIGEN(sqrt);
             return sqrt(x);,
             static_cast<void>(x);
             return 0.5 * seed / f;);

// Erf struct
UNARY_STRUCT(Erf,
             USING_STD_AD_EIGEN(erf);
             return erf(x);,
             static_cast<void>(f); 
             static constexpr double two_over_sqrt_pi =
                1.1283791670955126;
             return two_over_sqrt_pi * seed * Exp::fmap(-x * x););

// sigmoid
UNARY_STRUCT(Sigmoid, 
             USING_STD_AD_EIGEN(exp);
             return 1/(1+exp(-x));, 
             static_cast<void>(x); 
             return seed * exp(-x)/((exp(-x)+1)*(exp(-x)+1)););

// sinh
UNARY_STRUCT(Sinh, 
             //USING_STD_AD_EIGEN(tanh);
             return Eigen::sinh(x);, 
             static_cast<void>(f); 
             return seed *(Eigen::cosh(x)););
// cosh
UNARY_STRUCT(Cosh, 
             //USING_STD_AD_EIGEN(tanh);
             return Eigen::cosh(x);, 
             static_cast<void>(f); 
             return seed *(Eigen::sinh(x)););			 
// tanh
UNARY_STRUCT(Tanh, 
             //USING_STD_AD_EIGEN(tanh);
             return Eigen::tanh(x);, 
             static_cast<void>(f); 
             return seed *(1-f*f););
			 
// operator- (IMPORTANT TO DECLARE IN core)
ADNODE_UNARY_FUNC(operator-, UnaryMinus)

} // namespace core

// ad::sin(ADNode)
ADNODE_UNARY_FUNC(sin, Sin)
// ad::cos(ADNode)
ADNODE_UNARY_FUNC(cos, Cos)
// ad::tan(ADNode)
ADNODE_UNARY_FUNC(tan, Tan)
// ad::asin(ADNode)
ADNODE_UNARY_FUNC(asin, Arcsin)
// ad::acos(ADNode)
ADNODE_UNARY_FUNC(acos, Arccos)
// ad::atan(ADNode)
ADNODE_UNARY_FUNC(atan, Arctan)
// ad::exp(ADNode)
ADNODE_UNARY_FUNC(exp, Exp)
// ad::log(ADNode)
ADNODE_UNARY_FUNC(log, Log)
// ad::sqrt(ADNode)
ADNODE_UNARY_FUNC(sqrt, Sqrt)
// ad::erf(ADNode)
ADNODE_UNARY_FUNC(erf, Erf)

// ad::sigmoid(ADNode)
ADNODE_UNARY_FUNC(sigmoid, Sigmoid)

// ad::tanh(ADNode)
ADNODE_UNARY_FUNC(sinh, Sinh)
ADNODE_UNARY_FUNC(cosh, Cosh)
ADNODE_UNARY_FUNC(tanh, Tanh)

} // namespace ad

#undef USING_STD_AD_EIGEN
#undef UNARY_STRUCT
#undef ADNODE_UNARY_FUNC
