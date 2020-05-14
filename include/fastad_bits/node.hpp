#pragma once
#include <vector>
#include <algorithm>
#include <stdexcept>
#include "dualnum.hpp"

namespace ad {
namespace core {

// ADNode Expression
// Any ADNodeExpr can be thought of as a node when graphically representing
// Nodes (Derived) can be defined with different names and template arguments
// like the Basic Nodes as well as ForEach, etc.
// They only need to implement :
// DualNum (stores value and adjoint), feval (forward eval), beval (back eval)
template <class Derived>
struct ADNodeExpr
{
    const Derived& self() const
    {
        return *static_cast<const Derived*>(this);
    }
    Derived& self()
    {
        return *static_cast<Derived*>(this);
    }
};


//====================================================================================================
// Basic Node Structures 

// Forward declaration of EqNode
template <class ValueType, class ExprType>
struct EqNode;

// LeafNode represents the mathematical variable.
// LeafNodes are precisely the leaves of the computation tree.
// LeafNode stores the variable value (value) and partial derivative (adjoint).
// @tparam ValueType    underlying data type
template <class ValueType>
struct LeafNode:
    public DualNum<ValueType>, ADNodeExpr<LeafNode<ValueType>>
{
    using data_t = DualNum<ValueType>;

    // By default, initialize value and adjoint to 0,
    // and adjoint destination points to current adjoint.
    LeafNode() 
        : data_t(0, 0), w_ptr_(&(data_t::get_value())), df_ptr_(&(data_t::get_adjoint()))
    {}
    // If value is provided, initialize current value to that value.
    LeafNode(ValueType w)
        : data_t(w, 0), w_ptr_(&(data_t::get_value())), df_ptr_(&(data_t::get_adjoint()))
    {}
    // If adjoint dest is also provided, initialize adjoint pointer to that dest.
    LeafNode(ValueType w, ValueType* df_ptr)
        : data_t(w, 0), w_ptr_(&(data_t::get_value())), df_ptr_(df_ptr)
    {}

    // (leaf = expression) returns EqNode
    //          =
    //      leaf  expr
    template <class Derived>
    inline auto operator=(const ADNodeExpr<Derived>&) const;

    // Forward evaluation updates current value with value in value dest.
    // @return  value destination value
    ValueType feval()
    {
        return data_t::set_value(get_value());
    }

    // Backwards evaluation sets the current adjoint value to seed and
    // increments the value df_ptr_ points to by seed.
    // Mathematically, seed is precisely a component of partial derivative
    // that was computed and passed down from the root of computation tree.
    // By default, seed is set to 0, which will not modify value in adjoint dest.
    void beval(ValueType seed = static_cast<ValueType>(0))
    {
        data_t::set_adjoint(seed);
        if (df_ptr_ != &data_t::get_adjoint()) {
            *df_ptr_ += data_t::get_adjoint();
        }
    }

    // This function returns the value in value dest.
    // It is crucial for ForEach::feval to work properly that the value dest be provided.
    ValueType& get_value() 
    {
        return *w_ptr_;
    }

    const ValueType& get_value() const
    {
        return *w_ptr_;
    }

    // This function sets the value in value dest as well current value.
    // It is crucial for ForEach::feval to work properly that the value dest be provided.
    // Note that calling set_value on any copy of current LeafNode will also overwrite
    // current LeafNode's value in its value dest.
    // It is best practice to create only expressions on LeafNodes, 
    // rather than calling this function.
    // @param   w   the new value to set current value and value dest to
    // @return w
    ValueType& set_value(ValueType w)
    {
        return *w_ptr_ = data_t::set_value(w);
    }

    // Change value pointer to point to the same place as ptr.
    ValueType* set_value_ptr(ValueType* ptr)
    {
        return w_ptr_ = ptr;
    }

    // This function returns the adjoint in adjoint dest (full partial derivative).
    // It is crucial for EqNode::beval to work properly that the full adjoint be provided.
    ValueType& get_adjoint() 
    {
        return *df_ptr_;
    }

    const ValueType& get_adjoint() const
    {
        return *df_ptr_;
    }

    // This function sets both current adjoint and adjoint dest to df.
    // @param   df  the new adjoint to set current adjoint and adjoint dest to
    // @return  df
    ValueType& set_adjoint(ValueType df)
    {
        return *df_ptr_ = data_t::set_adjoint(df);
    }

    // Change adjoint pointer to point to the same place as ptr.
    ValueType* set_adjoint_ptr(ValueType* ptr)
    {
        return df_ptr_ = ptr;
    }

    // Equivalent to set_adjoint(0)
    ValueType& reset_adjoint()
    {
        return set_adjoint(0);
    }

private:
    // explicitly declare some DualNum member functions private
    using data_t::get_value;
    using data_t::set_value;
    using data_t::get_adjoint;
    using data_t::set_adjoint;

    ValueType* w_ptr_;
    // The current adjoint is only a component of the total derivative value.
    // Consider the following:
    //
    // f(g(x)) : R -> R^n -> R 
    // fog(x)' = grad(f)(g(x)) * g'(x) = sum_i df/dx_i * dg_i/dx
    //
    // The current adjoint will simply be df/dx_i * dg_i/dx for some i.
    // df_ptr_ points to the unique location that will accumulate all such components.
    // A single LeafNode will be copied in different parts of the expression and hence
    // will have different storages for adjoints.
    // However, every copy will have df_ptr_ pointing to the same location.
    ValueType* df_ptr_; 
};

// UnaryNode represents a univariate function on an expression.
// All mathematical functions defined admath.hpp for example will
// simply return a UnaryNode that stores all information to compute forward and backward direction.
// @tparam  ValueType   underlying data type
// @tparam  Unary       univariate functor that stores fmap and bmap defining
//                      its corresponding function and derivative mapping
// @tparam  ExprType    type of expression to apply Unary on
template <class ValueType, class Unary, class ExprType>
struct UnaryNode:
    public DualNum<ValueType>, ADNodeExpr<UnaryNode<ValueType, Unary, ExprType>>
{
    using data_t = DualNum<ValueType>;

    UnaryNode(const ExprType& expr)
        : data_t(0, 0), expr_(expr)
    {}

    // Forward evaluation first evaluates given expression,
    // evaluates univariate functor on the result, and sets it to current value.
    ValueType feval()
    {
        return this->set_value(Unary::fmap(expr_.feval()));
    }

    // Backward evaluation sets current adjoint to seed,
    // multiplies seed with univariate function derivative on expression value,
    // and backward evaluate expression with the result as the new seed.
    //
    // seed * df/dx(w) -> new seed for expression
    //
    // where f is the univariate function, and w is the expression value.
    // It is assumed that feval is called before beval.
    void beval(ValueType seed)
    {
        this->set_adjoint(seed); 
        expr_.beval(this->get_adjoint() * Unary::bmap(expr_.get_value()));
    }

private:
    ExprType expr_;
};


// BinaryNode represents a bivariate function on two expressions.
// All mathematical functions defined admath.hpp for example will
// simply return a BinaryNode that stores all information to compute forward and backward direction.
// @tparam  ValueType       underlying data type
// @tparam  Binary          bivariate functor that stores fmap, blmap, brmap defining
//                          its corresponding function and derivative mapping w.r.t both variables
// @tparam  LeftExprType    type of left expression to apply Binary on
// @tparam  RightExprType   type of right expression to apply Binary on
template <class ValueType, class Binary, class LeftExprType, class RightExprType>
struct BinaryNode:
    public DualNum<ValueType>, 
    ADNodeExpr<BinaryNode<ValueType, Binary, LeftExprType, RightExprType>>
{
    using data_t = DualNum<ValueType>;

    BinaryNode(const LeftExprType& expr_lhs, const RightExprType & expr_rhs)
        : data_t(0, 0), expr_lhs_(expr_lhs), expr_rhs_(expr_rhs)
    {}

    // Forward evaluation first evaluates both expressions,
    // computes Binary value on the two values, and sets current value
    // to the result.
    // @return  forward evaluation value
    ValueType feval()
    {
        return this->set_value(Binary::fmap(expr_lhs_.feval(), expr_rhs_.feval()));
    }

    // Backward evaluation sets current adjoint to seed,
    // computes Binary partial derivative on expression values, 
    // and multiplies the two quantities as the new seed for the respective expression
    // backward evaluation.
    //
    // seed * df(w,z)/dx -> new seed for left expression
    // seed * df(w,z)/dy -> new seed for right expression
    //
    // where f is the bivariate function, w and z are the left and right expression values, respectively.
    // It is assumed that feval is called before beval.
    void beval(ValueType seed = static_cast<ValueType>(1))
    {
        this->set_adjoint(seed);
        expr_lhs_.beval(this->get_adjoint() * 
                Binary::blmap(expr_lhs_.get_value(), expr_rhs_.get_value()));
        expr_rhs_.beval(this->get_adjoint() * 
                Binary::brmap(expr_lhs_.get_value(), expr_rhs_.get_value()));
    }

private:
    LeftExprType expr_lhs_;
    RightExprType expr_rhs_;
};

// EqNode represents the mathematical placeholders for later substitution.
// Ex. u = sin(cos(y))
// u is the placeholder for the expression sin(cos(y)).
// EqNode allows users to optimize calculation by placing placeholders 
// for quantities that are used often.
// As a result, users can form new expressions using "u".
// It is guaranteed then that sin(cos(y)) will only be evaluated once.
// @tparam ValueType    underlying value type
// @tparam ExprType     type of expression to placehold
template <class ValueType, class ExprType>
struct EqNode:
    public DualNum<ValueType>, ADNodeExpr<EqNode<ValueType, ExprType>>
{
    using data_t = DualNum<ValueType>;

    EqNode(const LeafNode<ValueType>& leaf, const ExprType& expr)
        : data_t(0, 0), leaf_(leaf), expr_(expr)
    {}

    // Forward evaluation first evaluates expression,
    // sets the leaf (placeholder) to the same value
    // and finally sets current value to the same value.
    // @return forward evaluation on expression value
    ValueType feval()
    {
        return this->set_value(leaf_.set_value(expr_.feval()));
    }

    // Backward evaluation sets current adjoint to seed,
    // backward evaluates leaf using the seed,
    // and then backward evaluates expression using leaf's (full) adjoint.
    // We use the full adjoint because many expressions could use the same placeholder,
    // and hence current seed is only a component of the full partial derivative.
    // It is assumed that at the time of calling beval,
    // all expressions using placeholder have backward evaluated.
    void beval(ValueType seed = static_cast<ValueType>(0))
    {
        this->set_adjoint(seed);
        leaf_.beval(seed); 
        expr_.beval(leaf_.get_adjoint());
    }

private:
    LeafNode<ValueType> leaf_;
    ExprType expr_;
};


// Forward declaration of GlueNode 
template <class ValueType, class LeftExprType, class RightExprType>
struct GlueNode;

namespace details {
 
// Check if type T is of the form GlueNode or EqNode
template <class T>
struct is_glue_eq : std::false_type
{};

template <class ValueType, class LeftExprType, class RightExprType>
struct is_glue_eq<GlueNode<ValueType, LeftExprType, RightExprType>> : std::true_type
{};

template <class ValueType, class ExprType>
struct is_glue_eq<EqNode<ValueType, ExprType>> : std::true_type
{};

// Find number of EqNodes that are glued
template <class GlueType>
struct glue_size
{
    static inline constexpr size_t value = 0;
};

template <class ValueType, class LeftExprType, class ExprType>
struct glue_size<GlueNode<ValueType, LeftExprType, EqNode<ValueType, ExprType>>>
{
    static_assert(is_glue_eq<LeftExprType>::value, "Left expression is not a GlueNode.");
    static inline constexpr size_t value = 1 + glue_size<LeftExprType>::value;
};

template <class ValueType, class ExprType1>
struct glue_size<EqNode<ValueType, ExprType1>>
{
    static inline constexpr size_t value = 1;
};

} // namespace details

// GlueNode represents evaluation of multiple expressions in a certain order.
// Ex.
// u = sin(x) * cos(y); z = exp(u) + x
// We must forward evaluate the first expression first then the second.
// Conversely, we must backward evaluate the second expression first, then the first.
// GlueNode delegates evaluations in the correct order.
// GlueNode can only "glue" other GlueNodes and EqNodes.
//
// @tparam  ValueType       underlying data type
// @tparam  LeftExprType    type of left expression to evaluate 
// @tparam  RightExprType   type of right expression to evaluate 
template <class ValueType, class LeftExprType, class RightExprType>
struct GlueNode:
    public DualNum<ValueType>, 
    ADNodeExpr<GlueNode<ValueType, LeftExprType, RightExprType>>
{
    // Restrics definition to when LeftExprType and RightExprType are both
    // either EqNode or GlueNode.
    static_assert(details::is_glue_eq<LeftExprType>::value, 
            "Left expression of comma must be an EqNode or a GlueNode");
    static_assert(details::is_glue_eq<RightExprType>::value, 
            "Right expression of comma must be an EqNode or a GlueNode");

    using data_t = DualNum<ValueType>;

    GlueNode(const LeftExprType& expr_lhs, const RightExprType& expr_rhs)
        : data_t(0, 0), expr_lhs_(expr_lhs), expr_rhs_(expr_rhs)
    {}

    // Forward evaluation evaluates the left expression first,
    // then evaluates the right expression and sets current value to right expression value.
    // The choice of right expression value is purely by convention.
    // @return  right expression forward evaluation value
    ValueType feval()
    {
        expr_lhs_.feval(); 
        return this->set_value(expr_rhs_.feval());
    }

    // Backward evaluation sets the seed to current adjoint,
    // backward evaluates right expression with seed,
    // and backward evaluates left expression with seed equal to 0.
    // It is assumed that the right expression represents the function of interest to compute.
    // Then, the seed must be passed to the right side.
    // We do not seed the left expression since all seeding has been implicitly done
    // from the backward evaluation on right expression which will have updated
    // all placeholder adjoints (assuming user passed the correct order of expressions to evaluate).
    void beval(ValueType seed = static_cast<ValueType>(0))
    {
        this->set_adjoint(seed); 
        expr_rhs_.beval(seed); 
        expr_lhs_.beval();
    }

private:
    LeftExprType expr_lhs_;
    RightExprType expr_rhs_;
};


// ConstNode represents constants in a mathematical formula.
// Using raw data types such as double will not be compatible with node types
// when creating an expression, hence users should always create ConstNode
// when creating an expression with constants.
// @tparam  ValueType   underlying data type
template <class ValueType>
struct ConstNode :
    public DualNum<ValueType>, ADNodeExpr<ConstNode<ValueType>>
{
    using data_t = DualNum<ValueType>;

    ConstNode(ValueType w)
        : data_t(w, 0)
    {}

    // Forward evaluation simply returns the constant value.
    // @return  constant value
    ValueType feval()
    {
        return this->get_value();
    }

    // Backward evaluation does not do anything since
    // adjoint is always initialized to 0 and user cannot modify it.
    void beval(ValueType) 
    {}

private:
    using data_t::set_adjoint;  // prevent users from modifying adjoint
};

} // namespace core

// ad::constant(ValueType)
template <class ValueType>
inline auto constant(ValueType x)
{
    return core::ConstNode<ValueType>(x);
}

//======================================================================================
// Advanced Nodes

namespace core {

// SumNode represents a summation of an arbitrary function on many expressions.
// Ex.
// f(x1) + f(x2) + ... + f(xn)
// Mathematically, the derivative of this expression can be optimized
// since the partial derivative w.r.t. xi is simply f'(xi)
// and does not depend on other xj values.
// @tparam  ValueType   underlying data type
// @tparam  Iter        type of iterator of values.
// @tparam  Lmda        type of functor to apply on each iterated value returning an expression.
template <class ValueType, class Iter, class Lmda>
struct SumNode :
    public DualNum<ValueType>, ADNodeExpr<SumNode<ValueType, Iter, Lmda>>
{
    using data_t = DualNum<ValueType>;
    using iter_value_type = typename std::iterator_traits<Iter>::value_type;
    using vec_elem_t = std::decay_t<decltype(
        std::declval<Lmda>()(std::declval<iter_value_type>())
    )>;

    SumNode(Iter start, Iter end, Lmda f)
        : data_t(0, 0)
        , exprs_{}
    {
        exprs_.reserve(std::distance(start, end));
        std::for_each(start, end,
            [&](const auto& val) {
                exprs_.emplace_back(f(val));
            });
    }

    // Forward evaluation details are in eval function below.
    // @return forward evaluation of sum of functor on every expr.
    ValueType feval()
    {
        // re-evaluation requires current value to start at 0
        this->set_value(0.);
        std::for_each(exprs_.begin(), exprs_.end(),
            [this](auto& expr) {
                this->get_value() += expr.feval();
            });
        return this->get_value();
    }

    // Backward evaluation details are in eval function below.
    void beval(ValueType seed)
    {
        this->set_adjoint(seed);
        std::for_each(exprs_.begin(), exprs_.end(),
            [=](auto& expr) {
                expr.beval(seed);
            });
    }

private:
    std::vector<vec_elem_t> exprs_;
};

} // namespace core

// ad::sum(Iter, Iter, Lmda&&)
template <class Iter, class Lmda>
inline auto sum(Iter begin, Iter end, Lmda&& f)
{
    using node_t = std::decay_t<decltype(f(*begin))>;
    using value_t = typename node_t::value_type;
    // optimized for f that returns a constant node
    if constexpr (std::is_same_v<node_t, core::ConstNode<value_t>>) {
        value_t sum = 0.;
        std::for_each(begin, end, 
                [&](const auto& x) 
                { sum += f(x).get_value(); });
        return ad::constant(sum);
    } else {
        return core::SumNode<value_t, Iter, Lmda>(
                begin, end, std::forward<Lmda>(f));
    }
}

namespace core {

// ForEach represents collection of expressions to evaluate.
// It can be thought of as a generalization of GlueNode.
// Applies functor on every iterated values as an expression.
// @tparam  ValueType   underlying data type
// @tparam  Iter        type of iterator on any values
// @tparam  Lmda        type of functor to apply on iterated values that returns an expression
template <class ValueType, class Iter, class Lmda>
struct ForEach: 
    public DualNum<ValueType>, ADNodeExpr<ForEach<ValueType, Iter, Lmda>>
{
    using data_t = DualNum<ValueType>;
    using expr_t = decltype(std::declval<Lmda>()(*std::declval<Iter>()));

    ForEach(Iter start, Iter end, Lmda f)
        : data_t(0, 0), start_(start), end_(end), f_(f), vec_()
    {
        // Store return of functored value as a new expression in vec_.
        std::for_each(start_, end_, 
                [this](const typename std::iterator_traits<Iter>::value_type& i) {
                    this->vec_.emplace_back(this->f_(i)); 
                }
        );

        if (this->vec_.size() == 0) {
            throw std::length_error("Not enough elements.");
        }
    }

    // Forward evaluation on every functored expressions.
    // @return  last functored expression forward evaluation value
    ValueType feval()
    {
        std::for_each(vec_.begin(), vec_.end(), 
                [](expr_t& expr) {
                    expr.feval(); 
                }
        );
        return this->set_value(vec_[vec_.size() - 1].get_value());
    }

    // Backward evaluation first sets current adjoint to seed,
    // seeds the last functored expression with seed,
    // and backward evaluates every expression in reverse order with 0 seed.
    // See GlueNode::beval for reasons for this design choice.
    void beval(ValueType seed)
    {
        this->set_adjoint(seed);
        auto it = vec_.rbegin();
        it->beval(seed);
        std::for_each(std::next(it), vec_.rend(), 
                [](expr_t& expr) {
                    expr.beval(); 
                }
        );
    }

private:
    Iter start_, end_;
    Lmda f_;
    std::vector<expr_t> vec_;
};

} // namespace core

// ad::for_each(Iter, Iter, Lmda&&)
template <class Iter, class Lmda>
inline auto for_each(Iter start, Iter end, const Lmda& f)
{
    using value_t = typename std::decay_t<decltype(f(*start))>::value_type;
    return core::ForEach<value_t, Iter, Lmda>(start, end, f);
}

namespace core {

////////////////////////////////////////////////////////////////////////////////////
// Node makers
////////////////////////////////////////////////////////////////////////////////////

// Creates a UnaryNode given arguments
template <class ValueType, class Unary, class ExprType>
inline auto make_unary(const ExprType& expr)
{
    return UnaryNode<ValueType, Unary, ExprType>(expr);
}

// Creates a BinaryNode given arguments
template <class ValueType, class Binary, class LeftExprType, class RightExprType>
inline auto make_binary(const LeftExprType& expr_lhs, const RightExprType& expr_rhs)
{
    return BinaryNode<ValueType, Binary, LeftExprType, RightExprType>(expr_lhs, expr_rhs);
}

// Creates a EqNode given arguments
template <class ValueType, class ExprType>
inline auto make_eq(const LeafNode<ValueType>& leaf, const ExprType& expr)
{
    return EqNode<ValueType, ExprType>(leaf, expr);
}

// Creates a GlueNode given arguments
template <class ValueType, class LeftExprType, class RightExprType>
inline auto make_glue(const LeftExprType& expr_lhs, const RightExprType& expr_rhs)
{
    return GlueNode<ValueType, LeftExprType, RightExprType>(expr_lhs, expr_rhs);
}

// DO NOT MAKE make_leaf
// Pointers will point to garbage

////////////////////////////////////////////////////////////////////////////////////
// Operator overloads
////////////////////////////////////////////////////////////////////////////////////

// ad::core::LeafNode<ValueType>::operator=(const ADNodeExpr&)
template <class ValueType>
template <class Derived>
inline auto LeafNode<ValueType>::operator=(const ADNodeExpr<Derived>& expr) const
{
    return make_eq(*this, expr.self());
}

// ad::core::operator,(const ADNodeExpr&, const ADNodeExpr&)
template <class Derived1, class Derived2>
inline auto operator,(const ADNodeExpr<Derived1>& node1
                    , const ADNodeExpr<Derived2>& node2)
{
    using value_type = typename std::common_type<
        typename Derived1::value_type
        , typename Derived2::value_type
    >::type;
    return make_glue<value_type>(node1.self(), node2.self());
}

} // namespace core

//====================================================================================

// Aliases for users
template <class T>
using Var = core::LeafNode<T>;

} // namespace ad
