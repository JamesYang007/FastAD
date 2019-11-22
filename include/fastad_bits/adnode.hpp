#pragma once
#include "dualnum.hpp"
#include <vector>
#include <boost/iterator/counting_iterator.hpp>

namespace ad {
namespace core {

// ADNode Expression
// Any ADNodeExpr can be thought of as a node when graphically representing
// Nodes (Derived) can be defined with different names and template arguments
// like the Basic Nodes as well as ForEach, etc.
// They only need to implement :
// DualNum<T> (node data), feval (forward eval), beval (back eval)
template <class Derived>
struct ADNodeExpr
{
    inline Derived const& self() const
    {
        return *static_cast<Derived const*>(this);
    }
    inline Derived& self()
    {
        return *static_cast<Derived*>(this);
    }
};


//====================================================================================================
// Basic Node Structures (all specializing ADNode)

// Binary Node
template <class T = void, class Binary = void, class TL = void, class TR = void>
struct ADNode :
    public DualNum<T>, ADNodeExpr<ADNode<T, Binary, TL, TR>>
{
    using datatype = DualNum<T>;
    TL lhs;
    TR rhs;
    ADNode(TL const& lhs, TR const& rhs)
        : datatype(0, 0), lhs(lhs), rhs(rhs)
    {}

    inline T feval()
    {
        return (this->w = Binary::fmap(lhs.feval(), rhs.feval()));
    }
    // feval() must be called BEFORE beval()
    inline void beval(T seed = static_cast<T>(1))
    {
        this->df = seed;
        lhs.beval(this->df * Binary::blmap(lhs.w, rhs.w));
        rhs.beval(this->df * Binary::brmap(lhs.w, rhs.w));
    }
};

// GlueNode (operator,)
template <class T, class TL, class TR>
struct ADNode<T, void, TL, TR> :
    public DualNum<T>, ADNodeExpr<ADNode<T, void, TL, TR>>
{
    using datatype = DualNum<T>;
    TL lhs;
    TR rhs;
    ADNode(TL const& lhs, TR const& rhs)
        : datatype(0, 0), lhs(lhs), rhs(rhs)
    {}

    inline T feval()
    {
        lhs.feval(); return (this->w = rhs.feval());
    }
    inline void beval(T seed)
    {
        this->df = seed; rhs.beval(seed); lhs.beval();
    }
    inline void beval()
    {
        rhs.beval(); lhs.beval();
    }
};


// Equal (simply for type computation)
struct Equal
{};

// EqNode (operator=)
template <class T, class TR>
struct ADNode<T, Equal, ADNode<T>, TR> :
    public DualNum<T>, ADNodeExpr<ADNode<T, Equal, ADNode<T>, TR>>
{
    using datatype = DualNum<T>;
    ADNode<T> lhs;
    TR rhs;
    ADNode(ADNode<T> const& lhs, TR const& rhs)
        : datatype(0, 0), lhs(lhs), rhs(rhs)
    {}

    inline T feval()
    {
        return this->w = lhs.w = *lhs.w_ptr = rhs.feval();
    }
    inline void beval(T seed)
    {
        lhs.beval(seed); this->df = seed; this->beval();
    }
    // Only called when lhs variable df has been updated
    // Important to use ptr and not its local copy
    inline void beval()
    {
        rhs.beval(*lhs.df_ptr);
    }
};

// Unary Node
template <class T, class Unary, class TL>
struct ADNode<T, Unary, TL> :
    public DualNum<T>, ADNodeExpr<ADNode<T, Unary, TL>>
{
    using datatype = DualNum<T>;
    TL lhs;
    ADNode(TL const& lhs)
        : datatype(0, 0), lhs(lhs)
    {}

    inline T feval()
    {
        return (this->w = Unary::fmap(lhs.feval()));
    }
    inline void beval(T seed)
    {
        this->df = seed; lhs.beval(this->df * Unary::bmap(lhs.w));
    }
};

// Leaf node
template <class T>
struct ADNode<T> :
    public DualNum<T>, ADNodeExpr<ADNode<T>>
{
    using datatype = DualNum<T>;
    T* w_ptr;
    T* df_ptr;
    ADNode() : datatype(0, 0), w_ptr(&(this->w)), df_ptr(&(this->df))
    {}
    ADNode(T w)
        : datatype(w, 0), w_ptr(&(this->w)), df_ptr(&(this->df))
    {}
    ADNode(T w, T* w_ptr, T* df_ptr, T df)
        : datatype(w, df), w_ptr(w_ptr), df_ptr(df_ptr)
    {}
    ADNode(T w, T* df_ptr, T df = static_cast<T>(0))
        : datatype(w, df), w_ptr(&(this->w)), df_ptr(df_ptr)
    {}

    // leaf = expression returns EqNode
    //          =
    //        //   \\
    //      leaf  expr
    // Note: auto leaf = expression; is different!
    template <class Derived>
    inline auto operator=(ADNodeExpr<Derived> const& node)
        -> ADNode<T, Equal, ADNode<T>, Derived>
    {
        return ADNode<T, Equal, ADNode<T>, Derived>(*this, node.self());
    }

    inline T feval()
    {
        return (this->w = *(this->w_ptr));
    }

    // f(g(x)) : R -> R^n -> R 
    // fog(x)' = grad(f)(g(x)) * g'(x) = sum_i df_i/dx * dg/dx
    // Note: df_ptr will never be &(this->df) in practice for Leaf in tree
    // User will always provide a storage either through a Var<T> or
    // setting pointer df_ptr
    // This implies *df_ptr must be the one incremented
    inline void beval(T seed)
    {
        *df_ptr += (this->df = seed);
    }
};

// Intuitive typedefs
template <class T>
using LeafNode = ADNode<T>;
template <class T, class Unary, class TL>
using UnaryNode = ADNode<T, Unary, TL>;
template <class T, class Binary, class TL, class TR>
using BinaryNode = ADNode<T, Binary, TL, TR>;
template <class T, class TR>
using EqNode = ADNode<T, Equal, ADNode<T>, TR>;
template <class T, class TL, class TR>
using GlueNode = ADNode<T, void, TL, TR>;

//====================================================================================================
// Advanced Nodes

// Constant Nodes
template <class T>
struct ConstNode :
    public DualNum<T>, ADNodeExpr<ConstNode<T>>
{
    using datatype = DualNum<T>;
    ConstNode(T w)
        : datatype(w, 0)
    {}

    T feval()
    {
        return this->w;
    }

    void beval(T x) {}
};

} // end core

template <class T>
inline auto constant(T x)
{
    return core::ConstNode<T>(x);
}

namespace core {

// General sum expression
template <class T, class Iter, class Lmda>
struct SumNode :
    public DualNum<T>, ADNodeExpr<SumNode<T, Iter, Lmda>>
{
    using datatype = DualNum<T>;
    Iter start, end;
    Lmda f;

    SumNode(Iter start, Iter end, Lmda f)
        : datatype(0, 0)
        , start(start)
        , end(end)
        , f(f)
    {}

    inline T feval()
    {
        this->eval(false); return this->w;
    }

    inline void beval(T seed)
    {
        this->eval(true, seed);
    }

private:
    inline void eval(bool do_grad, T seed = static_cast<T>(0))
    {
        this->w = 0; // reset
        this->df = seed;
        std::for_each(start, end,
            [this, do_grad](typename std::iterator_traits<Iter>::value_type const& expr)
        {
            auto&& f_expr = f(expr);
            this->w += f_expr.feval();
            if (do_grad) f_expr.beval(this->df);
        }
        );
    }
};

} // namespace core

// ad::sum(Iter start, Iter end, lmda fn)
template <class Iter, class Lmda>
inline auto sum(Iter start, Iter end, Lmda&& f)
{
    return core::SumNode<
        typename decltype(f(*start))::value_type
        , Iter
        , Lmda
    >(start, end, std::forward<Lmda>(f));
}

namespace core {

// ForEach Node
// Generalization of GlueNode
// FIX: vec should be local to a eval function not as a member
template <class T
    , class Iter
    , class Lmda
>
    struct ForEach
    : public DualNum<T>, ADNodeExpr<ForEach<T, Iter, Lmda>>
{
    using datatype = DualNum<T>;
    Iter start;
    Iter end;
    Lmda f;
    std::vector<decltype(f(*start))> vec;

    ForEach(Iter start, Iter end, Lmda f)
        : datatype(0, 0)
        , start(start)
        , end(end)
        , f(f)
        , vec()
    {
        std::for_each(start, end, [this](
            typename std::iterator_traits<Iter>::value_type const& i)
        {this->vec.push_back(this->f(i)); }
        );
        if (this->vec.size() == 0) throw std::length_error("Not enough elements.");
    }

    inline T feval()
    {
        std::for_each(vec.begin(), vec.end(), [](decltype(f(*start))& expr)
        {expr.feval(); }
        );
        return this->w = vec[vec.size() - 1].w;
    }

    inline void beval(T seed)
    {
        this->df = seed;
        auto it = vec.rbegin();
        it->beval(seed);
        std::for_each(std::next(it), vec.rend(), [](decltype(f(*start))& expr)
        {expr.beval(); }
        );
    }
};

} // namespace core

// ad::for_each
template <class Iter, class Lmda>
inline auto for_each(Iter start, Iter end, Lmda const& f)
{
    return core::ForEach<
        typename decltype(f(*start))::value_type
        , Iter
        , Lmda>(start, end, f);
}


namespace core {

// General product expression
// No simplification like SumNode found
// Recursion -> seg fault with large dimension
template <class T, class Iter, class Lmda>
struct ProdNode :
    public DualNum<T>, ADNodeExpr<ProdNode<T, Iter, Lmda>>
{
    using datatype = DualNum<T>;
    Iter start, end;
    Lmda f;

    ProdNode(Iter start, Iter end, Lmda const& f)
        : datatype(0, 0)
        , start(start)
        , end(end)
        , f(f)
    {}

    inline T feval()
    {
        eval(false); return this->w;
    }

    inline void beval(T seed)
    {
        eval(true, seed);
    }

private:
    inline void eval(bool do_grad, T seed = static_cast<T>(0))
    {
        std::vector<ADNode<T>> vec(static_cast<size_t>(std::distance(start, end)));
        auto&& first = (vec[0] = this->f(*start));
        this->w = first.feval();
        if (std::distance(start, end) == 1 && !do_grad) return;
        else if (std::distance(start, end) == 1) {
            first.beval(seed);
            return;
        }
        auto it = start;
        auto&& foreach = ad::for_each(boost::counting_iterator<size_t>(1)
            , boost::counting_iterator<size_t>(static_cast<size_t>(std::distance(start, end)))
            , [&](size_t i)
        {return vec[i] = vec[i - 1] * (this->f)(*(++it)); });
        this->w = foreach.feval();
        if (do_grad) {
            this->df = seed;
            foreach.beval(this->df); first.beval();
        }
    }
};

} // namespace core

// ad::prod(Iter start, Iter end, lmda fn)
template <class Iter, class Lmda>
inline auto prod(Iter start, Iter end, Lmda&& f)
{
    return core::ProdNode<
        typename decltype(f(*start))::value_type
        , Iter
        , Lmda
    >(start, end, std::forward<Lmda>(f));
}


// Make functions
namespace core {

// BinaryNode (GlueNode if Op == void)
template <class T, class Op = void, class TL = void, class TR = void>
inline auto make_node(TL const& lhs, TR const& rhs)
{
    return ADNode<T, Op, TL, TR>(lhs, rhs);
}

// EqNode
template <class T, class TR>
inline auto make_node(ADNode<T> const& lhs, TR const& rhs)
{
    return ADNode<T, Equal, ADNode<T>, TR>(lhs, rhs);
}

// UnaryNode
template <class T, class Unary, class TL>
inline auto make_node(TL const& lhs)
{
    return ADNode<T, Unary, TL>(lhs);
}

// DO NOT MAKE make_node (LeafNode)
// Pointers will point to garbage

} // namespace core

// Operator overloads
namespace core {

// ad::core::operator,(ADNodeExpr)
// expr, expr
// Glues two expressions together by comma
template <class Derived1, class Derived2>
inline auto operator,(
    core::ADNodeExpr<Derived1> const& node1
    , core::ADNodeExpr<Derived2> const& node2)
{
    return make_node<
        typename std::common_type<typename Derived1::value_type, typename Derived2::value_type>::type
    >(node1.self(), node2.self());
}

}

//====================================================================================================
// User typedefs
template <class T>
using Var = core::ADNode<T>;


} // namespace ad
