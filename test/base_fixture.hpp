#pragma once

namespace ad {

// Represents f(x) = 2*x
template <class T>
struct MockUnary
{
    static T fmap(T x)
    {
        return 2 * x;
    }

    static T bmap(T x) 
    {
        return 2; 
    }
};

// Represents f(x, y) = x + y
template <class T>
struct MockBinary
{
    static T fmap(T x, T y)
    {
        return x + y;
    }

    static T blmap(T x, T y) 
    {
        return 1; 
    }

    static T brmap(T x, T y)
    {
        return 1;
    }
};

// Identity expression
template <class T>
struct MockExpr
{
    using value_type = T;

    MockExpr(T w)
        : w(w), df(0), df_ptr(&df) 
    {}

    MockExpr(T w, T* df_ptr)
        : w(w), df(0), df_ptr(df_ptr) 
    {}

    T feval() 
    {
        return w;
    }

    void beval(T seed = static_cast<T>(0))
    {  
        df = seed;
        if (df_ptr != &df) {
            *df_ptr += df;
        }
    }

    T get_value() const
    {
        return w;
    }

    void set_value(T x)
    {
        w = x;
    }

    T get_adjoint() const
    {
        return *df_ptr;
    }

    T get_curr_adjoint() const
    {
        return df;
    }

private:
    T w;
    T df;
    T* df_ptr;
};

} // namespace ad
