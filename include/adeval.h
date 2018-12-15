#pragma once
#include "adnode.h"
#include "advec.h"
#include <armadillo>

namespace ad {

// Glue then evaluate
// Forward propagation
template <class ExprType>
inline void Evaluate(ExprType&& expr)
{expr.feval();}

// Backward propagation
template <class ExprType>
inline void EvaluateAdj(ExprType&& expr)
{expr.beval(1);}

// Both forward and backward
template <class ExprType>
inline void autodiff(ExprType&& expr)
{expr.feval(); expr.beval(1);}

//====================================================================================================

// If f is a function and T numeric
template <class F
        , class Iter
        , class T = typename std::iterator_traits<Iter>::value_type // data type
        , class = typename std::result_of<F(Vec<T>&)>::type // is callable
         >
inline typename std::enable_if<
    std::is_arithmetic<T>::value
    , arma::Mat<T>
    >::type
autodiff(F f, Iter begin, Iter end)
{
    // Length of x vector
    auto n = std::distance(begin, end);
    // Initialize ADVector with 0 variables with capacity n
    Vec<T> vec(0, n);
    // Emplace variables with given x values
    std::for_each(begin, end, [&vec](T x){vec.emplace_back(x);});
    // Auto differentiate
    auto&& expr = f(vec);
    autodiff(expr);
    // Resulting gradient as row vector
    arma::Mat<T> res(1, n);
    auto it = vec.begin();
    res.for_each([&it](T& df) mutable {df = (it++)->df;});

    return res;
}

// If function f is a tuple of functions and T numeric
// Base case: tuple of size 1
template <
        size_t I = 0
        , class ...Ts
        , class Iter
        , class T = typename std::iterator_traits<Iter>::value_type
         >
inline typename std::enable_if<
    std::is_arithmetic<T>::value
    && (I == sizeof...(Ts) - 1)
    , arma::Mat<T>
    >::type
autodiff(std::tuple<Ts...> const& f, Iter begin, Iter end)
{return autodiff(std::get<I>(f), begin, end);}

// General case: tuple of size (sizeof...(Ts))
template <
        size_t I = 0
        , class ...Ts
        , class Iter
        , class T = typename std::iterator_traits<Iter>::value_type
         >
inline typename std::enable_if<
    std::is_arithmetic<T>::value
    && (I < sizeof...(Ts) - 1)
    , arma::Mat<T>
    >::type
autodiff(std::tuple<Ts...> const& f, Iter begin, Iter end)
{
    auto&& grad = autodiff(std::get<I>(f), begin, end);
    auto&& rest = autodiff<I+1>(f, begin, end);
    rest.insert_rows(0, grad);
    return rest;
}

//====================================================================================================

} // end namespace ad
