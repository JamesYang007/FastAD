#pragma once
#include <type_traits>
#include "exgen.hpp"
#include "eval.hpp"
#include "vec.hpp"
#include "utility.hpp"

namespace ad {
namespace details {

template <size_t I=0, class RowIter, class ValueType, class... ExprTypes>
inline void jacobian_unpack(RowIter it, Vec<ValueType>& x
                     , std::tuple<ExprTypes...>& tup)
{
    if constexpr (I == sizeof...(ExprTypes)) {
        static_cast<void>(it);
        static_cast<void>(x);
        static_cast<void>(tup);
        return;
    }
    else {
        x.reset_adjoint(); // reset adjoint to 0
        autodiff(std::get<I>(tup));
        for (const auto& var : x) {
            *it = var.get_adjoint();
            ++it;
        }
        jacobian_unpack<I+1>(it, x, tup);
    }
} 

} // namespace details

// This function computes the jacobian of f (scalar or vector-valued function)
// given by expression generator exgen and stores the partial derivatives using iterator it.
// If f is vector-valued, it is assumed that "it" iterates by rows.
// Users can pass in optimization placeholder sizes as described in exgen. 
// @param   begin   begin iterator of underlying data for x-values
// @param   end     end iterator of underlying data for x-values
// @param   it      matrix iterator that iterates by rows (in increasing column).
// @param   f       lambda function to get jacobian of
template <size_t... opt_sizes, class RowIter, class Iter, class ExgenType
        , class = std::enable_if_t<core::is_exgen<std::decay_t<ExgenType>>> >
inline void jacobian(RowIter it, Iter begin, Iter end, ExgenType&& exgen)
{
    using value_t = typename std::iterator_traits<Iter>::value_type;
    // initialize x with given values
    Vec<value_t> x(begin, end);
    // create tuple of expressions
    auto&& tup = exgen.template generate<opt_sizes...>(x); 
    details::jacobian_unpack(it, x, tup);
}

// This function computes the jacobian of f (scalar or vector-valued function)
// given by lambda functions fs and stores the partial derivatives using iterator it.
// If f is vector-valued, it is assumed that "it" iterates by rows.
// Users can pass in optimization placeholder sizes as described in exgen. 
// @param   begin   begin iterator of underlying data for x-values
// @param   end     end iterator of underlying data for x-values
// @param   it      matrix iterator that iterates by rows (in increasing column).
// @param   f       lambda function to get jacobian of
template <size_t... opt_sizes, class RowIter, class Iter, class... Fs
        , class = std::enable_if_t<
            (!core::is_exgen<std::decay_t<Fs>> ||...) &&
            (utils::is_pointer_like_dereferenceable<std::decay_t<RowIter>>::value)
        >>
inline void jacobian(RowIter it, Iter begin, Iter end, Fs&&... fs)
{
    using value_t = typename std::iterator_traits<Iter>::value_type;
    auto&& exgen = make_exgen<value_t>(std::forward<Fs>(fs)...);
    jacobian<opt_sizes...>(it, begin, end, std::move(exgen));
}

template <size_t... opt_sizes, class Matrix_T, class Iter, class... Fs
        , class = std::enable_if_t<
            (!core::is_exgen<std::decay_t<Fs>> || ...)   // no Fs is an Exgen and
        >>
inline void jacobian(Mat<Matrix_T>& mat, Iter begin, Iter end, Fs&&... fs)
{
    mat.zeros(sizeof...(Fs), std::distance(begin, end));
     jacobian<opt_sizes...>(mat.begin(), begin, end, std::forward<Fs>(fs)...);
}

} // namespace ad 
