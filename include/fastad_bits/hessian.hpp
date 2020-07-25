#pragma once
#include <type_traits>
#include <algorithm>
#include "eval.hpp"
#include "forward.hpp"
#include "type_traits.hpp"

namespace ad {
namespace core {

template <class Iter, class T>
Iter compute_hess(Iter it, const Vec<ForwardVar<T>>& x)
{
    std::for_each(x.cbegin(), x.cend(),
        [&it](const typename Vec<ForwardVar<T>>::value_type& xi) mutable {
            *it = xi.get_adjoint().get_adjoint();
            ++it;
        });
    return it;
}

template <size_t... opt_sizes, class HessIter, class T, class Exgen>
inline void hessian_gen(HessIter hess_begin, Vec<ForwardVar<T>>& x, Exgen&& gen)
{
    auto it = hess_begin;
    auto expr = std::get<0>(gen.template generate<opt_sizes...>(x));

    for (size_t i = 0; i < x.size(); ++i)
    {
        x.reset_adjoint();
        if (i > 0) {
            x[i-1].get_value().set_adjoint(0);  // set the (i-1)th forward variable's adjoint back to 0
        }
        x[i].get_value().set_adjoint(1); // set the ith forward variable's adjoint to 1
        autodiff(expr);
        // Record Hessian first column
        // Advance iterator to next column of matrix
        it = compute_hess(it, x);
    }
}

template <size_t... opt_sizes, class HessIter, class GradIter, class T, class Exgen>
inline void hessian_gen(HessIter hess_begin, GradIter grad_begin, Vec<ForwardVar<T>>& x, Exgen&& gen)
{
    auto it = hess_begin;
    auto expr = std::get<0>(gen.template generate<opt_sizes...>(x));

    for (size_t i = 0; i < x.size(); ++i)
    {
        x.reset_adjoint();
        if (i > 0) {
            x[i-1].get_value().set_adjoint(0);  // set the (i-1)th forward variable's adjoint back to 0
        }
        x[i].get_value().set_adjoint(1); // set the ith forward variable's adjoint to 1
        autodiff(expr);

        // Record gradient
        *grad_begin = expr.get_value().get_adjoint(); // get expression forward value's adjoint
        ++grad_begin;
        // Record Hessian first column
        // Advance iterator to next column of matrix
        it = compute_hess(it, x);
    }
}

} // namespace core

// The function will store only the hessian into object pointed by hess_begin.
// This overload will only be enabled if all of the iterators
// are dereferenceable like a pointer, i.e., valid iterators.
// @param   hess_begin  column-wise iterator of a 2d-array or matrix to store hessian
// @param   begin       iterator to first underlying data
// @param   end         iterator to one past last underlying data
// @param   f           lambda function to compute hessian and gradient of
//                      it is expected that the arguments follow the specs of exgen
//                      and in addition are declared with auto or Vec<ForwardVar<T>> for some T.
template <size_t... opt_sizes, class HessIter, class Iter, class F
    , class = std::enable_if_t<
    utils::is_pointer_like_dereferenceable<HessIter>::value &&
    utils::is_pointer_like_dereferenceable<Iter>::value
    >
>
inline void hessian(HessIter hess_begin, Iter begin, Iter end, F&& f)
{
    using T = typename std::iterator_traits<Iter>::value_type;
    auto gen = make_exgen<ForwardVar<T>>(std::forward<F>(f));
    Vec<ForwardVar<T>> x(begin, end);
    core::hessian_gen<opt_sizes...>(hess_begin, x, gen);
}

// The function will store the hessian into object pointed by hess_begin,
// and store gradient as well into object pointed by grad_begin.
// This overload will only be enabled if all of the iterators
// are dereferenceable like a pointer, i.e., valid iterators.
// @param   hess_begin  column-wise iterator of a 2d-array or matrix to store hessian
// @param   grad_begin  iterator of a array or 1-d matrix to store gradient
// @param   begin       iterator to first underlying data
// @param   end         iterator to one past last underlying data
// @param   f           lambda function to compute hessian and gradient of
//                      it is expected that the arguments follow the specs of exgen
//                      and in addition are declared with auto or Vec<ForwardVar<T>> for some T.
template <size_t... opt_sizes, class HessIter, class GradIter, class Iter, class F
    , class = std::enable_if_t<
    utils::is_pointer_like_dereferenceable<HessIter>::value &&
    utils::is_pointer_like_dereferenceable<GradIter>::value &&
    utils::is_pointer_like_dereferenceable<Iter>::value
    > >
inline void hessian(HessIter hess_begin, GradIter grad_begin, Iter begin, Iter end, F&& f)
{
    using T = typename std::iterator_traits<Iter>::value_type;
    auto gen = make_exgen<ForwardVar<T>>(std::forward<F>(f));
    Vec<ForwardVar<T>> x(begin, end);
    core::hessian_gen<opt_sizes...>(hess_begin, grad_begin, x, gen);
}

template <size_t... opt_sizes, class Matrix_T, class Iter, class F>
inline void hessian(Mat<Matrix_T>& hess_mat, Iter begin, Iter end, F&& f)
{
    const size_t n = static_cast<size_t>(std::distance(begin, end));
    hess_mat.zeros(n, n);
    hessian<opt_sizes...>(hess_mat.begin(), begin, end, std::forward<F>(f));
}

template <size_t... opt_sizes, class Matrix_T, class F, class Iter>
inline void hessian(Mat<Matrix_T>& hess_mat, Mat<Matrix_T>& grad_mat, Iter begin, Iter end, F&& f)
{
    const size_t n = static_cast<size_t>(std::distance(begin, end));
    hess_mat.zeros(n, n);
    grad_mat.zeros(1, n);
    hessian<opt_sizes...>(hess_mat.begin(), grad_mat.begin(), begin, end, std::forward<F>(f));
}

} // namespace ad
