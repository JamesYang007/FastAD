#pragma once
#include <thread>                       // hardware_concurrency
#include <boost/asio/thread_pool.hpp>   // thread_pool
#include <boost/asio.hpp>               // post

namespace ad {

// Minimum number of expressions in tuple to evaluate to use thread pool.
// See autodiff(std::tuple) for more details.
static constexpr size_t THR_THRESHOLD = 10;

// Evaluates expression in the forward direction of reverse-mode AD.
// @tparam ExprType expression type
// @param expr  expression to forward evaluate
// @return the expression value
template <class ExprType>
inline auto evaluate(ExprType&& expr)
{
    return expr.feval();
}

// Evaluates expression in the backward direction of reverse-mode AD.
// @tparam ExprType expression type
// @param expr  expression to backward evaluate
template <class ExprType>
inline void evaluate_adj(ExprType&& expr)
{
    expr.beval(1);
}

// Evaluates expression both in the forward and backward direction of reverse-mode AD.
// @tparam ExprType expression type
// @param expr  expression to forward and backward evaluate
// Returns the forward expression value
template <class ExprType>
inline auto autodiff(ExprType&& expr)
{
    auto t = evaluate(expr);
    evaluate_adj(expr);
    return t;
}

//====================================================================================================

namespace eval {
namespace details {

///////////////////////////////////////////////////////
// Sequential autodiff
///////////////////////////////////////////////////////

// This function is the ending condition when number of expressions is equal to I.
// @tparam I    index of first expression to auto-differeniate
// @tparam ExprTypes expression types
template <size_t I, class... ExprTypes>
inline typename std::enable_if<I == sizeof...(ExprTypes)>::type
autodiff(std::tuple<ExprTypes...>&) 
{}

// This function calls ad::autodiff from the Ith expression to the last expression in tup.
// @tparam I    index of first expression to auto-differeniate
// @tparam ExprTypes    expression types
// @param tup   the tuple of expressions to auto-differentiate
template <size_t I, class... ExprTypes>
inline typename std::enable_if < I < sizeof...(ExprTypes)>::type
autodiff(std::tuple<ExprTypes...>& tup)
{
    ad::autodiff(std::get<I>(tup)); 
    autodiff<I + 1>(tup);
}

///////////////////////////////////////////////////////
// Multi-threaded autodiff
///////////////////////////////////////////////////////

// This function is the ending condition when there are no expressions to auto-differeniate.
template <size_t I, class... ExprTypes>
inline typename std::enable_if<(I == sizeof...(ExprTypes))>::type 
autodiff(boost::asio::thread_pool&, std::tuple<ExprTypes...>&)
{}

// This function auto-differentiates from the Ith expression by posting as jobs to pool.
// @tparam ExprTypes    rest of the expression types
// @tparam I    index of first expression to auto-differeniate
// @param   pool    thread pool in which to post auto-differentiating job
// @param   tup tuple of expressions to auto-differentiate
template <size_t I, class... ExprTypes>
inline typename std::enable_if<(I < sizeof...(ExprTypes))>::type
autodiff(boost::asio::thread_pool& pool, std::tuple<ExprTypes...>& tup)
{
    boost::asio::post(pool, [&](){
            ad::autodiff(std::get<I>(tup));
            });
    autodiff<I+1>(pool, tup);
}

} // namespace details 

///////////////////////////////////////////////////////
// Sequential/Multi-threaded Chooser 
///////////////////////////////////////////////////////

// This function initializes a thread pool with the hardware max number of threads
// and auto-differentiates every expression in tup.
// It is a blocking call since it waits until pool finishes executing all jobs.
// @tparam  ExprTypes   expression types
// @param   tup tuple of expressions to auto-differentiate
template <class... ExprTypes>
inline void autodiff(std::tuple<ExprTypes...>& tup, std::true_type)
{
    const int thread_num = std::thread::hardware_concurrency();
    boost::asio::thread_pool pool(thread_num);
    details::autodiff<0>(pool, tup);
    pool.join();
}

// This function auto-differeniates every expression in tup.
// @tparam  ExprTypes   expression types
// @param   tup tuple of expressions to auto-differentiate
template <class... ExprTypes>
inline void autodiff(std::tuple<ExprTypes...>& tup, std::false_type)
{
    details::autodiff<0>(tup);
}

} // namespace eval

// Auto-differentiator for lvalue reference of tuple of expressions
// @tparam  ExprTypes   expression types
// @param   tup tuple of expressions to auto-differentiate
template <class... ExprTypes>
inline void autodiff(std::tuple<ExprTypes...>& tup)
{
    eval::autodiff(tup
        , std::integral_constant<bool, (sizeof...(ExprTypes) >= THR_THRESHOLD)>());
}

// Auto-differentiator for rvalue reference of tuple of expressions
// @tparam  ExprTypes   expression types
// @param   tup tuple of expressions to auto-differentiate
template <class... ExprTypes>
inline void autodiff(std::tuple<ExprTypes...>&& tup)
{
    eval::autodiff(tup
        , std::integral_constant<bool, (sizeof...(ExprTypes) >= THR_THRESHOLD)>());
}

} // end namespace ad
