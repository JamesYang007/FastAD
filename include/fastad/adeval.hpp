#pragma once
#include <thread>
#include <boost/asio/thread_pool.hpp>
#include <boost/asio.hpp>
#include "adnode.hpp"

namespace ad {

static constexpr size_t THR_THRESHOLD = 10;

// Glue then evaluate
// Forward propagation
template <class ExprType>
inline auto Evaluate(ExprType&& expr)
{
    return expr.feval();
}

// Backward propagation
template <class ExprType>
inline void EvaluateAdj(ExprType&& expr)
{
    expr.beval(1);
}

// Both forward and backward
// This is primarily for details_tuple namespace usage
// std::thread argument needs any overloaded function to be specified
// workaround: create a non-overloaded fn and autodiff will call this
template <class ExprType>
inline auto EvaluateBoth(ExprType&& expr)
{
    auto t = Evaluate(expr);
    EvaluateAdj(expr);
    return t;
}

template <class ExprType>
inline auto autodiff(ExprType&& expr)
{
    return EvaluateBoth(expr);
}

//====================================================================================================

// autodiff on tuple of expressions
namespace details_tuple {

// No multi-threading
// Lvalue
template <size_t I, class...ExprType>
inline typename std::enable_if<I == sizeof...(ExprType), void>::type
    autodiff_(std::tuple<ExprType...>& tup) {}

template <size_t I, class... ExprType>
inline typename std::enable_if < I < sizeof...(ExprType), void>::type
    autodiff_(std::tuple<ExprType...>& tup)
{
    ad::autodiff(std::get<I>(tup)); autodiff_<I + 1>(tup);
}

// multi-threading using boost thread_pool
template <class ExprType>
inline void autodiff_(boost::asio::thread_pool& pool, ExprType& expr)
{
    boost::asio::post(pool, [&](){
            ad::EvaluateBoth(expr);
            });
}

template <class ExprType1, class...ExprType>
inline void autodiff_(boost::asio::thread_pool& pool, ExprType1& expr1, ExprType&... expr)
{
    boost::asio::post(pool, [&](){
            ad::EvaluateBoth(expr1);
            });
    autodiff_(pool, expr...);
}

template <class...ExprType, size_t...I>
inline void autodiff_(std::tuple<ExprType...>& tup, std::index_sequence<I...>)
{
    const int thread_num = std::thread::hardware_concurrency();
    boost::asio::thread_pool pool(thread_num);
    autodiff_(pool, std::get<I>(tup)...);
    pool.join();
}

// use multi-threading to compute autodiff on tuple of expressions
template <class...ExprType>
inline void autodiff_(std::tuple<ExprType...>& tup, std::true_type)
{
    autodiff_(tup, std::make_index_sequence<sizeof...(ExprType)>());
}

// false_type
template <class...ExprType>
inline void autodiff_(std::tuple<ExprType...>& tup, std::false_type)
{
    autodiff_<0>(tup);
}

} // namespace details_tuple

// USER-CALLABLE autodiff 
template <class...ExprType>
inline void autodiff(std::tuple<ExprType...>& tup)
{
    details_tuple::autodiff_(tup
        , std::integral_constant<bool, (sizeof...(ExprType) >= THR_THRESHOLD)>());
}

template <class...ExprType>
inline void autodiff(std::tuple<ExprType...>&& tup)
{
    details_tuple::autodiff_(tup
        , std::integral_constant<bool, (sizeof...(ExprType) >= THR_THRESHOLD)>());
}

} // end namespace ad
