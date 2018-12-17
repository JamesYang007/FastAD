#pragma once
#include "adnode.h"
#include "admath.h"
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

// autodiff on tuple of expressions
template <size_t I = 0, class...ExprType>
inline typename std::enable_if<I == sizeof...(ExprType), void>::type
autodiff(std::tuple<ExprType...> tup) {}

template <size_t I = 0, class... ExprType>
inline typename std::enable_if<I < sizeof...(ExprType), void>::type
autodiff(std::tuple<ExprType...> tup)
{autodiff(std::get<I>(tup)); autodiff<I+1>(tup);}


} // end namespace ad
