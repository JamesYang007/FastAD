#pragma once
#include <tuple>
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_view.hpp>
#include <fastad_bits/reverse/core/var_view.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/util/numeric.hpp>
#include <Eigen/Dense>

namespace ad {
namespace core {

/**
 * UniformAdjLogPDFNode represents the uniform log pdf 
 * adjusted to omit all fixed constants, i.e. omits -n/2*log(2*pi).
 *
 * It assumes the value type that is common to all three expressions.
 * Since it represents a log-pdf, it is always a scalar expression.
 *
 * The only possible shape combinations are as follows:
 * x -> scalar, mean -> scalar, sigma -> scalar
 * x -> vec, mean -> scalar | vector, sigma -> scalar | vector | self adjoint matrix
 *
 * No other shapes are permitted for this node.
 *
 * At construction, the actual sizes of the three expressions are checked -
 * specifically if x is a vector, and mean and sigma are not scalar,
 * then size of x must be the same as that of mean rows and sigma rows.
 * Additionally, we check that sigma is square if it is a matrix.
 *
 * @tparam  XExprType           type of x expression at which to evaluate log-pdf
 * @tparam  MeanExprType        type of mean expression
 * @tparam  SigmaExprType       type of sigma expression
 */
template <class XExprType
        , class MinExprType
        , class MaxExprType
        , class = std::tuple<
            typename util::shape_traits<XExprType>::shape_t,
            typename util::shape_traits<MinExprType>::shape_t,
            typename util::shape_traits<MaxExprType>::shape_t> >
struct UniformAdjLogPDFNode;

} // namespace core
} // namespace ad
