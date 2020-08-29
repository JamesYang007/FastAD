#pragma once
#include <Eigen/Core>
#include <type_traits>
#include <iterator>
#include <tuple>
#include <fastad_bits/util/shape_traits.hpp>

namespace ad {

// forward declaration (namespace matters)
template <class ValueType, class ShapeType>
struct VarView;
template <class ValueType, class ShapeType>
struct Var;

namespace core {

template <class T>
struct ExprBase;

template <class Derived>
struct ConstantBase;

template <class V, class S>
struct Constant;

} // namespace core

namespace util {

/* 
 * Define traits for AD expressions
 */
template <class T>
struct expr_traits
{
    using value_t = typename T::value_t;
    using shape_t = typename T::shape_t;
    using var_t = typename T::var_t;
    using ptr_pack_t = typename T::ptr_pack_t;
};

/*
 * Check if T is an AD expression
 */
template <class T>
inline constexpr bool is_expr_v =
    std::is_base_of_v<core::ExprBase<T>, T>;


/*
 * Check if type T is Eigen type
 */
template <class T>
inline constexpr bool is_eigen_v =
    std::is_base_of_v<Eigen::EigenBase<T>, T>;

/*
 * Check if T is Eigen (column) vector shape
 * Should only be used when T is known to be an Eigen type.
 */
template <class T>
inline constexpr bool is_eigen_vector_v =
    is_eigen_v<T> &&
    T::ColsAtCompileTime == 1
    ;

/*
 * Check if T is Eigen Matrix shape
 * Should only be used when T is known to be an Eigen type.
 */
template <class T>
inline constexpr bool is_eigen_matrix_v =
    is_eigen_v<T> &&
    T::RowsAtCompileTime == Eigen::Dynamic &&
    T::ColsAtCompileTime == Eigen::Dynamic
    ;

/*
 * Check if type T is VarView
 */
namespace details {

template <class T>
struct is_var_view : std::false_type
{};

template <class ValueType
        , class ShapeType>
struct is_var_view<VarView<ValueType, ShapeType>>:
    std::true_type
{};

} // namespace details

template <class T>
inline constexpr bool is_var_view_v =
    details::is_var_view<T>::value;

/*
 * Check if type T is Var
 */
namespace details {

template <class T>
struct is_var : std::false_type
{};

template <class ValueType
        , class ShapeType>
struct is_var<Var<ValueType, ShapeType>>:
    std::true_type
{};

} // namespace details

template <class T>
inline constexpr bool is_var_v =
    details::is_var<T>::value;

/*
 * Check if type T is Constant
 */
template <class T>
inline constexpr bool is_constant_v =
    std::is_base_of_v<core::ConstantBase<T>, T>;

/**
 * Constant represents constants in a mathematical formula.
 * It owns the constant values rather than viewing them elsewhere.
 *
 * @tparam  ValueType   underlying data type
 */
namespace details {

template <class ValueType, class ShapeType>
struct constant_var; 

template <class ValueType>
struct constant_var<ValueType, ad::scl>
{
    using type = ValueType;
};

template <class ValueType>
struct constant_var<ValueType, ad::vec>
{
    using type = Eigen::Matrix<ValueType, Eigen::Dynamic, 1>;
};

template <class ValueType>
struct constant_var<ValueType, ad::mat>
{
    using type = Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic>;
};

} // namespace details

template <class ValueType, class ShapeType>
using constant_var_t = typename 
    details::constant_var<ValueType, ShapeType>::type;


/*
 * Get common value type among variadic types Ts...
 * where Ts are AD expressions.
 */
namespace details {

template <class T, class... Ts>
struct common_value
{
    using type = std::common_type_t<
        typename util::expr_traits<T>::value_t,
        typename common_value<Ts...>::type
            >;
};

template <class T>
struct common_value<T>
{
    using type = typename util::expr_traits<T>::value_t;
};

} // namespace details

template <class... Ts>
using common_value_t = typename
    details::common_value<Ts...>::type;

/*
 * Convert T to correct corresponding AD expression.
 * Note that some Eigen specializations require code duplication -
 * otherwise SFINAE won't work but will turn into hard errors.
 */
namespace details {

template <class T, class = void>
struct convert_to_ad;

template <class T>
struct convert_to_ad<T, std::enable_if_t<util::is_expr_v<T>>>
{
    using type = T;
};

// specialization: var
template <class T>
struct convert_to_ad<T, std::enable_if_t<is_var_v<T>> >
{
    using type = ad::VarView<
        typename util::expr_traits<T>::value_t,
        typename util::expr_traits<T>::shape_t >;
};

// specialization: arithmetic 
template <class T>
struct convert_to_ad<T, std::enable_if_t<std::is_arithmetic_v<T>>>
{
    using type = core::Constant<T, ad::scl>;
};

// specialization: column vector
template <class T>
struct convert_to_ad<T, std::enable_if_t<
    is_eigen_v<T> &&
    T::ColsAtCompileTime == 1 > >
{
    using type = core::Constant<typename T::Scalar, ad::vec>;
};

// specialization: matrix 
template <class T>
struct convert_to_ad<T, std::enable_if_t<
    is_eigen_v<T> &&
    T::RowsAtCompileTime == Eigen::Dynamic &&
    T::ColsAtCompileTime == Eigen::Dynamic> >
{
    using type = core::Constant<typename T::Scalar, ad::mat>;
};

} // namespace details

template <class T>
using convert_to_ad_t = typename
    details::convert_to_ad<T>::type;

/*
 * Checks if T can be converted to AD expression as specified by
 * convert_to_ad_t
 */
namespace details {

template <class T, class=std::void_t<>>
struct is_convertible_to_ad:
    std::false_type
{};

template <class T>
struct is_convertible_to_ad<T, std::void_t<convert_to_ad_t<T>>>:
    std::true_type
{};

} // namespace details

template <class T>
inline constexpr bool is_convertible_to_ad_v =
    details::is_convertible_to_ad<T>::value;

/*
 * Checks at least one of the types in Ts is an AD-like object.
 * We say AD-like since we consider Var types as true.
 * While Var itself is not an AD expression, its base class is.
 */
namespace details {

template <class... Ts>
struct any_ad : std::false_type
{};

template <class T, class... Ts>
struct any_ad<T, Ts...>
{
    static constexpr bool value = 
        (util::is_expr_v<T> || util::is_var_v<T>) ||
        any_ad<Ts...>::value;
};

} // namespace details

template <class... Ts>
inline constexpr bool any_ad_v = 
    details::any_ad<Ts...>::value;

} // namespace util
} // namespace ad
