#pragma once
#include <Eigen/Core>
#include <type_traits>
#include <iterator>
#include <tuple>

namespace ad {

// forward declaration (namespace matters)
template <class ValueType, class ShapeType>
struct VarView;
template <class ValueType, class ShapeType>
struct Var;

namespace util {

// forward declaration
template <class T>
struct expr_traits;

/*
 * Check if type T is Eigen type
 */
template <class T>
inline constexpr bool is_eigen_matrix_v =
    std::is_base_of_v<Eigen::EigenBase<T>, T>;

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

} // namespace util
} // namespace ad
