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

/*
 * Metaprogramming tools to check if type T is Eigen matrix.
 */
template <class T>
inline constexpr bool is_eigen_matrix_v =
    std::is_base_of_v<Eigen::EigenBase<T>, T>;

/*
 * Metaprogramming tool to check if a type is VarView
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
 * Converts a variable to viewer type.
 * If not a variable, kept the same.
 */
namespace details {

template <class T>
struct convert_to_view
{
    using type = T;
};

template <class ValueType, class ShapeType>
struct convert_to_view<Var<ValueType, ShapeType>>
{
    using type = VarView<ValueType, ShapeType>;
};

} // namespace details

template <class T>
using convert_to_view_t = typename
    details::convert_to_view<T>::type;

// TODO: are these needed?
// Dummy function used to SFINAE on return value.
// Checks if type T is pointer-like (dereferenceable)
template <class T>
auto is_pointer_like()
    -> decltype(*std::declval<T>(), std::true_type{});

template <class T>
using is_pointer_like_dereferenceable = decltype(is_pointer_like<T>());

// Checks if type T is a tuple
template <class T>
struct is_tuple : std::false_type 
{};
template <class... Ts>
struct is_tuple<std::tuple<Ts...>> : std::true_type
{};
template <class... Ts>
struct is_tuple<std::tuple<Ts...>&> : std::true_type
{};
template <class... Ts>
struct is_tuple<const std::tuple<Ts...>> : std::true_type
{};
template <class... Ts>
struct is_tuple<const std::tuple<Ts...>&> : std::true_type
{};


} // namespace util
} // namespace ad
