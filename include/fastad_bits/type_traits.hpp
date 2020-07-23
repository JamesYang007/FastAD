#pragma once
#include <Eigen/Core>
#include <type_traits>
#include <iterator>
#include <tuple>

namespace ad {
namespace util {

/*
 * Metaprogramming tools to check if type T is Eigen matrix.
 */
template <class T>
inline constexpr bool is_eigen_matrix_v =
    std::is_base_of_v<Eigen::EigenBase<T>, T>;

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
