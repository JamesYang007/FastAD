#pragma once
#include <iterator>
#include <tuple>

namespace utils {

// type_traits extension

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

} // namespace utils
