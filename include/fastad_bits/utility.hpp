#pragma once
#include <iterator>

#ifdef USE_ARMA

#include <armadillo>

#endif

namespace utils {

// type_traits extension

// Dummy function used to SFINAE on return value.
// Checks if type T is pointer-like (dereferenceable)
template <typename T>
auto is_pointer_like()
    -> decltype(*std::declval<T>(), std::true_type{});

template <typename T>
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

#ifdef USE_ARMA

namespace details {

template <class T>
struct is_arma_mat : std::false_type
{};

template <class T>
struct is_arma_mat<::arma::Mat<T>> : std::true_type
{};

} // namespace details

template <class T>
inline constexpr bool is_arma_mat = details::is_arma_mat<T>::value;

#endif 

} // namespace utils
