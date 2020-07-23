#pragma once
#include <type_traits>
#include <Eigen/Core>

namespace ad {

struct scl { static constexpr size_t dim = 0; };
struct vec { static constexpr size_t dim = 1; };
struct mat { static constexpr size_t dim = 2; };

namespace util {

template <class T>
struct shape_traits
{
    using shape_t = typename T::shape_t;
};

template <class T>
inline constexpr bool is_scl_v =
    std::is_same_v<typename shape_traits<T>::shape_t,
                   scl>;

template <class T>
inline constexpr bool is_vec_v =
    std::is_same_v<typename shape_traits<T>::shape_t,
                   vec>;

template <class T>
inline constexpr bool is_mat_v =
    std::is_same_v<typename shape_traits<T>::shape_t,
                   mat>;


/**
 * Defines a mapping from shape tags to corresponding
 * Eigen::Map/scalar viewers.
 *
 * scl -> T*
 * vec -> Map<Matrix<T, Dynamic, 1>>
 * mat -> Map<Matrix<T, Dynamic, Dynamic>>
 */
namespace details {

template <class T, class ShapeType>
struct shape_to_raw_view;

template <class T>
struct shape_to_raw_view<T, scl>
{
    using type = T*;
};

template <class T>
struct shape_to_raw_view<T, vec>
{
    using type = Eigen::Map<
        Eigen::Matrix<T, Eigen::Dynamic, 1> >;
};

template <class T>
struct shape_to_raw_view<T, mat>
{
    using type = Eigen::Map<
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >;
};

} // namespace details

template <class T, class ShapeType>
using shape_to_raw_view_t = typename
    details::shape_to_raw_view<T, ShapeType>::type;

/**
 * Finds the max of the two shapes based on their dimensions.
 */
template <class T1, class T2>
using max_shape_t = std::conditional_t<
    (T1::dim > T2::dim), T1, T2
    >;

} // namespace util
} // namespace ad
