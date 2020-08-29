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

namespace details {

template <class T, class = std::void_t<>>
struct get_shape 
{
    using type = void;
};

template <class T>
struct get_shape<T, std::void_t<typename T::shape_t>>
{
    using type = typename T::shape_t;
};

template <class T>
using get_shape_t = typename get_shape<T>::type;

} // namespace details

template <class T>
inline constexpr bool is_scl_v =
    std::is_same_v<details::get_shape_t<T>,
                   scl>;

template <class T>
inline constexpr bool is_vec_v =
    std::is_same_v<details::get_shape_t<T>,
                   vec>;

template <class T>
inline constexpr bool is_mat_v =
    std::is_base_of_v<mat, details::get_shape_t<T>>;

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
 *
 * If one of the shapes is a mat, then automatically the result is mat.
 * Otherwise, choose the biggest sized shape.
 *
 * (FOLLOWING DEPRECATED: selfadjmat is deprecated) 
 * This is so that for cases when one is a mat and the other is selfadjmat, the result is mat,
 * and when both are selfadjmats, then the result is selfadjmat.
 */

template <class T1, class T2>
using max_shape_t = std::conditional_t<
    std::is_same_v<T1, mat> ||
    std::is_same_v<T2, mat>,
    mat,
    std::conditional_t<
        (T1::dim > T2::dim), T1, T2
    >
>;

} // namespace util
} // namespace ad
