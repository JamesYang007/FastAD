#pragma once
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_view.hpp>
#include <fastad_bits/util/shape_traits.hpp>
#include <fastad_bits/util/type_traits.hpp>

namespace ad {
namespace core {

template <class Derived>
struct ConstantBase: ExprBase<Derived>
{};

} // namespace core

namespace util {

template <class T>
inline constexpr bool is_constant_v =
    std::is_base_of_v<core::ConstantBase<T>, T>;

namespace details {

template <class ValueType, class ShapeType>
struct shape_to_raw_const_view;

template <class ValueType>
struct shape_to_raw_const_view<ValueType, ad::vec>
{
    using type = Eigen::Map<
        const Eigen::Matrix<ValueType, Eigen::Dynamic, 1>>;
};

template <class ValueType>
struct shape_to_raw_const_view<ValueType, ad::mat>
{
    using type = Eigen::Map<
        const Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic>>;
};

} // namespace details

template <class ValueType, class ShapeType>
using shape_to_raw_const_view_t = typename
    details::shape_to_raw_const_view<ValueType, ShapeType>::type;

} // namespace util

namespace core {

/**
 * ConstantView represents constants in a mathematical formula.
 * Specifically, it treats the values it is viewing as a constant.
 *
 * If shape is selfadjmat, it does not check or symmetrify in any way.
 * It is user's responsibility to make sure the matrix is self adjoint.
 *
 * @tparam  ValueType   underlying data type
 */

template <class ValueType
        , class ShapeType>
struct ConstantView:
    ConstantBase<ConstantView<ValueType, ShapeType>>
{
    using value_t = ValueType;
    using shape_t = ShapeType;
    using var_t = util::shape_to_raw_const_view_t<value_t, shape_t>;

    ConstantView(const value_t* begin,
                 size_t rows,
                 size_t cols)
        : val_(begin, rows, cols)
    {}

    /** 
     * Forward evaluation simply returns the constant value.
     * @return  constant value
     */
    const var_t& feval() const { return this->get(); }

    /**
     * Backward evaluation does nothing.
     */
    void beval(value_t, size_t, size_t, util::beval_policy) const {}

    constexpr size_t bind_size() const { return 0; }
    constexpr size_t single_bind_size() const { return 0; }

    const var_t& get() const { return val_; }
    const value_t& get(size_t i, size_t j) const { return val_(i, j); }

    template <class T>
    constexpr T bind(T begin) const { return begin; }

    size_t size() const { return val_.size(); }
    size_t rows() const { return val_.rows(); }
    size_t cols() const { return val_.cols(); }
    const value_t* data() const { return val_.data(); }

private:
    var_t val_;
};

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

template <class ValueType>
struct constant_var<ValueType, ad::selfadjmat>
{
    using type = Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic>;
};

template <class ValueType, class ShapeType>
using constant_var_t = typename 
    constant_var<ValueType, ShapeType>::type;

} // namespace details

template <class ValueType, class ShapeType>
struct Constant:
    ConstantBase<Constant<ValueType, ShapeType>>
{
private:
    using this_t = Constant<ValueType, ShapeType>;
public:
    using value_t = ValueType;
    using shape_t = ShapeType;
    using var_t = details::constant_var_t<value_t, shape_t>;

    template <class T>
    Constant(const T& c)
        :c_(c)
    {}

    const var_t& feval() const { return c_; }
    void beval(value_t, size_t, size_t, util::beval_policy) const {}
    constexpr size_t bind_size() const { return 0; }
    constexpr size_t single_bind_size() const { return 0; }

    const var_t& get() const { return c_; }
    const value_t& get(size_t i, size_t j) const { 
        if constexpr (util::is_scl_v<this_t>) {
            static_cast<void>(i);
            static_cast<void>(j);
            return c_;
        } else {
            return c_(i,j); 
        }
    }

    template <class T>
    constexpr T bind(T begin) const { return begin; }

    const value_t* data() const { 
        if constexpr (util::is_scl_v<this_t>) {
            return &c_;
        } else {
            return c_.data(); 
        }
    }

    size_t rows() const { 
        if constexpr (util::is_scl_v<this_t>) {
            return 1;
        } else {
            return c_.rows(); 
        }
    }

    size_t cols() const { 
        if constexpr (util::is_scl_v<this_t>) {
            return 1;
        } else {
            return c_.cols(); 
        }
    }

private:
    var_t c_;
};

} // namespace core

// Helper function: 
// ad::constant(...) and ad::constant_view(...)

template <class ValueType>
inline auto constant_view(const ValueType* x,
                          size_t rows)
{
    return core::ConstantView<ValueType, ad::vec>(x, rows, 1);
}

template <class ShapeType = ad::mat, class ValueType>
inline auto constant_view(const ValueType* x,
                          size_t rows,
                          size_t cols)
{
    return core::ConstantView<ValueType, ShapeType>(x, rows, cols);
}

template <class ValueType
        , class = std::enable_if_t<std::is_arithmetic_v<ValueType>> >
inline auto constant(ValueType x)
{
    return core::Constant<ValueType, ad::scl>(x);
}

template <class Derived
        , class = std::enable_if_t<util::is_eigen_vector_v<Derived>> >
inline auto constant(const Eigen::EigenBase<Derived>& x)
{
    using value_t = typename Derived::Scalar;
    return core::Constant<value_t, ad::vec>(x);
}

/** 
 * By default, uses ad::mat as shape, but if user knows and
 * wishes to treat the matrix as a self-adjoint matrix, they can 
 * specify the shape type to be ad::selfadjmat.
 */
template <class ShapeType = ad::mat
        , class Derived
        , class = std::enable_if_t<util::is_eigen_matrix_v<Derived>> >
inline auto constant(const Eigen::EigenBase<Derived>& x)
{
    using value_t = typename Derived::Scalar;
    return core::Constant<value_t, ShapeType>(x);
}

} // namespace ad
