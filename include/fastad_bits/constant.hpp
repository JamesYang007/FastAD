#pragma once
#include <fastad_bits/expr_base.hpp>
#include <fastad_bits/value_view.hpp>
#include <fastad_bits/shape_traits.hpp>

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

} // namespace util

namespace core {

/**
 * ConstantView represents constants in a mathematical formula.
 * Specifically, it treats the values it is viewing as a constant.
 *
 * @tparam  ValueType   underlying data type
 */

template <class ValueType
        , class ShapeType>
struct ConstantView:
    ValueView<ValueType, ShapeType>,
    ConstantBase<ConstantView<ValueType, ShapeType>>
{
private:
    static_assert(std::is_same_v<ShapeType, ad::scl> ||
                  std::is_same_v<ShapeType, ad::vec> ||
                  std::is_same_v<ShapeType, ad::mat>);

public:
    using value_view_t = ValueView<ValueType, ShapeType>;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;

    ConstantView(value_t* begin,
                 size_t rows,
                 size_t cols)
        : value_view_t(begin, rows, cols)
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

    /**
     * No binding required for constants
     */
    constexpr size_t bind_size() const { return 0; }
    constexpr size_t single_bind_size() const { return 0; }
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

template <class ValueType, class ShapeType>
using constant_var_t = typename 
    constant_var<ValueType, ShapeType>::type;

} // namespace details

template <class ValueType, class ShapeType>
struct Constant:
    ConstantBase<Constant<ValueType, ShapeType>>
{
    using value_t = ValueType;
    using shape_t = ShapeType;
    using var_t = details::constant_var_t<value_t, shape_t>;

    Constant(const var_t& c)
        :c_(c)
    {}

    const var_t& feval() const { return c_; }
    void beval(value_t, size_t, size_t) const {}
    constexpr size_t bind_size() const { return 0; }
    constexpr size_t single_bind_size() const { return 0; }

private:
    var_t c_;
};

} // namespace core

// Helper function: 
// ad::constant(...) and ad::constant_view(...)

template <class ValueType>
inline auto constant_view(const ValueType* x)
{
    return core::ConstantView<ValueType, ad::scl>(x);
}

template <class ValueType>
inline auto constant_view(const ValueType* x,
                          size_t rows)
{
    return core::ConstantView<ValueType, ad::vec>(x, rows);
}

template <class ValueType>
inline auto constant_view(const ValueType* x,
                          size_t rows,
                          size_t cols)
{
    return core::ConstantView<ValueType, ad::mat>(x, rows, cols);
}

template <class ValueType>
inline auto constant(ValueType x)
{
    static_assert(std::is_arithmetic_v<ValueType>);
    return core::Constant<ValueType, ad::scl>(x);
}

template <class ValueType>
inline auto constant(const Eigen::Matrix<ValueType, Eigen::Dynamic, 1>& x)
{
    return core::Constant<ValueType, ad::vec>(x);
}

template <class ValueType>
inline auto constant(const Eigen::Matrix<
                                ValueType, 
                                Eigen::Dynamic, 
                                Eigen::Dynamic>& x)
{
    return core::Constant<ValueType, ad::mat>(x);
}


} // namespace ad
