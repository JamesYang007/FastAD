#pragma once
#include <cassert>
#include <fastad_bits/util/shape_traits.hpp>

namespace ad {
namespace core {

template <class ValueType, class ShapeType>
struct ValueView;

template <class ValueType>
struct ValueView<ValueType, scl>
{
    using value_t = ValueType;
    using shape_t = scl;
    using var_t = value_t;

    ValueView(value_t* begin, size_t=1, size_t=1)
        : val_(begin)
    {}

    /**
     * Get underlying value as writeable.
     * @return  reference to underlying value.
     */
    var_t& get() {
        assert(val_);
        return *val_;
    }

    /**
     * Get underlying value as read-only.
     * @return  const reference to underlying value.
     */
    const var_t& get() const {
        assert(val_);
        return *val_;
    }

    /**
     * Get underlying value. This is for consistent API.
     */
    value_t& get(size_t, size_t) {
        assert(val_);
        return *val_;
    }
     
    const value_t& get(size_t, size_t) const {
        assert(val_);
        return *val_;
    }

    /**
     * Binds value pointer to view the same value that val_begin points to.
     * @return  the next pointer from val_begin that is not viewed by current object.
     */
    value_t* bind(value_t* begin)
    { val_ = begin; return val_ + this->size(); }

    /**
     * Returns the raw pointer to the first value viewed by current object.
     * @return  raw pointer
     */
    value_t* data() { return val_; }
    const value_t* data() const { return val_; }

    /**
     * Returns the size of the variable.
     * @return  size of variable.
     */
    constexpr size_t size() const { return 1; }

    /**
     * Returns number of rows of the variable.
     * A scalar and a row vector are defined to have 1 row.
     * @return  number of rows
     */
    constexpr size_t rows() const { return 1; }

    /**
     * Returns number of cols of the variable.
     * A scalar and a column vector are defined to have 1 col.
     * @return  number of cols
     */
    constexpr size_t cols() const { return 1; }

    /**
     * Zero out the underlying value(s).
     */
    void zero() { 
        assert(val_);
        *val_ = 0;
    }

    /**
     * One out the underlying value(s).
     */
    void ones() { 
        assert(val_);
        *val_ = 1;
    }

private:
    value_t* val_;
};

template <class ValueType>
struct ValueView<ValueType, vec>
{
    using value_t = ValueType;
    using shape_t = vec;
    using var_t = util::shape_to_raw_view_t<value_t, shape_t>;

    ValueView(value_t* begin, size_t rows, size_t=1)
        : val_(begin, rows)
    {}
     
    var_t& get() { return val_; }
    const var_t& get() const { return val_; }
    value_t& get(size_t i, size_t) { return val_(i); }
    const value_t& get(size_t i, size_t) const { return val_(i); }

    value_t* bind(value_t* begin)
    { 
        new (&val_) var_t(begin, this->size());
        return begin + this->size(); 
    }

    constexpr size_t size() const { return val_.size(); }
    constexpr size_t rows() const { return this->size(); }
    constexpr size_t cols() const { return 1; }
    value_t* data() { return val_.data(); }
    const value_t* data() const { return val_.data(); }
    void zero() { val_.setZero(); }
    void ones() { val_.setOnes(); }

private:
    var_t val_;
};

template <class ValueType>
struct ValueView<ValueType, mat>
{
    using value_t = ValueType;
    using shape_t = mat;
    using var_t = util::shape_to_raw_view_t<value_t, shape_t>;

    ValueView(value_t* begin, size_t rows, size_t cols)
        : val_(begin, rows, cols)
    {}
     
    var_t& get() { return val_; }
    const var_t& get() const { return val_; }
    value_t& get(size_t i, size_t j) { return val_(i,j); }
    const value_t& get(size_t i, size_t j) const { return val_(i,j); }

    value_t* bind(value_t* begin)
    { 
        new (&val_) var_t(begin, this->rows(), this->cols());
        return begin + this->size(); 
    }

    constexpr size_t size() const { return val_.size(); }
    constexpr size_t rows() const { return val_.rows(); }
    constexpr size_t cols() const { return val_.cols(); }
    value_t* data() { return val_.data(); }
    const value_t* data() const { return val_.data(); }
    void zero() { val_.setZero(); }
    void ones() { val_.setOnes(); }

private:
    var_t val_;
};

} // namespace core
} // namespace ad
