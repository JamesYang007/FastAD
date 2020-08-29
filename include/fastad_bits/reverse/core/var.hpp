#pragma once
#include <fastad_bits/reverse/core/var_view.hpp>

namespace ad {

/* 
 * Var is a variable, which could be a scalar, vector, or matrix.
 * Var objects are VarView, since they view themselves.
 * Var objects own the variable value(s) and partial derivative(s), or adjoint(s).
 *
 * ShapeType must be one of scl, vec, or mat.
 * All other specializations are disabled (see VarView).
 *
 * @tparam ValueType    underlying data type
 * @tparam ShapeType    shape of variable (one of scl, vec, mat, selfadjmat).
 *                      Default is scl.
 */

template <class ValueType
        , class ShapeType = scl>
struct Var;

template <class ValueType>
struct Var<ValueType, scl>:
    VarView<ValueType, scl>
{
private:
    using base_t = VarView<ValueType, scl>;

public:
    using typename base_t::value_t;
    using typename base_t::shape_t;
    using typename base_t::var_t;
    using base_t::operator=;

    Var()
        : base_t(&val_, &adj_) 
        , val_(0)
        , adj_(0)
    {}

    explicit Var(value_t v)
        : base_t(&val_, &adj_)
        , val_(v)
        , adj_(0)
    {}

    Var(const Var& v)
        : base_t(v)
        , val_(v.val_)
        , adj_(v.adj_)
    { rebind(); }

    Var(Var&& v)
        : base_t(std::move(v))
        , val_(std::move(v.val_))
        , adj_(std::move(v.adj_))
    { rebind(); }

    Var& operator=(const Var& v)
    {
        if (this == &v) return *this;
        assert(v.rows() == this->rows());
        assert(v.cols() == this->cols());
        val_ = v.val_;
        adj_ = v.adj_;
        rebind();
        return *this;
    }

    Var& operator=(Var&& v) 
    {
        if (this == &v) return *this;
        assert(v.rows() == this->rows());
        assert(v.cols() == this->cols());
        val_ = std::move(v.val_);
        adj_ = std::move(v.adj_);
        rebind();
        return *this;
    }

private:
    void rebind() 
    {
        this->bind({&val_, &adj_});
    }

    value_t val_;
    value_t adj_;
};

template <class ValueType>
struct Var<ValueType, vec>:
    VarView<ValueType, vec>
{
private:
    using base_t = VarView<ValueType, vec>;
    using vec_t = Eigen::Matrix<
        typename base_t::value_t, Eigen::Dynamic, 1>;

public:
    using typename base_t::value_t;
    using typename base_t::shape_t;
    using typename base_t::var_t;
    using base_t::operator=;

    explicit Var(size_t size)
        : base_t(nullptr, nullptr, size) 
        , val_(vec_t::Zero(size))
        , adj_(vec_t::Zero(size))
    { rebind(); }

    Var(const Var& v)
        : base_t(v)
        , val_(v.val_)
        , adj_(v.adj_)
    { rebind(); }

    Var(Var&& v)
        : base_t(std::move(v))
        , val_(std::move(v.val_))
        , adj_(std::move(v.adj_))
    { rebind(); }

    Var& operator=(const Var& v)
    {
        if (this == &v) return *this;
        assert(v.rows() == this->rows());
        assert(v.cols() == this->cols());
        val_ = v.val_;
        adj_ = v.adj_;
        rebind();
        return *this;
    }

    Var& operator=(Var&& v) 
    {
        if (this == &v) return *this;
        assert(v.rows() == this->rows());
        assert(v.cols() == this->cols());
        val_ = std::move(v.val_);
        adj_ = std::move(v.adj_);
        rebind();
        return *this;
    }

private:
    void rebind() 
    {
        this->bind({val_.data(), adj_.data()});
    }

    vec_t val_;
    vec_t adj_;
};

template <class ValueType>
struct Var<ValueType, mat>:
    VarView<ValueType, mat>
{
private:
    using base_t = VarView<ValueType, mat>;
    using mat_t = Eigen::Matrix<
        typename base_t::value_t, Eigen::Dynamic, Eigen::Dynamic>;

public:
    using typename base_t::value_t;
    using typename base_t::shape_t;
    using typename base_t::var_t;
    using base_t::operator=;

    explicit Var(size_t n_rows, size_t n_cols)
        : base_t(nullptr, nullptr, n_rows, n_cols) 
        , val_(mat_t::Zero(n_rows, n_cols))
        , adj_(mat_t::Zero(n_rows, n_cols))
    { rebind(); }

    Var(const Var& v)
        : base_t(v)
        , val_(v.val_)
        , adj_(v.adj_)
    { rebind(); }

    Var(Var&& v)
        : base_t(std::move(v))
        , val_(std::move(v.val_))
        , adj_(std::move(v.adj_))
    { rebind(); }

    Var& operator=(const Var& v)
    {
        if (this == &v) return *this;
        assert(v.rows() == this->rows());
        assert(v.cols() == this->cols());
        val_ = v.val_;
        adj_ = v.adj_;
        rebind();
        return *this;
    }

    Var& operator=(Var&& v) 
    {
        if (this == &v) return *this;
        assert(v.rows() == this->rows());
        assert(v.cols() == this->cols());
        val_ = std::move(v.val_);
        adj_ = std::move(v.adj_);
        rebind();
        return *this;
    }

private:
    void rebind() 
    {
        this->bind({val_.data(), adj_.data()});
    }

    mat_t val_;
    mat_t adj_;
};

template struct Var<double, scl>;
template struct Var<double, vec>;
template struct Var<double, mat>;

} // namespace ad
