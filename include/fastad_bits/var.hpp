#pragma once
#include <fastad_bits/var_view.hpp>

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
 * @tparam ShapeType    shape of variable (one of scl, vec, mat).
 *                      Default is scl.
 */

template <class ValueType
        , class ShapeType = scl>
struct Var;

template <class ValueType>
struct Var<ValueType, scl>:
    VarView<ValueType, scl>,
    core::ExprBase<Var<ValueType, scl>>
{
private:
    using base_t = VarView<ValueType, scl>;

public:
    using typename base_t::value_t;
    using typename base_t::shape_t;
    using typename base_t::var_t;

    using base_t::feval;
    using base_t::beval;
    using base_t::bind;
    using base_t::bind_adj;
    using base_t::get;
    using base_t::get_adj;
    using base_t::size;
    using base_t::rows;
    using base_t::cols;
    using base_t::data;
    using base_t::data_adj;
    using base_t::reset_adj;
    using base_t::operator=;

    Var()
        : base_t(&val_, &adj_) 
        , val_(0)
        , adj_(0)
    {}

private:
    value_t val_;
    value_t adj_;
};

template <class ValueType>
struct Var<ValueType, vec>:
    VarView<ValueType, vec>,
    core::ExprBase<Var<ValueType, vec>>
{
private:
    using base_t = VarView<ValueType, vec>;
    using vec_t = Eigen::Matrix<
        typename base_t::value_t, Eigen::Dynamic, 1>;

public:
    using typename base_t::value_t;
    using typename base_t::shape_t;
    using typename base_t::var_t;

    using base_t::feval;
    using base_t::beval;
    using base_t::bind;
    using base_t::bind_adj;
    using base_t::get;
    using base_t::get_adj;
    using base_t::size;
    using base_t::rows;
    using base_t::cols;
    using base_t::data;
    using base_t::data_adj;
    using base_t::reset_adj;
    using base_t::operator=;

    Var(size_t size)
        : base_t(nullptr, nullptr, size) 
        , val_(vec_t::Zero(size))
        , adj_(vec_t::Zero(size))
    {
        this->bind(val_.data());
        this->bind_adj(adj_.data());
    }

private:
    vec_t val_;
    vec_t adj_;
};

template <class ValueType>
struct Var<ValueType, mat>:
    VarView<ValueType, mat>,
    core::ExprBase<Var<ValueType, mat>>
{
private:
    using base_t = VarView<ValueType, mat>;
    using mat_t = Eigen::Matrix<
        typename base_t::value_t, Eigen::Dynamic, Eigen::Dynamic>;

public:
    using typename base_t::value_t;
    using typename base_t::shape_t;
    using typename base_t::var_t;

    using base_t::feval;
    using base_t::beval;
    using base_t::bind;
    using base_t::bind_adj;
    using base_t::get;
    using base_t::get_adj;
    using base_t::size;
    using base_t::rows;
    using base_t::cols;
    using base_t::data;
    using base_t::data_adj;
    using base_t::reset_adj;
    using base_t::operator=;

    Var(size_t n_rows, size_t n_cols)
        : base_t(nullptr, nullptr, n_rows, n_cols) 
        , val_(mat_t::Zero(n_rows, n_cols))
        , adj_(mat_t::Zero(n_rows, n_cols))
    {
        this->bind(val_.data());
        this->bind_adj(adj_.data());
    }

private:
    mat_t val_;
    mat_t adj_;
};

} // namespace ad