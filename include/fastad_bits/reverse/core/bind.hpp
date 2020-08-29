#pragma once
#include <vector>
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/util/type_traits.hpp>

namespace ad {
namespace core {

/**
 * ExprBind is a helper class that wraps an AD expression
 * and binds it with an internal cache for temporaries.
 * This is for convenience purposes so that users do not have
 * to worry about creating the cache line themselves.
 *
 * @tparam  ExprType    expression type
 */

template <class ExprType>
struct ExprBind
{
    using expr_t = ExprType;
    using value_t = typename util::expr_traits<expr_t>::value_t;

    ExprBind(const expr_t& expr)
        : expr_{expr}
        , val_cache_()
        , adj_cache_()
    {
        auto size_pack = expr_.bind_cache_size();
        val_cache_.resize(size_pack(0));
        adj_cache_.resize(size_pack(1));
        expr_.bind_cache({val_cache_.data(), adj_cache_.data()});
    }
    
    expr_t& get() { return expr_; }

private:
    expr_t expr_; 
    Eigen::Matrix<value_t, Eigen::Dynamic, 1> val_cache_;
    Eigen::Matrix<value_t, Eigen::Dynamic, 1> adj_cache_;
};

} // namespace core

template <class Derived>
inline auto bind(const core::ExprBase<Derived>& expr)
{
    return core::ExprBind<Derived>(expr.self());
}

} // namespace ad
