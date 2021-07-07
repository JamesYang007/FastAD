#pragma once
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_adj_view.hpp>
#include <fastad_bits/util/size_pack.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/util/value.hpp>

namespace ad {
namespace core {

/**
 * TransposeNode represents transpose of a matrix or vector.
 * If matrix, then uses Frobenius norm.
 * Currently, we do not support the feature for row vectors.
 * No other shapes are permitted for this node.
 *
 * The node assumes the same value type as that of the vector expression.
 * It is always a scalar shape.
 *
 * @tparam  ExprType     type of vector expression
 */

template <class ExprType>
struct TransposeNode : ValueAdjView<typename util::expr_traits<ExprType>::value_t, ad::mat>,
                       ExprBase<TransposeNode<ExprType>> {
  private:
    using expr_t = ExprType;
    using expr_value_t = typename util::expr_traits<expr_t>::value_t;

    static_assert(!util::is_scl_v<expr_t>);

  public:
    using value_adj_view_t = ValueAdjView<expr_value_t, ad::mat>;
    using typename value_adj_view_t::ptr_pack_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::var_t;

    TransposeNode(const expr_t &expr)
        : value_adj_view_t(nullptr, nullptr, expr.cols(), expr.rows()), expr_{expr} {}

    const var_t &feval() {
        std::cout << "forward..." << std::endl;
        auto &&res = expr_.feval();
        return this->get() = res.transpose();
    }

    template <class T> void beval(const T &seed) {
        // auto &&a_expr = util::to_array(expr_.get());
        // std::cout << "backward..." << std::endl;
        // std::cout << seed << std::endl;
        // std::cout << util::to_array(this->get_adj()) << std::endl;
        util::to_array(this->get_adj()) = seed;
        // std::cout << "backward 1..." << std::endl;
        auto adj = util::to_array(this->get_adj().transpose());
        // std::cout << "backward 2..." << std::endl;
        expr_.beval(adj);
    }

    ptr_pack_t bind_cache(ptr_pack_t begin) {
        begin = expr_.bind_cache(begin);
        return value_adj_view_t::bind(begin);
    };

    util::SizePack bind_cache_size() const {
        return expr_.bind_cache_size() + single_bind_cache_size();
    };

    util::SizePack single_bind_cache_size() const { return {this->size(), this->size()}; }

  private:
    expr_t expr_;
};

} // namespace core

template <class T, class = std::enable_if_t<util::is_convertible_to_ad_v<T> && util::any_ad_v<T>>>
inline auto transpose(const T &x) {
    using expr_t = util::convert_to_ad_t<T>;
    using value_t = typename util::expr_traits<expr_t>::value_t;
    expr_t expr = x;

    // optimization for when expression is constant
    if constexpr (util::is_constant_v<expr_t>) {
        static_assert(!util::is_scl_v<expr_t>);
        using var_t = util::constant_var_t<value_t, ad::scl>;
        var_t out = expr.feval().transpose();
        return ad::constant(out);
    } else {
        return core::TransposeNode<expr_t>(expr);
    }
}

} // namespace ad
