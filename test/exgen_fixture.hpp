#pragma once
#include <fastad_bits/vec.hpp>

namespace ad {
namespace core {

auto f_lmda_no_opt = [](const Vec<double>& x, const Vec<double>& w) {
    auto&& eq1 = make_eq(w[0], x[0]);
    auto&& eq2 = make_eq(w[1], w[0]);
    return make_glue<double>(eq1, eq2);
};

auto f_lmda_opt = [](const Vec<double>& x, const Vec<double>& w) {
    auto&& eq1 = make_eq(w[0], x[0]);
    auto&& eq2 = make_eq(w[2], w[0]);
    return make_glue<double>(eq1, eq2);
};

} // namespace core
} // namespace ad
