#include <random>
#include <fastad_bits/node.hpp>
#include <fastad_bits/math.hpp>
#include <fastad_bits/vec.hpp>
#include <fastad_bits/eval.hpp>
#include "gtest/gtest.h"

namespace ad {

TEST(benchmark, foreach) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<long double> dist(0,1);
    std::uniform_real_distribution<long double> udist(-1e-8, 1e-8);

    Vec<long double> vec(0); // set capacity
    vec.emplace_back(udist(gen));
    for (size_t i = 0; i < 1e1; ++i) {
        vec.emplace_back(dist(gen));        
    }

    Var<long double> w4, w5;
    Vec<long double> prod(vec.size());
    prod[0] = vec[0];

    auto vec_it = vec.begin();
    auto prod_prev = prod.begin();
    auto expr = ad::for_each(std::next(prod.begin()), prod.end()
            , [&](const Vec<long double>::value_type& curr) {
                return curr = *(prod_prev++) * *(++vec_it);
            });
    autodiff((w4=expr, w5 = w4*w4 + vec[0]));

    long double total = 1.0;
    for (size_t i = 0; i < vec.size(); ++i) {
        total *= vec[i].get_value();
    }
    EXPECT_DOUBLE_EQ(w5.get_value(), total*total + vec[0].get_value());

    for (size_t i = 0; i < vec.size(); ++i) {
        long double correction = (i==0) ? 1:0;
        long double factor = (vec[i].get_value() == 0) ? 0:2*w4.get_value()*w4.get_value()/vec[i].get_value();
        EXPECT_DOUBLE_EQ(vec[i].get_adjoint(), correction + factor);
    }
}

} // namespace ad
