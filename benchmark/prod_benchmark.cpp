#include <random>
#include <ctime>
#include <fastad_bits/node.hpp>
#include <fastad_bits/prod.hpp>
#include <fastad_bits/math.hpp>
#include <fastad_bits/vec.hpp>
#include <fastad_bits/eval.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include "gtest/gtest.h"

namespace ad {

TEST(benchmark, prod) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<long double> dist(0, 1);
    std::uniform_real_distribution<long double> udist(-1e-8, 1e-8);
    std::clock_t time;

    Vec<long double> vec; // set capacity
    vec.emplace_back(udist(gen));
    for (size_t i = 0; i < 1e4 - 1; ++i) {
        vec.emplace_back(dist(gen));
    }

    Var<long double> w4, w5;
    auto&& expr = ad::prod(
        boost::counting_iterator<size_t>(0)
        , boost::counting_iterator<size_t>(vec.size())
        , [&vec](size_t i) {return vec[i] * vec[i]; });

    time = std::clock();
    autodiff((w4 = expr, w5 = w4 * w4 + vec[0]));
    std::cout << "Autodiff only: "
        << 1e3 * (std::clock() - time) / (double)CLOCKS_PER_SEC
        << " ms" << std::endl;

    long double total = 1.0;
    for (size_t i = 0; i < vec.size(); ++i) {
        total *= vec[i].get_value() * vec[i].get_value();
    }
    EXPECT_DOUBLE_EQ(w5.get_value(), total*total + vec[0].get_value());

    for (size_t i = 0; i < vec.size(); ++i) {
        long double correction = (i == 0) ? 1 : 0;
        long double factor = (vec[i].get_value() == 0) ? 0 : 4 * w4.get_value()*w4.get_value() / vec[i].get_value();
        EXPECT_DOUBLE_EQ(vec[i].get_adjoint(), correction + factor);
    }
}

} // namespace ad
