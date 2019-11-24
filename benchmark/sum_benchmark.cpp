#include <random>
#include <ctime>
#include <fastad_bits/vec.hpp>
#include <fastad_bits/math.hpp>
#include <fastad_bits/node.hpp>
#include <fastad_bits/eval.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include "gtest/gtest.h"

namespace ad {

TEST(benchmark, sumnode) {
    Vec<double> vec;
    std::default_random_engine gen;
    std::normal_distribution<double> dist(0.0,1.0);
    for (size_t i = 0; i < 1e6; ++i) {
        vec.emplace_back(dist(gen));        
    }

    Vec<double> sumvec(vec.size());
    sumvec[0] = vec[0];

    Var<double> w4, w5;

    auto&& expr = ad::for_each(
            boost::counting_iterator<size_t>(1)
            , boost::counting_iterator<size_t>(vec.size())
            , [&](size_t i) {
                return sumvec[i] = sumvec[i-1] + vec[i];
            });

    std::clock_t time;
    time = std::clock();
    autodiff((w4=expr, w5 = w4*w4 + vec[0]));
    std::cout << "Autodiff only: " 
        << 1e3 * (std::clock() - time) / (double) CLOCKS_PER_SEC
        << " ms" << std::endl;

    double sqsum = 0.0;
    for (size_t i = 0; i < vec.size(); ++i) {
        sqsum += vec[i].get_value();
    }
    EXPECT_DOUBLE_EQ(w5.get_value(), sqsum*sqsum + vec[0].get_value());

}

} // namespace ad
