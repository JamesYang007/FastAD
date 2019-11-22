#include <fastad_bits/adforward.hpp>
#include "gtest/gtest.h"

namespace {

// Test trigonometric functions
TEST(forwardvar, trigonometry) {
    using namespace ad;
    ForwardVar<double> w[4] = { 1.3, -2.0, 1.4, -0.230041 };
    w[0].df = 1;
    auto res = -w[0] * ad::sin(w[1]) + ad::cos(w[2]) - tan(w[3]);
    EXPECT_DOUBLE_EQ(res.w, -w[0].w * std::sin(w[1].w) + std::cos(w[2].w) - std::tan(w[3].w));
    EXPECT_DOUBLE_EQ(res.df, -std::sin(w[1].w));
}

// Test exp, log
TEST(forwardvar, power) {
    using namespace ad;
    ForwardVar<double> w[] = { 0.002, 0.5, -3 };
    w[1].df = 1;
    auto res = exp(w[0]) / log(w[1]) + w[2] * acos(w[1]);
    EXPECT_DOUBLE_EQ(res.w, std::exp(w[0].w) / std::log(w[1].w) + w[2].w * std::acos(w[1].w));
    EXPECT_DOUBLE_EQ(res.df,
        -std::exp(w[0].w) / (std::log(w[1].w) * std::log(w[1].w) * w[1].w)
        - w[2].w / std::sqrt(1 - w[1].w*w[1].w));
}

} // end namespace
