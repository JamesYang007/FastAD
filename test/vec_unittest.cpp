#include <fastad_bits/vec.hpp>
#include "gtest/gtest.h"

namespace ad {

struct advec_fixture: ::testing::Test
{
protected:
    double xs[2] = {1.0, 2.0};
    double df[2] = {3., 4.};
    Vec<double> vec;

    advec_fixture()
        : vec({xs[0], xs[1]}, df)
    {}
};

// Both xi's and dfs
TEST_F(advec_fixture, constructor) {
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(vec[i].get_value(), xs[i]);
        EXPECT_EQ(vec[i].get_adjoint(), df[i]);
    }
}

TEST_F(advec_fixture, constructor_iter) {
    double x[] = {-1., -3., -5.};
    Vec<double> v(x, x + 3);
    EXPECT_DOUBLE_EQ(v[0].get_value(), -1.);
    EXPECT_DOUBLE_EQ(v[1].get_value(), -3.);
    EXPECT_DOUBLE_EQ(v[2].get_value(), -5.);
}

TEST_F(advec_fixture, reset_adjoint) {
    vec.reset_adjoint();
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(vec[i].get_value(), xs[i]);   // value did not change
        EXPECT_EQ(vec[i].get_adjoint(), 0.); // adjoint is 0
    }
}

} // namespace 
